import os

import yaml
import pandas as pd
import numpy as np
from cnc_genai.src.data_proc.sensor_proc import (
    _find_time_baseline,
    _match,
    _post_process,
    process_sensor_data,
    retain_max_occurrence_rows,
    match_sensor_code,
)
from cnc_genai.src.v1_algo.experiment_data_proc.data_prep import load_sensor


def _find_sensor_startline(df, num_points=3, rapid_speed=10000):

    # 滑动窗口计算是否有连续 N 个 True
    mask = (df["f_speed"] > 0) & (df["f_speed"] < rapid_speed)
    rolling = mask.rolling(window=num_points).sum()

    # 找到第一次连续 N 个满足条件的位置
    index = (
        rolling[rolling == num_points].index.min() - (num_points - 1)
        if not rolling[rolling == num_points].empty
        else None
    )
    return index


def _find_code_startline(code_df):
    index = (
        code_df[code_df["move_code"].isin(["G01", "G02", "G03"])]
        .head(1)["row_id"]
        .values[0]
    )
    return index


def _match_sensors_data_subprogram(
    sensor_df, sensor_df_opt, idx, sub_program, time_tolerance=0.2, dist_tolerance=5
):

    # 篩選子程序
    print(idx, sub_program)
    sensor_df = sensor_df[sensor_df["processing_code"].astype(str) == sub_program]
    sensor_df = sensor_df.reset_index(drop=True).reset_index()

    sensor_df_opt = sensor_df_opt[
        sensor_df_opt["processing_code"].astype(str) == sub_program
    ]
    sensor_df_opt = sensor_df_opt.reset_index(drop=True).reset_index()

    # 刀具補償
    sensor_df["H"] = (
        sensor_df["tool_cpn"].str.split("/").str[0].str.split(":").str[1].astype(int)
    )
    sensor_df["Z"] = sensor_df["Z"] - sensor_df["H"]

    sensor_df_opt["H"] = (
        sensor_df_opt["tool_cpn"]
        .str.split("/")
        .str[0]
        .str.split(":")
        .str[1]
        .astype(int)
    )
    sensor_df_opt["Z"] = sensor_df_opt["Z"] - sensor_df_opt["H"]

    # 尋找時間基準，即機械坐標0位置的最後一行（開始下刀時刻）
    starting_line_idx_sensor = _find_sensor_startline(sensor_df)
    starting_line_idx_sensor_opt = _find_sensor_startline(sensor_df_opt)

    # 計算相對時間
    if (
        starting_line_idx_sensor is not None
        and starting_line_idx_sensor_opt is not None
    ):
        time_baseline_sensor = sensor_df.loc[starting_line_idx_sensor, "datetime"]
        time_baseline_sensor_opt = sensor_df_opt.loc[
            starting_line_idx_sensor_opt, "datetime"
        ]
    else:
        time_baseline_sensor = pd.to_datetime(
            sensor_df["datetime"].min(), "%Y-%m-%d %H:%M:%S"
        )
        time_baseline_sensor_opt = pd.to_datetime(
            sensor_df_opt["datetime"].min(), "%Y-%m-%d %H:%M:%S"
        )

    # 計算相對時間比例
    sensor_df["time_acc"] = (
        pd.to_datetime(sensor_df["datetime"], "%Y-%m-%d %H:%M:%S")
        - time_baseline_sensor
    ).dt.total_seconds()
    sensor_df["time_acc_perc"] = (
        sensor_df["time_acc"] / sensor_df["time_acc"].to_list()[-1]
    )

    sensor_df_opt["time_acc"] = (
        pd.to_datetime(sensor_df_opt["datetime"], "%Y-%m-%d %H:%M:%S")
        - time_baseline_sensor_opt
    ).dt.total_seconds()
    sensor_df_opt["time_acc_perc"] = (
        sensor_df_opt["time_acc"] / sensor_df_opt["time_acc"].to_list()[-1]
    )

    # 轉換到sensor_df坐標系
    sensor_df_opt["X_translated"] = (
        sensor_df_opt["X"]
        + sensor_df.loc[starting_line_idx_sensor, "X"]
        - sensor_df_opt.loc[starting_line_idx_sensor_opt, "X"]
    )
    sensor_df_opt["Y_translated"] = (
        sensor_df_opt["Y"]
        + sensor_df.loc[starting_line_idx_sensor, "Y"]
        - sensor_df_opt.loc[starting_line_idx_sensor_opt, "Y"]
    )
    sensor_df_opt["Z_translated"] = sensor_df_opt["Z"]
    # sensor_df_opt['Z_translated'] = (
    #     sensor_df_opt['Z']
    #     + sensor_df.loc[starting_line_idx_sensor-1, 'Z']
    #     - sensor_df_opt.loc[starting_line_idx_sensor_opt-1, 'Z']
    # )

    # 遍历 sensor_df 中的每一行
    matches = []
    for i, row in sensor_df.iterrows():
        time, x, y, z = row["time_acc_perc"], row["X"], row["Y"], row["Z"]

        # 计算时间差和欧氏距离
        time_diff = np.abs(sensor_df_opt["time_acc_perc"] - time)
        distance = np.sqrt(
            (sensor_df_opt["X_translated"] - x) ** 2
            + (sensor_df_opt["Y_translated"] - y) ** 2
            + (sensor_df_opt["Z_translated"] - z) ** 2
        )

        # 筛选时间相距time_tolerance，位置相距dist_tolerance的行
        time_valid = time_diff <= time_tolerance
        dist_valid = distance <= dist_tolerance
        valid = time_valid * dist_valid
        if valid.any():
            closest_idx = valid[valid].index[np.argmin(distance[valid])]
            matches.append((i, closest_idx))
        else:
            matches.append((i, None))

    # 遍历 sensor_df_opt 中的未匹配行
    unmatched = set(sensor_df_opt.index) - {
        match[1] for match in matches if match[1] is not None
    }
    for opt_idx in unmatched:
        matches.append((None, opt_idx))

    # 分开处理 sensor_df 和 sensor_df_opt 的匹配结果
    sensor_matches = sorted(
        [m for m in matches if m[0] is not None], key=lambda x: x[0]
    )
    opt_matches = sorted(
        [m for m in matches if m[1] is not None and m[0] is None], key=lambda x: x[1]
    )

    # 合并结果，保持 sensor_df 和 sensor_df_opt 的索引顺序
    sorted_matches = sensor_matches + opt_matches

    # 构建最终结果表
    result = []
    for sensor_idx, opt_idx in sorted_matches:
        row_sensor = (
            sensor_df.loc[sensor_idx]
            if sensor_idx is not None
            else pd.Series(dtype="float64")
        )
        row_opt = (
            sensor_df_opt.loc[opt_idx]
            if opt_idx is not None
            else pd.Series(dtype="float64")
        )
        merged_row = pd.concat(
            [row_sensor, row_opt], keys=["sensor_df", "sensor_df_opt"]
        )
        result.append(merged_row)

    output = pd.DataFrame(result)
    output.columns = ["_".join(map(str, col)) for col in output.columns]
    output["combined_acc_time_perc"] = (
        (
            (output["sensor_df_time_acc_perc"] + output["sensor_df_opt_time_acc_perc"])
            / 2
        )
        .fillna(output["sensor_df_time_acc_perc"])
        .fillna(output["sensor_df_opt_time_acc_perc"])
    )
    output = output.sort_values("combined_acc_time_perc")
    output["matched"] = (
        output["sensor_df_time_acc_perc"].notna()
        & output["sensor_df_opt_time_acc_perc"].notna()
    )
    output["sub_program"] = "_".join([str(idx).zfill(2), sub_program])
    return output


def match_sensor_data(
    conf_ct,
    conf_datapath,
    time_tolerance=0.2,
    dist_tolerance=5,
    sensor_ip="172.16.226.77",
    sensor_opt_ip="172.16.227.33",
):

    sub_programs = conf_ct["sub_programs"]
    df_sensor_dict = load_sensor(conf_datapath)

    # 預處理
    sensor_df_full = process_sensor_data(
        df_sensor_dict["df_sensor"], sub_programs, device_ip_to_keep=sensor_ip
    )
    sensor_df_full = retain_max_occurrence_rows(sensor_df_full)

    sensor_df_opt_full = process_sensor_data(
        df_sensor_dict["df_sensor_opt"], sub_programs, device_ip_to_keep=sensor_opt_ip
    )
    sensor_df_opt_full = retain_max_occurrence_rows(sensor_df_opt_full)

    # 遍歷每個子程序
    dfs_res = []
    for idx, sub_program in enumerate(sub_programs, start=1):
        output = _match_sensors_data_subprogram(
            sensor_df_full,
            sensor_df_opt_full,
            idx,
            sub_program,
            time_tolerance,
            dist_tolerance,
        )
        os.makedirs(conf_ct["output_path"]["sensors_match_out"], exist_ok=True)
        output.to_excel(
            f'{conf_ct["output_path"]["sensors_match_out"]}/{str(idx).zfill(2)}-{sub_program}.xlsx',
            index=False,
        )
        dfs_res.append(output)
    output_all = pd.concat(dfs_res, axis=0)
    output_all.to_excel(
        f'{conf_ct["output_path"]["sensors_match_out"]}/all.xlsx', index=False
    )


if __name__ == "__main__":
    from cnc_genai.src.utils import utils

    conf = utils.load_config(base_config_path="cnc_genai/conf/v1_config.yaml")

    with open("cnc_genai/conf/data_path.yaml", "r") as file:
        conf_datapath = yaml.safe_load(file)

    match_sensor_data(conf, conf_datapath)
