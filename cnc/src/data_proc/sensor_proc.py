import os
import numpy as np
import pandas as pd
import math
import warnings
import yaml

from cnc_genai.src.v1_algo.calculate_load_time import run_calculate_cycle_time
from cnc_genai.src.v1_algo.load_analysis import run_analysis
from cnc_genai.src.v1_algo.experiment_data_proc.data_prep import load_sensor

warnings.filterwarnings("ignore")


def qu_tou_qu_wei(df):
    # Remove the first block of rows corresponding to the first processing_code for each deviceIP
    df.reset_index(inplace=True, drop=True)
    first_processing_code = df.iloc[0]["processing_code"]
    first_block_end_index = df[df["processing_code"] != first_processing_code].index[0]
    df = df.iloc[first_block_end_index:]

    # Remove the last block of rows corresponding to the last processing_code for each deviceIP
    df.reset_index(inplace=True, drop=True)
    last_processing_code = df.iloc[-1]["processing_code"]
    last_block_start_index = df[df["processing_code"] != last_processing_code].index[-1]
    df = df.iloc[: last_block_start_index + 1]

    # Step 3: Add a column for occurrence count
    occurrence_column = []
    last_seen = None
    occurrence_count = {}

    for index, row in df.iterrows():
        current_code = row["processing_code"]
        if current_code != last_seen:
            # If new code is encountered, update the occurrence count
            occurrence_count[current_code] = occurrence_count.get(current_code, 0) + 1
        occurrence_column.append(occurrence_count[current_code])
        last_seen = current_code

    df["occurrence_count"] = occurrence_column
    return df


def process_sensor_data(df, sub_programs, device_ip_to_keep=None):

    # Step 1: Sort by datetime
    df = df.sort_values(by="datetime")

    # Step 2: Filter rows based on sub_programs
    df = df[df["processing_code"].astype(str).isin(sub_programs)]

    # Step 1: Sort by datetime
    df = df.sort_values(by="datetime")

    # Step 2: Filter rows based on sub_programs
    df = df[df["processing_code"].astype(str).isin(sub_programs)]

    # Step 3: Split the dataframe by deviceIP
    device_ip_dfs = {ip: sub_df for ip, sub_df in df.groupby("deviceIP")}

    for device_ip, device_df in device_ip_dfs.items():
        if device_ip_to_keep == device_ip:
            print("matching, device ip = ", device_ip)
            # 去頭去尾
            device_df = qu_tou_qu_wei(device_df)

            # Add deviceIP column
            device_df["device_ip"] = device_ip
            return device_df
    return pd.DataFrame()


def retain_max_occurrence_rows(df):
    # Find the maximum occurrence value for each processing_code
    max_occurrence_rows = df[
        df.groupby("processing_code")["occurrence_count"].transform("max")
        == df["occurrence_count"]
    ]
    return max_occurrence_rows


def match_sensor_code(
    sensor_df,
    code_df,
    time_tolerance=0.1,
    dist_tolerance=3,
):
    """
    Matches sensor data with code data based on time and spatial coordinates.

    Parameters:
    - sensor_df: Sensor data DataFrame, containing sensor readings and related information.
    - code_df: Code data DataFrame, containing code execution related information.
    - translation: A tuple containing the translation values for X, Y, and Z axes, used to calibrate sensor coordinates.
    - time_thres: The initial threshold for time matching.
    - time_thres_step: The step size to increase the time threshold by during each iteration.
    - dist_thres: The initial threshold for distance matching.
    - dist_thres_step: The step size to increase the distance threshold by during each iteration.

    Returns:
    - code_df: The code data DataFrame with matched sensor line information added.
    """
    # sensor_df = sensor_df[sensor_df['datetime'] < '2024-12-27 15:00:00']

    # Calibration for tool compensation
    sensor_df["H"] = (
        sensor_df["tool_cpn"].str.split("/").str[0].str.split(":").str[1].astype(int)
    )
    sensor_df["Z"] = sensor_df["Z"] - sensor_df["H"]

    translation = _find_translation(code_df, sensor_df)
    print("translation:", translation)

    # Apply translation to sensor coordinates
    sensor_df["X_translated"] = sensor_df["X"] + translation[0]
    sensor_df["Y_translated"] = sensor_df["Y"] + translation[1]
    sensor_df["Z_translated"] = sensor_df["Z"] + translation[2]

    # Calculate accumulated time for sensor data
    starting_line_idx_sensor = _find_time_baseline(sensor_df["Z"], 0)
    starting_line_idx_code = _find_time_baseline(code_df["Z"], -1)
    print(starting_line_idx_sensor, "starting_line_idx_sensor")
    print(starting_line_idx_code, "starting_line_idx_code")

    if starting_line_idx_sensor is not None and starting_line_idx_code is not None:
        time_baseline_sensor = sensor_df.loc[starting_line_idx_sensor, "datetime"]
        time_baseline_row_id = code_df.loc[starting_line_idx_code, "row_id"]
    else:
        time_baseline_sensor = pd.to_datetime(
            sensor_df["datetime"].min(), "%Y-%m-%d %H:%M:%S"
        )
        time_baseline_row_id = 0

    sensor_df["time_acc"] = (
        pd.to_datetime(sensor_df["datetime"], "%Y-%m-%d %H:%M:%S")
        - time_baseline_sensor
    ).dt.seconds
    sensor_df["time_acc_perc"] = (
        sensor_df["time_acc"] / sensor_df["time_acc"].to_list()[-1]
    )
    code_df["time_acc"] = (
        code_df["time_physical"].cumsum()
        - code_df.loc[code_df["row_id"] <= time_baseline_row_id, "time_physical"].sum()
    )
    code_df["time_acc_prev"] = code_df["time_acc"].shift(1)
    code_df["time_acc_perc"] = code_df["time_acc"] / code_df["time_acc"].to_list()[-1]

    # 对每一行应用插值并整合结果
    interpolated_results = [_interpolate_row(row) for _, row in code_df.iterrows()]
    interpolated_code_df = pd.concat(interpolated_results, ignore_index=True)
    interpolated_code_df["time_acc_perc_interpolated"] = (
        interpolated_code_df["time_acc_interpolated"]
        / interpolated_code_df["time_acc"].to_list()[-1]
    )

    # 匹配
    output_df = _match(interpolated_code_df, sensor_df, time_tolerance, dist_tolerance)
    return output_df


def _post_process(code_df):
    code_df["has_larger_before"] = code_df[
        "matched_sensor_line"
    ].expanding().max().shift(1) > code_df["matched_sensor_line"].fillna(-1)
    code_df["has_smaller_after"] = code_df["matched_sensor_line"] < code_df[
        "matched_sensor_line"
    ][::-1].expanding().min()[::-1].fillna(99999)
    code_df["invalid"] = code_df["has_larger_before"] | code_df["has_smaller_after"]
    code_df.loc[code_df["invalid"] == True, "matched_sensor_line"] = None

    code_df["matched_sensor_line_bfill"] = code_df["matched_sensor_line"].bfill()
    code_df["matched_sensor_line_ffill"] = (
        code_df["matched_sensor_line"].ffill().fillna(0)
    )
    code_df.loc[
        code_df["matched_sensor_line_bfill"] == code_df["matched_sensor_line_ffill"],
        "matched_sensor_line",
    ] = code_df["matched_sensor_line_bfill"]
    return code_df


def _match(
    interpolated_code_df, sensor_df, time_tolerance=0.05, dist_tolerance=math.sqrt(3)
):

    # 遍历 code_df 中的每一行
    matches = []
    for i, row in interpolated_code_df.iterrows():
        time, x, y, z = (
            row["time_acc_perc_interpolated"],
            row["X_interpolated"],
            row["Y_interpolated"],
            row["Z_interpolated"],
        )

        # 计算时间差和欧氏距离
        time_diff = np.abs(sensor_df["time_acc_perc"] - time)
        distance = np.sqrt(
            (sensor_df["X_translated"] - x) ** 2
            + (sensor_df["Y_translated"] - y) ** 2
            + (sensor_df["Z_translated"] - z) ** 2
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
    unmatched = set(sensor_df.index) - {
        match[1] for match in matches if match[1] is not None
    }
    for opt_idx in unmatched:
        matches.append((None, opt_idx))

    # 分开处理 interpolated_code_df 和 sensor_df_opt 的匹配结果
    sensor_matches = sorted(
        [m for m in matches if m[0] is not None], key=lambda x: x[0]
    )
    opt_matches = sorted(
        [m for m in matches if m[1] is not None and m[0] is None], key=lambda x: x[1]
    )

    # 合并结果，保持 interpolated_code_df 和 sensor_df_opt 的索引顺序
    sorted_matches = sensor_matches + opt_matches

    # 构建最终结果表
    result = []
    for sensor_idx, opt_idx in sorted_matches:
        row_sensor = (
            interpolated_code_df.loc[sensor_idx]
            if sensor_idx is not None
            else pd.Series(dtype="float64")
        )
        row_opt = (
            sensor_df.loc[opt_idx]
            if opt_idx is not None
            else pd.Series(dtype="float64")
        )
        merged_row = pd.concat(
            [row_sensor, row_opt], keys=["interpolated_code", "sensor_df_opt"]
        )
        result.append(merged_row)

    output = pd.DataFrame(result)
    return output


def _find_time_baseline(series, target_value=-1):

    # 找到所有值为 target_value 的索引
    indices = series[series == target_value].index
    # 确保有 target_value 存在
    if not indices.empty:
        # 找到相邻索引之间的差值
        diff = indices.to_series().diff().fillna(1)
        # 分组逻辑：连续部分的开始标记
        groups = (diff > 1).cumsum()

        # 第一个连续段的索引
        first_segment_indices = indices[groups == 0]

        # 获取第一段的最后一个索引
        last_index_of_first_segment = first_segment_indices[-1]
        return last_index_of_first_segment
    else:
        return None


def _find_translation(
    code_df,
    sensor_df,
    rapid_speed_assupmtion=12000,
    setting_range=0.8,
    self_change_range=0.5,
):
    # TODO: calculate translation based on
    # translation = (-308.075, 251.539, 256.317), only for 5601

    # code: the 1st non G00 XYZ
    # Edgar: 应该是上一行，因为坐标是改行的终点
    code_start_row_id = (
        code_df[code_df["move_code"].isin(["G01", "G02", "G03"])]
        .head(1)["row_id"]
        .values[0]
    )
    code_start_xyz = (
        code_df[code_df["row_id"] < code_start_row_id]
        .tail(1)[["X", "Y", "Z"]]
        .values[0]
    )
    first_F_setting = code_df.loc[code_df["row_id"] == code_start_row_id, "F"].values[0]

    # sensor: the 1st actual F when
    # 1. < rapid spd assumption 12000
    # 2. +-30% against the 1st non G00 XYZ F-setting
    # 3.next 3 lines F sensor within +-10% such F sensor (10%不够)
    # 3. 改为Z的接下的range在10%
    # its corresponding XYZ

    # sensor_df_start = sensor_df[
    #     (sensor_df['f_speed'].shift(-1).between(
    #         sensor_df['f_speed'] * (1-self_change_rage),
    #         sensor_df['f_speed'] * (1+self_change_rage)))
    #     & (sensor_df['f_speed'].shift(-2).between(
    #         sensor_df['f_speed'] * (1-self_change_rage),
    #         sensor_df['f_speed'] * (1+self_change_rage)))
    #     & (sensor_df['f_speed'].shift(-3).between(
    #         sensor_df['f_speed'] * (1-self_change_rage),
    #         sensor_df['f_speed'] * (1+self_change_rage)))
    #     ]

    sensor_df_start = sensor_df[
        (abs(sensor_df["Z"].shift(-2) - sensor_df["Z"].shift(-3)) < 0.01)
        & (sensor_df["Z"].shift(-2) != 0)
    ]
    sensor_df_start = sensor_df_start[
        (sensor_df_start["f_speed"] < rapid_speed_assupmtion)
        & (sensor_df_start["f_speed"] > 0)
    ]
    sensor_df_start = sensor_df_start[
        (sensor_df_start["f_speed"] >= first_F_setting * (1 - setting_range))
        & (sensor_df_start["f_speed"] <= first_F_setting * (1 + setting_range))
    ]

    if sensor_df_start.shape[0] > 0:
        sensor_start_xyz = sensor_df_start[["X", "Y", "Z"]].values[0]
        translation = code_start_xyz - sensor_start_xyz
        print("sensor_start_xyz", sensor_start_xyz)
        print("code_start_xyz", code_start_xyz)
    else:
        translation = (-308.075, 251.539, 256.317)  # set translation 5601 as default
    return translation


def _interpolate_row(row, time_interval=0.01):
    interpolated_df = pd.DataFrame([row])
    interpolated_df["time_acc_interpolated"] = row["time_acc"]
    interpolated_df["X_interpolated"] = row["X"]
    interpolated_df["Y_interpolated"] = row["Y"]
    interpolated_df["Z_interpolated"] = row["Z"]
    if row["time_acc_prev"] and row["time_acc"]:
        if row["time_acc_prev"] < row["time_acc"]:
            time_start, time_end = row["time_acc_prev"], row["time_acc"]
            X_prev, Y_prev, Z_prev = row["X_prev"], row["Y_prev"], row["Z_prev"]
            X, Y, Z = row["X"], row["Y"], row["Z"]

            # 生成插值时间点
            timestamps = np.arange(time_start, time_end, time_interval)
            if timestamps[-1] < time_end:
                timestamps = np.append(timestamps, time_end)

            # 线性插值计算位置
            X_interpolated = np.interp(timestamps, [time_start, time_end], [X_prev, X])
            Y_interpolated = np.interp(timestamps, [time_start, time_end], [Y_prev, Y])
            Z_interpolated = np.interp(timestamps, [time_start, time_end], [Z_prev, Z])

            # 创建结果 DataFrame
            interpolated_df = pd.DataFrame(
                {
                    "time_acc_interpolated": timestamps,
                    "X_interpolated": X_interpolated,
                    "Y_interpolated": Y_interpolated,
                    "Z_interpolated": Z_interpolated,
                }
            )
            for col in row.index:
                interpolated_df[col] = row[col]
    return interpolated_df


def match_code_to_sensor(code_df, sensor_df):
    matched_pair = code_df.groupby("matched_sensor_line")["row_id"].min().reset_index()
    for k, v in dict(
        zip(matched_pair["matched_sensor_line"], matched_pair["row_id"])
    ).items():
        sensor_df.loc[k, "matched_code_line"] = v
    # sensor_df['matched_code_line'] = sensor_df['matched_code_line'].bfill().fillna(code_df.shape[0]-1)

    code_df_to_merge = code_df.copy()
    code_df_to_merge["matched_code_line"] = code_df_to_merge.index + 1
    sensor_df = sensor_df.merge(
        code_df_to_merge,
        on="matched_code_line",
        how="right",
        suffixes=["", "_code"],
        validate="many_to_one",
    )
    return sensor_df


def process_combined_out(df_combined, rapid_type=["G0", "G00"], rapid_spd=12000):

    df_combined["F"] = np.where(
        (df_combined["type"].isin(rapid_type)) & (df_combined["f_speed"] > rapid_spd),
        df_combined["f_speed"],
        df_combined["F"],
    )
    df_combined["F"] = np.where(
        (df_combined["type"].isin(rapid_type)), rapid_spd, df_combined["F"]
    )

    df_combined = (
        df_combined.rename(columns={"index": "index_sensor"})
        .sort_values(by=["sub_program", "row_id", "index_sensor"])
        .reset_index()
    )
    return df_combined


def run_sensor_merge(conf_ct, conf_datapath, df_sensor="df_sensor"):
    ct_dfs = run_calculate_cycle_time(conf_ct["path"]["dir_g_code"])

    df_code_analysis = run_analysis(conf_ct)
    sub_programs = conf_ct["sub_programs"]
    path_out = conf_ct["out"]["sensor_v1_out"]

    df_sensor_dict = load_sensor(conf_datapath)
    sensor_df = process_sensor_data(df_sensor_dict[df_sensor], sub_programs)
    sensor_df = retain_max_occurrence_rows(sensor_df)

    for idx, sub_program in enumerate(sub_programs, start=1):
        print(f"--- analyzing {sub_program}...")
        sensor_df_subprogram = sensor_df[
            sensor_df["processing_code"].astype(str) == sub_program
        ]  # int(sub_program)
        sensor_df_subprogram = sensor_df_subprogram.reset_index(drop=True)

        code_df = ct_dfs[ct_dfs["O"].astype(str) == sub_program]

        # TODO: 需要把輔助M代碼拿走，目前默認它繼續G00 join會有問題 (一旦遇到M代码，move状态重置)
        code_df = code_df[~code_df["src"].str.contains("M|N", na=False)]

        output_df = match_sensor_code(sensor_df_subprogram, code_df)
        output_df.to_excel(f"{path_out}/{idx}-{sub_program}.xlsx")


def match_sensor_codes(sensor_df_1, sensor_df_2):
    pass


if __name__ == "__main__":

    with open("cnc_genai/conf/v1_config.yaml", "r") as file:
        conf_ct = yaml.safe_load(file)

    with open("cnc_genai/conf/data_path.yaml", "r") as file:
        conf_datapath = yaml.safe_load(file)

    run_sensor_merge(conf_ct, conf_datapath, df_sensor="df_sensor")
