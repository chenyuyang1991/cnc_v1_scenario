import os
import math
import pandas as pd
import numpy as np
from cnc_genai.src.utils.utils import load_config_v1
from cnc_genai.src.v1_algo.load_analysis import run_analysis


def generate_tool_group(df, conf):
    precision = conf["precision"]
    scale = 10 ** (3 - precision)
    if "tool_diameter_mm" not in df.columns:
        df["tool_diameter_mm"] = df["tool_diameter"] * scale
    if "tool_height_mm" not in df.columns:
        df["tool_height_mm"] = df["tool_height"] * scale
    tool_stats = (
        df.groupby(["T", "tool_diameter_mm", "tool_height_mm"])
        .agg(
            subprograms=(
                "sub_program",
                lambda x: list(x.unique()),
            ),  # 獲取唯一子程序列表
            pc_max=("power_pc", "max"),
            pc_min=("power_pc", "min"),
            pc_mean=("power_pc", "mean"),
            pc_std=("power_pc", "std"),
            count=("power_pc", "count"),
        )
        .reset_index()
    )
    # tool_stats["tool_diameter"] *= scale  # in mm
    # tool_stats["tool_height"] *= scale  # in mm
    tool_df_path = (
        f"{conf['path']['dir_app']}/{conf['clamping_name']}/{conf['path']['tool_path']}"
    )
    tools = pd.read_excel(tool_df_path)
    if "刀號" in tools.columns:
        pass  # 第一行作為 header
    else:
        tools = pd.read_excel(tool_df_path, header=1)  # 第二行作為 header
    tool_stats = tool_stats.merge(
        tools[["刀號", "規格型號"]], left_on="T", right_on="刀號", how="left"
    )
    tool_stats = tool_stats[tool_stats["count"] > 0].drop_duplicates("T")

    def _create_tool_groups(row):
        if row["tool_diameter_mm"] >= 10:
            if row["pc_mean"] > 0.5:  # 高負載粗加工
                return "G1-重型粗加工"
            else:  # 大直徑但低負載
                return "G2-大徑精修"

        elif 3 <= row["tool_diameter_mm"] < 10:
            if row["pc_mean"] > 0.2:  # 中等負載
                return "G3-中徑常規"
            elif row["pc_std"] > 0.1:  # 負載波動大
                return "G4-動態加工"
            else:  # 穩定低負載
                return "G5-精密加工"

        else:  # 小直徑刀具
            if row["pc_max"] < 0.05:  # 微細加工
                return "G6-微雕加工"
            else:  # 常規小徑
                return "G5-精密加工"

    tool_stats["T_group"] = tool_stats.apply(_create_tool_groups, axis=1)
    df["T_group"] = df["T"].map(tool_stats.set_index("T")["T_group"])
    return df


def calculate_percentiles_by_tool(df, percentile=0.7, group_col="T"):
    """
    Calculate the wanted percentile of MRR, Pc, and Mc, grouped by tool or tool group,
    ignoring rows with missing group values.
    """

    # 删除 'tool' 中为空值的行
    # df = generate_tool_group(df, conf)
    df_noNaN = df.dropna(subset=[group_col])

    result = {}
    grouped = df_noNaN.groupby(group_col)

    for group_name, group in grouped:
        mrr = group["MRR"].quantile(percentile)
        pc = group["power_pc"].quantile(percentile)
        mc = group["torque_mc"].quantile(percentile)

        result[group_name] = {"MRR": mrr, "pc": pc, "mc": mc}

    # 转换为 DataFrame，處理空字典的情況
    if result:
        tool_df = pd.DataFrame.from_dict(result, orient="index").reset_index()
        tool_df.rename(columns={"index": group_col}, inplace=True)
    else:
        # 如果沒有數據，返回空的 DataFrame 且具有正確的列結構
        tool_df = pd.DataFrame(columns=[group_col, "MRR", "pc", "mc"])
    return df, tool_df


def infer_finishing_tag(df, ae_thres=0.5, ap_thres=0.1):
    if "is_valid_slack" not in list(df):
        df["is_valid_slack"] = df["is_valid"]

    # 精修tag，以下三種都認為是精修：1）材料在刀具外緣外側的10絲以內；2）材料在刀具外緣內側的50絲以內，用ae_mm來判斷；3）材料在刀具下緣上方的10絲以內，用ap_mm來判斷
    df["is_finishing_outer"] = np.where(
        (df["is_valid"] == 0) & (df["is_valid_slack"] == 1), 1, 0
    )
    df["is_finishing_inner"] = np.where(
        (df["is_valid"] == 1) & ((df["ae_mm"] < ae_thres) | (df["ap_mm"] < ap_thres)),
        1,
        0,
    )
    df["is_finishing"] = (df["is_finishing_outer"] + df["is_finishing_inner"]).astype(
        bool
    )

    # 提高空切的判斷門檻，需要離材料slack//2以上才認為是空切
    df["is_valid"] = df["is_valid_slack"]
    return df


def add_continuous_duration(
    df, is_valid_col="is_valid", time_col="time_physical", short_threshold=0.5
):
    # 新增一列 validness_group_id，用于标记连续 is_valid 相同的组
    df["validness_group_id"] = (df[is_valid_col] != df[is_valid_col].shift()).cumsum()

    # 计算 continuous_duration
    continuous_duration = df.groupby("validness_group_id")[time_col].transform("sum")
    df["continuous_duration"] = continuous_duration
    df["continuous_num_lines"] = df.groupby("validness_group_id").transform("size")
    df["is_short_duration"] = (df["continuous_duration"] < short_threshold).astype(int)
    return df


def generate_applicable_tag(
    df,
    short_threshold=0.5,
    ae_thres=0.5,
    ap_thres=0.1,
    turning_G01_thres=0.5,
    pre_turning_thres=1,
    ban_n=[],
    ban_row=[],
):

    # 識別真正的空切(非精修)，並tag精修
    df = infer_finishing_tag(df, ae_thres=ae_thres, ap_thres=ap_thres)

    # 合併連續空切
    df = add_continuous_duration(df, short_threshold=short_threshold)

    # G00 -> G02/G03 tag
    df["is_curve_start"] = np.where(
        (df["move_code"].isin(["G02", "G03"])) & (df["move_code_prev"] == "G00"), 1, 0
    )

    # 轉角tag: 1. G02/G03, 2. G01 && XY均改變 && 直線距離<0.5mm=50條
    df["is_turning"] = np.where(
        (df["move_code"].isin(["G02", "G03"]))
        | (
            (df["move_code"] == "G01")
            & (df["path_length"] < turning_G01_thres)  # 0.5mm
            & (df["X"] != df["X_prev"])
            & (df["Y"] != df["Y_prev"])
        ),
        1,
        0,
    )
    df["is_pre_turning"] = np.where(
        (
            (df["move_code"].isin(["G01", "G02", "G03"]))
            & (df["path_length"] < pre_turning_thres)  # 1mm
            & (df["is_turning"].shift(-1) == 1)
        ),
        1,
        0,
    )
    df["is_banned_turning"] = ((df["is_turning"] + df["is_pre_turning"]) > 0) * (
        1 - df["apply_turning"]
    )

    # 螺旋下刀tag
    df["is_spiral"] = np.where(
        (df["X"] != df["X_prev"])
        & (df["Y"] != df["Y_prev"])
        & (df["Z"] != df["Z_prev"]),
        1,
        0,
    )

    # 篩選程序段
    df["is_ban_n"] = np.where((df["N"].isin(ban_n)), 1, 0)
    df["is_ban_n"] = np.where((df["row_id"].isin(ban_row)), 1, df["is_ban_n"])

    # 篩選空切
    df["apply_air"] = np.where(
        (df["is_ban_n"] == 0)  # not banned
        & (df["is_valid"] == 0)
        & (df["is_short_duration"] == 0)
        & (df["apply_subprogram_air"] == 1)
        & (df["is_banned_turning"] == 0),
        1,
        0,
    )

    # 篩選AFC
    df["apply_afc"] = np.where(
        (df["is_ban_n"] == 0)  # not banned
        & (df["is_valid"] == 1)
        & (df["is_short_duration"] == 0)
        & (df["is_curve_start"] == 0)
        & (df["is_spiral"] == 0)
        & (df["apply_subprogram_afc"] == 1)
        & (df["is_banned_turning"] == 0),
        1,
        0,
    )

    # 提升標籤
    df["apply"] = (df["apply_air"] + df["apply_afc"]).astype(bool)

    return df


def apply_adjust_feed_rate(
    df,
    conf,
    pc_threshold_dict,
    multiplier_max=1.5,
    multiplier_min=1,
    multiplier_air=2,
    apply_finishing=1,
    multiplier_finishing=1.1,
):
    """Calculate the feed rate adjustment based on the Power threshold."""
    df["multiplier_max"] = multiplier_max
    df["multiplier_min"] = multiplier_min
    df["multiplier_air"] = multiplier_air
    df["apply_finishing"] = apply_finishing
    df["multiplier_finishing"] = multiplier_finishing
    # todo add target_power_pc for all rows not only where apply_afc==1

    group_col = (
        "T_group" if conf["hyper_params"]["target_pwc_strategy"] == "按刀具組" else "T"
    )

    def adjust_feed_rate(row, conf=conf):

        min_air_speed = conf["hyper_params"]["min_air_speed"]
        max_air_speed = conf["hyper_params"]["max_air_speed"]
        max_step = conf["hyper_params"]["max_increase_step"]

        # 提升切割代碼
        if row["move_code"] in ["G01", "G02", "G03"]:  # G81, G82, G83, G84 暂不提升

            # 空切
            if row["apply_air"] == 1:
                return pd.Series(
                    [
                        max(
                            min(row["F"] * row["multiplier_air"], max_air_speed),
                            min_air_speed,
                        ),
                        None,
                    ]
                )

            # 提升進給
            elif row["apply_afc"] == 1:

                # 用户选择按刀具组
                if group_col == "T_group":
                    # 开粗子程序则使用刀具组目标功率
                    if conf["sub_programs"][str(row["sub_program"])]["finishing"] == 0:
                        pc_threshold = pc_threshold_dict.get(row[group_col], None)
                    else:
                        pc_threshold = pc_threshold_dict.get(row["T"], None)
                else:
                    pc_threshold = pc_threshold_dict.get(row[group_col], None)

                # 考慮提升
                power_pc_value = row.get("power_pc", 0)
                pc = (
                    power_pc_value
                    if pd.notna(power_pc_value) and np.isfinite(power_pc_value)
                    else 0
                )
                if pc < pc_threshold:
                    # 如果是精修子程序，则使用 multiplier_finishing
                    if row["is_finishing_subprogram"] == 1:
                        # 如果配置提升精修
                        if row["apply_finishing"] == 1:
                            if pc == 0:
                                return pd.Series(
                                    [
                                        min(
                                            row["F"] * row["multiplier_finishing"],
                                            row["F"] + max_step,
                                        ),
                                        pc_threshold,
                                    ]
                                )
                            else:
                                return pd.Series(
                                    [
                                        min(
                                            pc_threshold / row["power_pc"] * row["F"],
                                            row["F"] * row["multiplier_finishing"],
                                            row["F"] + max_step,
                                        ),
                                        pc_threshold,
                                    ]
                                )
                        # 如果不配置提升精修
                        else:
                            return pd.Series([row["F"], None])

                    # 如果不是精修子程序
                    else:
                        # 如果是精修代碼
                        if row["is_finishing"] == 1:
                            # 如果配置提升精修
                            if row["apply_finishing"] == 1:
                                return pd.Series(
                                    [
                                        min(
                                            pc_threshold / row["power_pc"] * row["F"],
                                            row["F"] * row["multiplier_finishing"],
                                            row["F"] + max_step,
                                        ),
                                        pc_threshold,
                                    ]
                                )
                            # 如果不配置提升精修
                            else:
                                return pd.Series([row["F"], None])
                        # 如果是非精修代碼
                        else:
                            return pd.Series(
                                [
                                    min(
                                        pc_threshold / row["power_pc"] * row["F"],
                                        row["F"] * row["multiplier_max"],
                                        row["F"] + max_step,
                                    ),
                                    pc_threshold,
                                ]
                            )

                        # 另一种逻辑：无论是否精修代码，只要是非精修子程式，全部正常提速
                        # if pc == 0:
                        #     return pd.Series(
                        #         [
                        #             min(
                        #                 row["F"] * row["multiplier_max"],
                        #                 row["F"] + max_step,
                        #             ),
                        #             pc_threshold,
                        #         ]
                        #     )
                        # else:
                        #     return pd.Series(
                        #         [
                        #             min(
                        #                 pc_threshold / row["power_pc"] * row["F"],
                        #                 row["F"] * row["multiplier_max"],
                        #                 row["F"] + max_step,
                        #             ),
                        #             pc_threshold,
                        #         ]
                        #     )

                # 考慮下降
                else:
                    try:
                        return pd.Series(
                            [
                                max(
                                    pc_threshold / row["power_pc"] * row["F"],
                                    row["F"] * row["multiplier_min"],
                                ),
                                pc_threshold,
                            ]
                        )
                    except:
                        return pd.Series(
                            [
                                row["F"],
                                pc_threshold,
                            ]
                        )

            else:
                pass

        # 提升G00
        elif row["move_code"] == "G00":
            return pd.Series([row["F"], None])

        # 其他代碼 - 確保所有情況都返回兩個值
        return pd.Series([row["F"], None])

    df[["F_adjusted", "target_power_pc"]] = df.apply(adjust_feed_rate, axis=1)
    # df['F_adjusted'] = df.apply(lambda row: row['F_adjusted'] // 100 * 100 if row['F_adjusted'] > row['F'] else row['F'])
    df["F_adjusted"] = np.where(
        df["F_adjusted"] // 100 * 100 > df["F"], df["F_adjusted"] // 100 * 100, df["F"]
    )
    df["multiplier_actual"] = (df["F_adjusted"] / df["F"]).fillna(1)
    df["time_physical_adjusted"] = df["time_physical"] / df["multiplier_actual"]
    df["time_physical_improved"] = df["time_physical"] - df["time_physical_adjusted"]
    return df


def analyze_subprogram(
    df,
    pc_threshold_dict,
    conf,
    short_threshold=0.5,
    ae_thres=0.5,
    ap_thres=0.1,
    multiplier_max=1.5,
    multiplier_min=1,
    multiplier_air=2,
    apply_finishing=1,
    multiplier_finishing=1.1,
    turning_G01_thres=0.5,
    pre_turning_thres=1,
    ban_n=[],
    ban_row=[],
    multiplier_sub_program=pd.DataFrame(),
):
    """Analyze a single Excel file and calculate feed rate adjustments."""

    # 篩選applicable的代碼行
    df = generate_applicable_tag(
        df,
        short_threshold=short_threshold,
        ae_thres=ae_thres,
        ap_thres=ap_thres,
        turning_G01_thres=turning_G01_thres,
        pre_turning_thres=pre_turning_thres,
        ban_n=ban_n,
        ban_row=ban_row,
    )

    try:
        multiplier_max_value = multiplier_sub_program["multiplier_max"][0]
        if multiplier_max_value is not None and multiplier_max_value > 0:
            multiplier_max = max(multiplier_max_value, multiplier_min)
    except KeyError:
        pass
        # print("No multiplier_max in multiplier_sub_program")

    # Add feed rate adjustment column
    df = apply_adjust_feed_rate(
        df,
        conf,
        pc_threshold_dict,
        multiplier_max=multiplier_max,
        multiplier_min=multiplier_min,
        multiplier_air=multiplier_air,
        apply_finishing=apply_finishing,
        multiplier_finishing=multiplier_finishing,
    )

    return df


def run_adjust_feed_rate(conf, verbose=True):

    df = run_analysis(conf)
    print("[INFO] STEP 3: Adjust Feed Rate")

    # 確保精修倍率低於非精修倍率
    conf["hyper_params"]["multiplier_finishing"] = min(
        conf["hyper_params"]["multiplier_finishing"],
        conf["hyper_params"]["multiplier_max"],
    )

    # Calculate percentiles on mrr, pc, mc
    df = generate_tool_group(df, conf)
    df, grouped_percentiles = calculate_percentiles_by_tool(
        df,
        percentile=conf["hyper_params"]["percentile_threshold"],
        group_col="T_group",
    )
    pc_threshold_dict = grouped_percentiles.set_index("T_group")["pc"].to_dict()

    _, grouped_percentiles = calculate_percentiles_by_tool(
        df,
        percentile=conf["hyper_params"]["percentile_threshold"],
        group_col="T",
    )
    tool_pc_threshold_dict = grouped_percentiles.set_index("T")["pc"].to_dict()
    pc_threshold_dict.update(tool_pc_threshold_dict)

    if verbose:
        print("-" * 30)
        print("[INFO] pc_threshold_dict:", pc_threshold_dict)
        print("-" * 30)

    df_outs = []

    # 修正feed_rate
    for sub_program in conf["sub_programs"]:
        print(f"[INFO] 正在提速{sub_program}")
        df_sub = df[
            df["sub_program"].astype(str).str.zfill(4) == str(sub_program).zfill(4)
        ]
        df_sub["is_finishing_subprogram"] = conf["sub_programs"][sub_program][
            "finishing"
        ]
        df_sub["apply_subprogram_air"] = conf["sub_programs"][sub_program].get(
            "apply_air", True
        )
        df_sub["apply_subprogram_afc"] = conf["sub_programs"][sub_program].get(
            "apply_afc", True
        )
        df_sub["apply_turning"] = conf["sub_programs"][sub_program].get(
            "apply_turning", False
        )
        df_outs.append(
            analyze_subprogram(
                df_sub,
                pc_threshold_dict,
                conf,
                short_threshold=conf["hyper_params"]["short_threshold"],
                ae_thres=conf["hyper_params"]["ae_thres"],
                ap_thres=conf["hyper_params"]["ap_thres"],
                multiplier_max=conf["hyper_params"]["multiplier_max"],
                multiplier_min=conf["hyper_params"]["multiplier_min"],
                multiplier_air=conf["hyper_params"]["multiplier_air"],
                apply_finishing=conf["hyper_params"]["apply_finishing"],
                multiplier_finishing=conf["hyper_params"]["multiplier_finishing"],
                turning_G01_thres=conf["hyper_params"]["turning_G01_thres"],
                pre_turning_thres=conf["hyper_params"]["pre_turning_thres"],
                ban_n=conf["sub_programs"][sub_program].get("ban_n", []),
                ban_row=conf["sub_programs"][sub_program].get("ban_row", []),
                multiplier_sub_program=pd.DataFrame(
                    [conf["sub_programs"][sub_program]]
                ),
            )
        )

    out_df = pd.concat(df_outs, axis=0)
    out_df = out_df[[x for x in conf["output_cols"] if x in out_df.columns]]
    return out_df


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    conf = load_config_v1("cnc_genai/conf/v1_config.yaml")
    out_df = run_adjust_feed_rate(conf)

    out_dir = f'{conf["path"]["dir_intermediate"]}/{conf["clamping_name"]}/{conf["output_path"]["adjust_feed_rate"]}'
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_excel(f"{out_dir}/{conf['scenario_name']}.xlsx", index=False)

    print("improved time in seconds:", out_df.time_physical_improved.sum())
    print(f"scenario_name, {conf['scenario_name']}")
    print(
        f"uplift, {out_df.time_physical_improved.sum()/out_df.drop_duplicates('sub_program')['real_ct'].sum()*100:.2f}%"
    )
