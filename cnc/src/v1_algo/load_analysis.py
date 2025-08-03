import os
import numpy as np
import pandas as pd
import math
from cnc_genai.src.utils.utils import load_config_v1
from cnc_genai.src.v1_algo.calculate_load_time import run_calculate_cycle_time


def calculate_mrr_and_loads(df, conf, Kc=600, eta=1):
    """
    Calculate material removal rate (MRR), power (Pc), torque (Mc), and cutting speed (Vc).
    """
    required_columns = {
        "ae_avg_over_hit",
        "ap_mm",
        "time_physical",
        "tool_diameter",
        "S",
        "F",
    }
    if not required_columns.issubset(df.columns):
        print(
            f"Dataframe missing required columns: {required_columns - set(df.columns)}"
        )
        return df
    # Calculate MRR
    # 用hit area 代替
    # df['MRR'] = (df['ae_avg_over_hit'] / 100 * df['ap_mm']) / (df['time_physical']/60)
    scale = 10 ** (3 - conf["precision"])

    df["MRR"] = (df["hit_area"] / scale**3) / (df["time_physical"] / 60)
    df["cutting_speed_vc"] = df["tool_diameter"] * math.pi * df["S"]  # mm/min

    # TODO 常数是否与体积precision有关
    df["power_pc"] = (
        df["ae_avg_over_hit"] / scale**2 * df["ap_mm"] / (df["time_physical"] / 60) * Kc
    ) / (
        60 * 10**6 * eta
    )  # 单位：kW
    df["torque_mc"] = (df["power_pc"] * 30 * 10**3) / (math.pi * df["S"])  # 单位：Nm

    return df


def calculate_ap_ae(df, conf):
    """Calculate 'ap_mm' and 'ae_mm' based on given formula."""
    # 注意： df['F'] is mm/min df['time_physical'] is seconds
    # Edgar 建議用max instead of avg，保持ap_mm和ae_mm口徑一致
    # df['ae_mm'] = (df['ae_avg_over_hit']/100) / (df['time_physical'] * df['F']/60)

    scale = 10 ** (3 - conf["precision"])
    df["ap_mm"] = df.apply(
        lambda row: min(row["ap_max_over_xy"] / scale, row["tool_height"] / scale),
        axis=1,
    )

    # TODO 应该分情况讨论，以下只适合在XY平面内切割的情况
    df["ae_mm"] = df.apply(
        lambda row: (
            min(
                row["ae_max_over_z"]
                / scale**2
                / (row["time_physical"] * row["F"] / 60),
                row["tool_diameter"] / scale,
            )
            if row["time_physical"] * row["F"] != 0
            else None
        ),
        axis=1,
    )
    # df['ae_mm'] = (df['ae_avg_over_hit'] / 100) / (df['time_physical'] * df['F'] / 60)

    return df


def infer_finishing_tag(df, inner_thres=50):
    if "is_valid_slack" not in list(df):
        df["is_valid_slack"] = df["is_valid"]

    # 精修tag，以下兩種都認為是精修：1）材料在刀具外緣外側的10絲以內；2）材料在刀具外緣內側的50絲以內，用ae_mm來判斷
    df["is_finishing_outer"] = np.where(
        (df["is_valid"] == 0) & (df["is_valid_slack"] == 1), 1, 0
    )
    df["is_finishing_inner"] = np.where(
        (df["is_valid"] == 1) & (df["ae_mm"] * 100 < inner_thres), 1, 0
    )
    df["is_finishing"] = (df["is_finishing_outer"] + df["is_finishing_inner"]).astype(
        bool
    )

    # 提高空切的判斷門檻，需要離材料slack//2以上才認為是空切
    df["is_valid"] = df["is_valid_slack"]
    return df


# Analyze a single file


def run_analysis(conf):
    ct_df = run_calculate_cycle_time(conf)
    print("[INFO] STEP 2: Calculate Load Analysis")

    # 計算ap_mm和ae_mm
    ct_df = calculate_ap_ae(ct_df, conf)

    # 計算MMR和負載等特征
    ct_df = calculate_mrr_and_loads(ct_df, conf)
    # ct_df = ct_df.sort_values(
    #     by=["sub_program_seq", "sub_program_key", "row_id"]
    # )

    return ct_df


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    conf = load_config_v1("./cnc_genai/conf/v1_config.yaml")

    out_df = run_analysis(conf)

    out_dir = f'{conf["path"]["dir_intermediate"]}/{conf["clamping_name"]}/{conf["output_path"]["calc_load"]}'
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_excel(f"{out_dir}/load_analysis.xlsx", index=False)
