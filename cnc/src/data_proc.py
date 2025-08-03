import pandas as pd
import numpy as np


def run_ct_delta_analysis(df_ct, df_machine, thrhld=1, result_dict={}, top_n=3):
    df_ct = df_ct.groupby(["機台號"])[["實際CT"]].mean().reset_index()
    df_ct["實際CT"] = df_ct["實際CT"].round(0).astype(int)
    # Create all possible pairs of machines excluding self-pairs
    df_pairs = df_ct.assign(key=1).merge(
        df_ct.assign(key=1), on="key", suffixes=("_A", "_B")
    )
    df_pairs = df_pairs[df_pairs["機台號_A"] != df_pairs["機台號_B"]].drop(
        "key", axis=1
    )

    # Compute the CT delta percentage
    df_pairs["機台A比機台B_CT高%"] = (
        (df_pairs["實際CT_A"] - df_pairs["實際CT_B"]) / df_pairs["實際CT_A"]
    ) * 100
    df_pairs["機台A比機台B_CT高%"] = df_pairs["機台A比機台B_CT高%"].round(0).astype(int)
    # Rank from highest to lowest
    # df_pairs = df_pairs.sort_values(by='機台A比機台B_CT高%', ascending=False).reset_index(drop=True)
    df_pairs = (
        df_pairs.sort_values(
            by=["機台號_A", "機台A比機台B_CT高%"], ascending=[True, False]
        )
        .reset_index(drop=True)
        .merge(
            df_machine.rename(columns={"机台号": "機台號_A", "年限": "機台號_A年限"}),
            on="機台號_A",
            how="left",
        )
        .merge(
            df_machine.rename(columns={"机台号": "機台號_B", "年限": "機台號_B年限"}),
            on="機台號_B",
            how="left",
        )
        .sort_values(by=["機台A比機台B_CT高%"], ascending=False)
    )

    df_pairs = df_pairs[df_pairs["機台A比機台B_CT高%"] > thrhld]
    df_pairs["top_n"] = df_pairs.groupby(["機台號_A", "機台號_A年限", "機台號_B年限"])[
        "機台A比機台B_CT高%"
    ].rank(method="first", ascending=False)
    df_pairs = (
        df_pairs[df_pairs["top_n"] <= top_n]
        .drop(columns=["top_n"])
        .sort_values(by=["機台A比機台B_CT高%"], ascending=False)
        .drop(columns=["機台號_A年限", "機台號_B年限"])
    )

    for index, row in df_pairs.iterrows():
        # Construct the key using '機台號_A' and '機台號_B'
        key = f"{row['機台號_A']}_vs_{row['機台號_B']}"

        # Create the value dictionary with the required fields
        value = {
            "實際CT_A": row["實際CT_A"],
            "實際CT_B": row["實際CT_B"],
            "機台A比機台B_CT高%": row["機台A比機台B_CT高%"],
        }

        # Add the key-value pair to the result dictionary
        result_dict[key] = value

    return result_dict


def run_subscript_definition(df_script_desc):

    df_script_desc["程式號碼"] = df_script_desc["程式號碼"].astype(str)

    return df_script_desc.set_index("程式號碼").to_dict("index")


def run_machine_definition(df_machine_desc):

    df_machine_desc = df_machine_desc[["机台号", "品牌", "系统", "年限"]]
    df_machine_desc.columns = ["機台號", "品牌", "系統", "年限"]

    return df_machine_desc.set_index("機台號").to_dict("index")
