import pandas as pd
import numpy as np
import os
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
import glob
from pathlib import Path, PurePath
import json


def load_landing_product_options(finished=True):

    # 獲取所有符合模式的 simulation_master 目錄
    master_paths = glob.glob("../app/*/simulation_master")

    # 存儲所有找到的資料夾
    folders = []

    # 遍歷每個 simulation_master 目錄
    for master_path in master_paths:
        # 獲取上一級目錄名稱（不限於 mac*）
        parent_dir = Path(master_path).parts[-2]
        master_path = Path(master_path)

        # 獲取該目錄下的所有資料夾及其創建時間
        for folder in master_path.iterdir():
            if folder.is_dir():
                # 新增條件檢查檢測仿真是否已經完成
                if finished:
                    process_info_path = folder / "simulation/latest/process_info.json"
                    # 檢查 process_info.json 中的 finish_flag 是否為 True
                    try:
                        with open(process_info_path, "r", encoding="utf-8") as f:
                            process_info = json.load(f)
                        if process_info.get("finish_flag") is True:
                            # 存儲格式為 (父目錄/資料夾名, 創建時間, 完整路徑)
                            folders.append(
                                (
                                    f"{parent_dir}/{folder.name}",
                                    folder.stat().st_mtime,
                                    str(folder),
                                )
                            )
                    except (FileNotFoundError, json.JSONDecodeError, KeyError):
                        # 如果檔案不存在或無法解析，則跳過
                        pass
                else:
                    folders.append(
                        (
                            f"{parent_dir}/{folder.name}",
                            folder.stat().st_mtime,
                            str(folder),
                        )
                    )

    # Sort folders by timestamp (newest first)
    folders.sort(key=lambda x: x[1], reverse=True)
    all_options = [folder[0] for folder in folders]

    # 按楼层权限筛选
    if st.session_state.username == "admin":
        pass
    elif st.session_state.username in ["mac1", "mac2", "mac3"]:
        all_options = [
            x for x in all_options if x.startswith(st.session_state.username)
        ]
    else:
        # 其他用戶，不顯示任何選項
        # TODO 需要設定楼层用戶的選項
        all_options = []

    return all_options


def load_landing_scenario_options():

    # Get the scenario directory path
    master_paths = glob.glob("../app/*/scenario")

    # Get folders and their creation times
    folders = []

    for scenario_path in master_paths:
        # 將字符串路徑轉換為 Path 對象
        parent_dir = Path(scenario_path).parts[-2]
        scenario_path = Path(scenario_path)

        for folder in scenario_path.iterdir():
            if folder.is_dir():
                creation_time = folder.stat().st_mtime
                folders.append((parent_dir, folder.name, creation_time))

    # Sort folders by timestamp (newest first)
    folders.sort(key=lambda x: x[2], reverse=True)
    all_options = [f"{folder[0]}/{folder[1]}" for folder in folders]

    # 按楼层权限筛选
    if st.session_state.username == "admin":
        pass
    elif st.session_state.username in ["mac1", "mac2", "mac3"]:
        all_options = [
            x for x in all_options if x.startswith(st.session_state.username)
        ]
    return all_options


def load_rerun_conf(department="mac1", scenario="X2867-CNC2_test1"):

    path = f"../app/{department}/scenario/{scenario}/{scenario}.xlsx"

    # 檢查檔案是否存在
    if not os.path.exists(path):
        # 返回空的默認值
        print(f"警告: 檔案不存在 {path}")
        hyper_params_config = {"department": department}
        df_sub_program = pd.DataFrame(columns=["sub_program"])
        df_ban_n = pd.DataFrame(columns=["sub_program"])
        df_ban_row = pd.DataFrame(columns=["sub_program"])
        st.session_state.bboxes_n_prev = pd.DataFrame()
        st.session_state.bboxes_row_prev = pd.DataFrame()
        return hyper_params_config, df_sub_program, df_ban_n, df_ban_row

    try:
        # sub_program setting
        df_sub_program = pd.read_excel(path, sheet_name="sub_program", skiprows=1)
        df_sub_program["sub_program"] = (
            df_sub_program["sub_program"].astype(str).str.zfill(4)
        )

        # hyper_params_config
        hyper_params_config = pd.read_excel(
            path, sheet_name="hyper_params", skiprows=1
        ).to_dict(orient="records")[0]
        hyper_params_config["department"] = department

        # Ban N
        df_ban_n = pd.read_excel(path, sheet_name="ban_n", skiprows=1)
        df_ban_n["sub_program"] = (
            df_ban_n["sub_program"].astype(str).astype(str).str.zfill(4)
        )

        # Ban Rows
        df_ban_row = pd.read_excel(path, sheet_name="ban_row", skiprows=1)
        df_ban_row["sub_program"] = (
            df_ban_row["sub_program"].astype(str).astype(str).str.zfill(4)
        )

    except Exception as e:
        # 如果讀取主要數據時發生錯誤，返回空的默認值
        print(f"警告: 無法讀取檔案 {path}: {e}")
        hyper_params_config = {"department": department}
        df_sub_program = pd.DataFrame(columns=["sub_program"])
        df_ban_n = pd.DataFrame(columns=["sub_program"])
        df_ban_row = pd.DataFrame(columns=["sub_program"])

    # Bboxes N
    try:
        st.session_state.bboxes_n_prev = pd.read_excel(
            path, sheet_name="bboxes_n", skiprows=1
        )
    except:
        st.session_state.bboxes_n_prev = pd.DataFrame()

    # Bboxes Row
    try:
        st.session_state.bboxes_row_prev = pd.read_excel(
            path, sheet_name="bboxes_row", skiprows=1
        )
    except:
        st.session_state.bboxes_row_prev = pd.DataFrame()

    return hyper_params_config, df_sub_program, df_ban_n, df_ban_row


def load_sub_program_init(department="mac1", product="X2867-CNC2"):

    path = f"../app/{department}/simulation_master/{product}"
    excel_file = path + "/product_master.xlsx"

    # 檢查檔案是否存在，如果不存在則返回空的 DataFrame
    if not os.path.exists(excel_file):
        # 返回空的 DataFrame，包含預期的欄位
        return pd.DataFrame(
            columns=[
                "sub_program",
                "function",
                "finishing",
                "apply_air",
                "apply_afc",
                "apply_turning",
                "multiplier_max",
            ]
        )

    try:
        df = pd.read_excel(excel_file)
        df["sub_program"] = df["sub_program"].astype(int).astype(str).str.zfill(4)
        df["function"] = df["function"].astype(str)
        # Set 'finishing' to 1 if it contains any of the specified strings, else 0
        df["finishing"] = np.where(
            df["function"].str.contains("精|外觀|外观|倒角", regex=True), 1, 0
        )
        df["finishing"] = np.where(
            df["function"].str.contains("開粗|开粗|粗|去余料", regex=True),
            0,
            df["finishing"],
        )

        df["apply_air"] = 1
        df["apply_afc"] = 1
        df["apply_turning"] = 1
        df["multiplier_max"] = np.nan
        return df
    except Exception as e:
        # 如果讀取檔案時發生任何錯誤，返回空的 DataFrame
        print(f"警告: 無法讀取檔案 {excel_file}: {e}")
        return pd.DataFrame(
            columns=[
                "sub_program",
                "function",
                "finishing",
                "apply_air",
                "apply_afc",
                "apply_turning",
                "multiplier_max",
            ]
        )


def load_all_nc_block(department="mac1", product="X2867-CNC2"):
    path = f"../app/{department}/simulation_master/{product}"
    excel_file = path + "/program_segments.xlsx"

    # 檢查檔案是否存在，如果不存在則返回空的 DataFrame
    if not os.path.exists(excel_file):
        # 返回空的 DataFrame，包含預期的欄位
        return pd.DataFrame(columns=["sub_program", "ban_n"])

    try:
        df_n = pd.read_excel(excel_file)
        df_n["sub_program"] = df_n["sub_program"].astype(int).astype(str).str.zfill(4)
        df_n = df_n.rename(columns={"n": "ban_n"})
        # Create a categorical type with the original order
        # df_n["sub_program"] = pd.Categorical(
        #     df_n["sub_program"], categories=sub_prg, ordered=True
        # )

        # Sort by the categorical column
        return df_n  # .sort_values("sub_program")
    except Exception as e:
        # 如果讀取檔案時發生任何錯誤，返回空的 DataFrame
        print(f"警告: 無法讀取檔案 {excel_file}: {e}")
        return pd.DataFrame(columns=["sub_program", "ban_n"])


def parse_folder_names_to_dataframe(directory_path):
    """
    解析指定目錄下的文件夾名稱，並將它們以 # 分隔生成 pandas DataFrame

    參數:
    directory_path - 要解析的目錄路徑

    返回:
    pandas DataFrame，每列對應一個文件夾，每列包含以 # 分隔的各部分
    """

    # 假設格式為: 部門#產品類型#夾位#機台ID#日期
    column_mapping = {
        "part_1": "floor",
        "part_2": "product_type",
        "part_3": "clamping",
        "part_4": "machine_id",
        "part_5": "date",
    }

    # 檢查目錄是否存在
    if not os.path.exists(directory_path):
        print(f"警告: 路徑不存在 '{directory_path}'")
        return pd.DataFrame(columns=column_mapping.values())

    # 存儲文件夾信息
    folder_data = []

    # 獲取所有文件夾
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # 確保是目錄
        if os.path.isdir(item_path):
            # 以 # 分隔文件夾名稱
            parts = item.split("#")

            # 創建一個字典，用於存儲分隔後的各部分
            folder_info = {}

            # 為每個部分分配一個列名
            for i, part in enumerate(parts):
                column_name = f"part_{i+1}"
                folder_info[column_name] = part

            # 添加原始文件夾名稱
            folder_info["folder_name"] = item

            # 添加到列表
            folder_data.append(folder_info)

    # 創建 DataFrame
    df = pd.DataFrame(folder_data)

    # 如果沒有找到文件夾，返回空 DataFrame
    if df.empty:
        return pd.DataFrame(columns=column_mapping.values())

    # 重命名存在的列
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    return df
