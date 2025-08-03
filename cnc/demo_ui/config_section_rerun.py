import streamlit as st
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import PurePath
import streamlit as st
from cnc_genai.demo_ui.config_tabs import (
    render_parameter_settings_tab,
    render_sub_program_settings_tab,
    render_block_disable_tab,
    render_row_disable_tab,
    render_get_n_from_image_annotation,
    render_get_row_from_image_annotation,
    render_advanced_settings_tab,
)
from cnc_genai.demo_ui import image_annotation
from cnc_genai.src.simulation.utils import load_from_zst
from cnc_genai.demo_ui import conf_init
import warnings
import os
import glob

warnings.filterwarnings("ignore")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_image_path(hyper_params_config, df_sub_program):
    product = st.session_state.selected_clamping
    last_prd = df_sub_program["sub_program"].tolist()[-1]
    dir_path = f"../app/{hyper_params_config['department']}/simulation_master/{product}/simulation/latest/"

    # 使用glob匹配符合條件的文件
    pattern = os.path.join(dir_path, f"{last_prd}_shape=*.zst")
    matching_files = glob.glob(pattern)

    # 如果找到匹配文件，取第一個(可根據需要調整排序邏輯)
    if matching_files:
        # 按文件名排序取最新(假設文件名包含時間戳)
        matching_files.sort(reverse=True)
        return matching_files[0]

    # 如果沒有匹配文件，保持原邏輯
    else:
        raise ValueError(f"找不到符合條件的文件：{pattern}")


def resize_image_to_target_width(image, target_width):
    target_height = int(target_width * image.shape[0] / image.shape[1])
    resize_scale = target_width / image.shape[1]
    resized_image = np.stack(
        [
            image_annotation.numpy_nearest_resize(
                image[:, :, z], (target_height, target_width)
            )
            for z in range(image.shape[2])
        ],
        axis=2,
    )
    return resized_image, resize_scale


def render_config_section_rerun():
    """Render the complete configuration section"""
    with st.container():
        st.subheader("設定優化配置")

        # Create tabs
        tab_names = [
            "關閉特定程序段",
            "關閉特定程序行",
            # "代碼可視化",
            "參數設定",
            "子程式設定",
            "高階默認配置設定",
        ]
        # block_n_tab, block_row_tab, code_vis_tab, prog_tab, param_tab, adv_tab = (
        block_n_tab, block_row_tab, param_tab, prog_tab, adv_tab = st.tabs(tab_names)

        # Only compute the active tab
        with block_n_tab:
            now = datetime.now()
            if block_n_tab.selected:
                col1, col2 = st.columns([2, 3])
                with col1:
                    ban_n_df_out = render_get_n_from_image_annotation(
                        st.session_state.rerun_tab_results["ban_n_df"]
                    )
                with col2:
                    st.session_state.rerun_tab_results["ban_n_df"] = (
                        render_block_disable_tab(ban_n_df_out)
                    )
            # print('block_n_tab', datetime.now() - now)

        with block_row_tab:
            now = datetime.now()
            if block_row_tab.selected:
                col1, col2 = st.columns([2, 3])
                with col1:
                    ban_row_df_out = render_get_row_from_image_annotation(
                        st.session_state.rerun_tab_results["ban_row_df"]
                    )
                with col2:
                    st.session_state.rerun_tab_results["ban_row_df"] = (
                        render_row_disable_tab(ban_row_df_out)
                    )
            # print('block_row_tab', datetime.now() - now)

        # with code_vis_tab:
        #     now = datetime.now()
        #     if code_vis_tab.selected:
        #         col1, col2 = st.columns([1.5, 1])
        #         with col2:
        #             # 从Excel文件读取所有可用的子程序和代码段
        #             df_code = pd.read_excel(
        #                 f"../app/scenario/{session_state.selected_scenario}/load_analysis.xlsx"
        #             )

        #             # 创建子程序选择下拉框
        #             sub_programs = df_code['sub_program'].unique()
        #             selected_sub_program = st.selectbox(
        #                 "選擇子程序",
        #                 options=sub_programs
        #             )

        #             # 创建代码段选择下拉框
        #             code_blocks = df_code[df_code['sub_program'] == selected_sub_program]['N'].unique()
        #             selected_block = st.selectbox(
        #                 "選擇代碼段",
        #                 options=code_blocks
        #             )

        #             # 获取选中代码段的所有相关行
        #             selected_code = df_code[
        #                 (df_code['sub_program'] == selected_sub_program) &
        #                 (df_code['N'] == selected_block)
        #             ]

        #             # 筛选移动代码行
        #             move_code_rows = selected_code[
        #                 selected_code['move_code'].isin(['G01', 'G02', 'G03'])
        #             ]

        #             if not move_code_rows.empty:
        #                 st.write("代碼段內容：")
        #                 st.code(selected_code['src'])

        #         with col1:
        #             if move_code_rows.empty:
        #                 st.warning("該代碼段沒有移動指令")
        #             else:
        #                 # 獲取刀具半徑
        #                 tool_radius = move_code_rows['tool_diameter'].iloc[0] / 2 if 'tool_diameter' in move_code_rows.columns else 0

        #                 # 获取X和Y坐标的范围，並加上刀具半徑
        #                 x_min = move_code_rows['X_pixel'].min() - tool_radius
        #                 x_max = move_code_rows['X_pixel'].max() + tool_radius
        #                 y_min = move_code_rows['Y_pixel'].min() - tool_radius
        #                 y_max = move_code_rows['Y_pixel'].max() + tool_radius
        #                 z_max = move_code_rows['Z_pixel'].max()

        #                 # 构建四个角点的坐标
        #                 bbox_coords = {
        #                     'x_coords': [x_min, x_max],
        #                     'y_coords': [y_min, y_max],
        #                     'z_coords': z_max,
        #                     'tool_radius': tool_radius  # 添加刀具半徑信息
        #                 }

        #                 # 读取并显示图像，在图像上标注代码段对应的区域
        #                 zst_path = get_image_path(hyper_params_config, df_sub_program)
        #                 image_annotation.visualize_code_area(bbox_coords=bbox_coords)
        #     print('code vis tab done...', datetime.now() - now)

        with param_tab:
            now = datetime.now()
            if param_tab.selected:
                hyper_params_dict = render_parameter_settings_tab(
                    st.session_state.rerun_tab_results["hyper_params_dict"]
                )
                st.session_state.rerun_tab_results["hyper_params_dict"] = (
                    hyper_params_dict
                )
                st.session_state.temp_hyper_params_dict = hyper_params_dict
            # print('hyper_params_dict tab', datetime.now() - now)

        with prog_tab:
            now = datetime.now()
            if prog_tab.selected:
                sub_programs_df = render_sub_program_settings_tab(
                    st.session_state.rerun_tab_results["sub_programs_df"]
                )
                st.session_state.rerun_tab_results["sub_programs_df"] = sub_programs_df
            # print('sub_programs_df tab', datetime.now() - now)

        with adv_tab:
            now = datetime.now()
            if adv_tab.selected:
                advanced_params_dict = render_advanced_settings_tab()
                st.session_state.rerun_tab_results["advanced_params_dict"] = (
                    advanced_params_dict
                )
                if (
                    advanced_params_dict
                    and st.session_state.rerun_tab_results["hyper_params_dict"]
                ):
                    st.session_state.rerun_tab_results["hyper_params_dict"].update(
                        advanced_params_dict
                    )
            # print('advanced_params_dict tab', datetime.now() - now)

        # print('all tab done...', datetime.now() - all_time)

        # Save Configuration Button
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("返回CNC360 V1首頁", use_container_width=True):
                st.session_state.current_page = "landing"
                st.rerun()
        with col2:
            if st.button("儲存設定", use_container_width=True, type="primary"):
                st.session_state["hyper_params_dict"] = (
                    st.session_state.rerun_tab_results["hyper_params_dict"]
                )
                st.session_state["sub_programs_df"] = (
                    st.session_state.rerun_tab_results["sub_programs_df"]
                )
                st.session_state["ban_n_df"] = st.session_state.rerun_tab_results[
                    "ban_n_df"
                ]
                st.session_state["ban_row_df"] = st.session_state.rerun_tab_results[
                    "ban_row_df"
                ]
                # Set flag to indicate configuration is saved
                st.session_state.config_saved = True

            else:
                st.session_state.config_saved = False


def prepare_config_section_rerun(
    selected_scenario_base, sel_method="基於當前代碼版本反饋迭代"
):

    selected_department, selected_scenario = PurePath(selected_scenario_base).parts
    st.session_state.run_type = "existing"
    st.session_state.selected_department = selected_department

    # Parse
    st.session_state.selected_clamping = selected_scenario.split("_")[0]
    st.session_state.selected_scenario = selected_scenario

    # 从process_info.json获取当前clamping的precision
    with open(
        f"../app/{selected_department}/simulation_master/{st.session_state.selected_clamping}/simulation/latest/process_info.json",
        "r",
        encoding="utf-8",
    ) as f:
        process_info = json.load(f)
    st.session_state.precision = process_info.get("precision", 4)

    # 基於當前代碼版本反饋迭代
    if sel_method == "基於當前代碼版本反饋迭代":
        st.session_state.selected_folder = selected_scenario.split("_")[1]
        if st.session_state.selected_folder != "製工標準":
            st.session_state.selected_floor = selected_scenario.split("#")[0]
            st.session_state.selected_machine = selected_scenario.split("#")[3]
            st.session_state.selected_date_version = selected_scenario.split("#")[4]
            st.session_state.baseline_display = f"{st.session_state.selected_floor}-{st.session_state.selected_machine}機台{st.session_state.selected_date_version}版本"
        else:
            st.session_state.selected_floor = "製工標準"
            st.session_state.selected_machine = "製工標準"
            st.session_state.selected_date_version = "製工標準"
            st.session_state.baseline_display = "製工標準"

    # 將當前策略遷移到其他代碼版本
    else:
        machine_df = st.session_state.machine_df.copy()
        machine_df = machine_df[
            machine_df["product_type"]
            == "-".join(st.session_state.selected_clamping.split("-")[:-1])
        ]
        machine_df = machine_df[
            machine_df["clamping"] == st.session_state.selected_clamping.split("-")[-1]
        ].sort_values("date", ascending=False)

        # 檢查是否有機台代碼資料
        if machine_df.empty or machine_df["machine_id"].isna().all():
            st.error(
                f"夾位 {st.session_state.selected_clamping} 沒有可用的機台代碼資料，無法進行策略遷移，請點擊「上傳代碼」進行代碼上傳"
            )
            st.rerun()
            return

        col11, col22, col33, col44 = st.columns(4)
        with col11:
            floor_options = list(machine_df["floor"].unique())
            selected_floor = st.selectbox("選擇樓層", floor_options)
            machine_df = machine_df[machine_df["floor"] == selected_floor]

        with col22:
            line_options = list(machine_df["machine_id"].str[0].unique())
            selected_line = st.selectbox("選擇線", sorted(line_options))
            machine_df = machine_df[
                machine_df["machine_id"].str.startswith(selected_line)
            ]
        with col33:
            # 第三個下拉框：選擇機台
            machine_options = list(machine_df["machine_id"].unique())
            selected_machine = st.selectbox("選擇機台", machine_options)
            machine_df = machine_df[machine_df["machine_id"] == selected_machine]

        with col44:
            date_version_options = list(machine_df["date"].unique())
            selected_date_version = st.selectbox("選擇日期版本", date_version_options)

        st.session_state.selected_folder = machine_df[
            machine_df["date"] == selected_date_version
        ].folder_name.to_list()[0]
        st.session_state.baseline_display = (
            f"{selected_floor}-{selected_machine}機台{selected_date_version}版本"
        )
        st.session_state.selected_scenario = selected_scenario
        st.session_state.selected_floor = selected_floor
        st.session_state.selected_machine = selected_machine
        st.session_state.selected_date_version = selected_date_version

    st.session_state.project_code_name = "_".join(selected_scenario.split("_")[2:])


def prepare_content_section_rerun():
    # prepare excels for this scenario
    try:
        st.session_state.df_product_master = pd.read_excel(
            f"../app/{st.session_state.selected_department}/simulation_master/{st.session_state.selected_clamping}/product_master.xlsx",
            engine="openpyxl",
        )
        st.session_state.df_product_master["sub_program"] = (
            st.session_state.df_product_master["sub_program"].astype(str).str.zfill(4)
        )
        st.session_state.df_tools = pd.read_excel(
            f"../app/{st.session_state.selected_department}/simulation_master/{st.session_state.selected_clamping}/tools.xlsx",
            engine="openpyxl",
        )
        st.session_state.df_load_analysis = pd.read_excel(
            f"../app/{st.session_state.selected_department}/scenario/{st.session_state.selected_scenario}/load_analysis.xlsx"
        )
        st.session_state.df_load_analysis["sub_program"] = (
            st.session_state.df_load_analysis["sub_program"].astype(str).str.zfill(4)
        )
    except UnicodeDecodeError as e:
        st.error(f"文件編碼錯誤：{e}")
        st.error("請確保Excel文件使用正確的編碼格式")
        return
    except Exception as e:
        st.error(f"讀取Excel文件時發生錯誤：{e}")
        return

    # 读取json NC codes
    try:
        code_lines = json.loads(
            open(
                f"../app/{st.session_state.selected_department}/scenario/{st.session_state.selected_scenario}/nc_programs.json",
                "r",
            ).read()
        )
        old_codes = json.loads(
            open(
                f"../app/{st.session_state.selected_department}/scenario/{st.session_state.selected_scenario}/nc_programs_old.json",
                "r",
            ).read()
        )
        st.session_state.code_lines = {
            k.replace("O", ""): v for k, v in code_lines.items()
        }
        st.session_state.old_codes = {
            k.replace("O", ""): v for k, v in old_codes.items()
        }
    except Exception as e:
        st.error(f"讀取NC程式時發生錯誤：{e}")
        return

    # Load initial configs
    hyper_params_dict, sub_programs_df, ban_n_df, ban_row_df = (
        conf_init.load_rerun_conf(
            st.session_state.selected_department, st.session_state.selected_scenario
        )
    )

    st.session_state.use_cnc_knowledge_base = False
    st.session_state.use_cnc_knowledge_base_changed = False

    # Initialize rerun_tab_results
    st.session_state.rerun_tab_results = {
        "hyper_params_dict": hyper_params_dict,
        "sub_programs_df": sub_programs_df,
        "ban_n_df": ban_n_df,
        "ban_row_df": ban_row_df,
    }

    # 讀取夾位的 product_image
    product_image, product_image_origin = load_from_zst(
        input_path=get_image_path(hyper_params_dict, sub_programs_df)
    )
    st.session_state.product_image = product_image
    # resized_image, resize_scale = resize_image_to_target_width(
    #     product_image, target_width=600
    # )
    # st.session_state.product_image = resized_image
    # st.session_state.resize_scale = resize_scale
    st.session_state.product_image_origin = product_image_origin
    st.session_state["bbox_data_ban_n"] = {}
    st.session_state["bbox_data_ban_row"] = {}
    print(
        f"load {st.session_state.selected_clamping} image from zst path done, reset annotation to empty..."
    )
