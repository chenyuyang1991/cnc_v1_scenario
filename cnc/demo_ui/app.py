# -*- coding: utf-8 -*-
from pathlib import PurePath
import streamlit as st
from datetime import datetime
from cnc_genai.demo_ui.styles import apply_custom_style
from cnc_genai.demo_ui.config_section import render_config_section
from cnc_genai.demo_ui.config_section_rerun import (
    render_config_section_rerun,
    prepare_config_section_rerun,
    prepare_content_section_rerun,
)
from cnc_genai.demo_ui.processing import (
    # process_optimization,
    process_optimization_async,
    render_optimization_status,
    is_optimization_complete,
    load_optimization_result,
)
from cnc_genai.demo_ui.summary_display import (
    render_summary_results,
    render_optimization_results,
    render_nc_code,
    render_nc_analysis,
)
from cnc_genai.demo_ui.simulation.create_simulation import render_create_simulation
from cnc_genai.demo_ui.simulation.simulation import render_simulation
from cnc_genai.demo_ui.simulation.simulation_dashboard import (
    render_simulation_dashboard,
)
from cnc_genai.demo_ui.login import render_login_page
from cnc_genai.demo_ui.upload import render_upload_page
from cnc_genai.demo_ui import conf_init
from cnc_genai.demo_ui.data_maintenance import render_data_maintenance_login, render_data_maintenance_page

# from cnc_genai.auth.auth import input_password_and_check
import warnings
import os
import logging
import pandas as pd


# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress Streamlit logging warnings - using CRITICAL level
logging.getLogger("streamlit").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)

# Disable all loggers
for logger_name in logging.root.manager.loggerDict:  # pylint: disable=no-member
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# if not input_password_and_check():
#    raise RuntimeError("password incorrect")


def main():
    # Session timeout management (45 minutes)
    import time

    # Check for stale sessions
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = time.time()

    # Force session cleanup after 45 minutes of inactivity
    # if time.time() - st.session_state.last_activity > 10:  # test
    if time.time() - st.session_state.last_activity > 2700:  # 45 minutes = 2700 seconds
        st.session_state.clear()
        st.session_state.current_page = "login"
        st.rerun()

    # Update activity timestamp on any interaction
    st.session_state.last_activity = time.time()

    # Force dark mode
    st.set_page_config(
        page_title="CNC360", layout="wide", initial_sidebar_state="collapsed"
    )

    # Force dark theme via custom CSS
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #0E1117;
        }
        .stApp {
            background-color: #0E1117;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    apply_custom_style()

    # 初始化登入狀態
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # 初始化頁面導航
    if "current_page" not in st.session_state:
        st.session_state.current_page = "homepage"  # 默認頁面改為主頁

    # 登入頁面
    if st.session_state.current_page == "login":
        render_login_page()
        return
        
    # 數據維護登入頁面
    if st.session_state.current_page == "data_maintenance_login":
        render_data_maintenance_login()
        return

    # 檢查是否已登入 - 只有在非homepage頁面才需要檢查
    if not st.session_state.logged_in and st.session_state.current_page not in ["homepage", "data_maintenance_login", "data_maintenance"]:
        st.session_state.current_page = "login"
        st.rerun()

    # 如果已登入，顯示用戶名和登出按鈕
    if st.session_state.logged_in:
        col1, col2 = st.columns([8, 1])
        with col2:
            if st.button(
                f"登出{st.session_state.username}賬戶", use_container_width=True
            ):
                st.session_state.logged_in = False
                st.session_state.current_page = "homepage"
                st.rerun()

    # # 顯示頁面標題，但homepage頁面有自己的標題樣式
    # if st.session_state.current_page != "homepage":
    #     st.markdown(
    #         f"<h1 style='text-align: center; color: #FFFFFF;'>CNC360 v1 CT時間智能優化助手</h1>",
    #         unsafe_allow_html=True,
    #     )

    # Homepage
    if st.session_state.current_page == "homepage":
        st.markdown(
            "<h1 style='text-align: center; color: #FFFFFF; margin-bottom: 40px; font-size: 4rem;'>歡迎使用CNC360系統</h1>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style='text-align: center; padding: 50px; background-color: rgba(0,145,255,0.1);
                border-radius: 10px; cursor: pointer; height: 300px; display: flex;
                flex-direction: column; justify-content: center; align-items: center;'>
                    <h2><span style='color: yellow;'>V0</span> 智能機台分析平台</h2>
                    <div style='font-size: 72px; margin: 30px 0;'>🌐</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("進入平台", key="platform_btn", use_container_width=True):
                js = f"""
                <script>
                window.open("http://cdmaciii.efoxconn.com/CNC360v0", "_blank");
                </script>
                """
                # window.open("http://74.176.169.29/login?redirect=/", "_blank");
                st.components.v1.html(js, height=0)

        with col2:
            st.markdown(
                """
                <div style='text-align: center; padding: 50px; background-color: rgba(0,255,255,0.1);
                border-radius: 10px; cursor: pointer; height: 300px; display: flex;
                flex-direction: column; justify-content: center; align-items: center;'>
                    <h2><span style='color: yellow;'>V1</span> CT時間優化助手</h2>
                    <div style='font-size: 72px; margin: 30px 0;'>⚙️</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("開始優化", key="optimize_btn", use_container_width=True):
                st.session_state.current_page = "login"
                st.rerun()

        # 添加數據維護入口
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; padding: 15px; background-color: rgba(255,0,0,0.1); border-radius: 8px; cursor: pointer;'>
                <h3 style='margin: 0;'>數據維護管理</h3>
                <div style='font-size: 32px; margin: 10px 0;'>🔧</div>
                <p style='margin: 0; font-size: 14px;'>管理數據文件和標杆機台</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if st.button("進入數據維護", key="data_maintenance", use_container_width=True, type="secondary"):
            st.session_state.current_page = "data_maintenance_login"
            st.rerun()

    # Landing page
    elif st.session_state.current_page == "landing":

        st.markdown(
            "<h1 style='text-align: center; color: #FFFFFF; margin-bottom: 30px;'>CNC360 - CT時間優化助手</h1>",
            unsafe_allow_html=True,
        )

        col1, col2, divider_col, col3 = st.columns([1, 1, 0.05, 1])

        with col1:

            st.markdown(
                """
                <div style='text-align: center; padding: 20px; background-color: rgba(0,255,255,0.1); border-radius: 10px; cursor: pointer;'>
                    <h2>首次CT優化方案</h2>
                    <div style='font-size: 48px;'>⚙️</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")

            # 獲取所有夾位-機台選項
            all_options = conf_init.load_landing_product_options()

            # 漂亮的 section 開頭
            st.markdown(
                "##### <span style='color:yellow'>1. 選擇夾位</span>",
                unsafe_allow_html=True,
            )

            # 創建左右兩列
            col11, col22, col33 = st.columns(3)

            # 第一個下拉框：選擇部門
            with col11:
                departments_options = list(set([x.split("/")[0] for x in all_options]))
                selected_department = st.selectbox("選擇部門", departments_options)

            # 第二個下拉框：根據部門選擇机种
            with col22:
                if selected_department:
                    product_options = sorted(
                        list(
                            set(
                                [
                                    "-".join(x.split("/")[1].split("-")[:-1])
                                    for x in all_options
                                    if x.startswith(selected_department)
                                ]
                            )
                        )
                    )
                else:
                    product_options = sorted(
                        list(
                            set(
                                [
                                    "-".join(x.split("/")[1].split("-")[:-1])
                                    for x in all_options
                                ]
                            )
                        )
                    )
                selected_productname = st.selectbox("選擇機種", product_options)

            # 第三個下拉框：根據机种選擇夾位
            with col33:
                if selected_productname:
                    clamping_options = sorted(
                        list(
                            set(
                                [
                                    x.split("/")[1].split("-")[-1]
                                    for x in all_options
                                    if x.startswith(
                                        f"{selected_department}/{selected_productname}"
                                    )
                                ]
                            )
                        )
                    )
                else:
                    clamping_options = sorted(
                        list(set([x.split("/")[1].split("-")[-1] for x in all_options]))
                    )
                selected_clampingname = st.selectbox("選擇夾位", clamping_options)

            selected_clamping_base = (
                f"{selected_department}/{selected_productname}-{selected_clampingname}"
            )

            if selected_clamping_base:
                selected_department, selected_clamping = PurePath(
                    selected_clamping_base
                ).parts[:2]
            else:
                selected_department = selected_clamping = ""

            # 刪除舊的 product_image
            if "product_image" in st.session_state:
                del st.session_state.product_image

            st.markdown(
                "##### <span style='color:yellow'>2. 選擇代碼版本</span>",
                unsafe_allow_html=True,
            )

            machine_df = st.session_state.machine_df.copy()
            machine_df = machine_df[
                machine_df["product_type"]
                == "-".join(selected_clamping.split("-")[:-1])
            ]
            machine_df = machine_df[
                machine_df["clamping"] == selected_clamping.split("-")[-1]
            ].sort_values("date", ascending=False)

            col11, col22, col33, col44 = st.columns(4)
            with col11:
                floor_options = ["製工標準"] + list(machine_df["floor"].unique())
                selected_floor = st.selectbox("樓層", floor_options)
                if selected_floor != "製工標準":
                    machine_df = machine_df[machine_df["floor"] == selected_floor]
            with col22:
                if selected_floor != "製工標準":
                    line_options = list(machine_df["machine_id"].str[0].unique())
                    selected_line = st.selectbox("線", sorted(line_options))
                    machine_df = machine_df[
                        machine_df["machine_id"].str.startswith(selected_line)
                    ]
                else:
                    selected_line = st.selectbox("線", ["製工標準"])
            with col33:
                # 第三個下拉框：選擇機台
                if selected_floor != "製工標準" and selected_line != "製工標準":
                    machine_options = list(machine_df["machine_id"].unique())
                    selected_machine = st.selectbox("機台", machine_options)
                    machine_df = machine_df[
                        machine_df["machine_id"] == selected_machine
                    ]
                else:
                    selected_machine = st.selectbox("機台", ["製工標準"])
            with col44:
                # 第四個下拉框：選擇日期
                if (
                    selected_floor != "製工標準"
                    and selected_line != "製工標準"
                    and selected_machine != "製工標準"
                ):
                    date_version_options = list(machine_df["date"].unique())
                    selected_date_version = st.selectbox(
                        "版本日期", date_version_options
                    )
                else:
                    selected_date_version = st.selectbox("版本日期", ["製工標準"])

            if selected_date_version != "製工標準":
                selected_folder = machine_df[
                    machine_df["date"] == selected_date_version
                ].folder_name.to_list()[0]
                baseline_display = f"{selected_floor}-{selected_machine}機台{selected_date_version}版本"
            else:
                selected_folder = "製工標準"
                baseline_display = "製工標準"
            st.success(f"已選擇{baseline_display}版本代碼進行提升")

            # 添加一個輸入框用於專項代號名，帶有默認值
            st.markdown(
                "##### <span style='color:yellow'>3. 輸入提速專案名稱</span>",
                unsafe_allow_html=True,
            )
            project_code_name = st.text_input(
                "您可以輸入任意文字，不需要額外標註時間", value="首次實驗"
            )

            col11, col22 = st.columns(2)
            with col22:

                # Button to start the process
                if st.button(
                    "開始提速",
                    key="new_model",
                    use_container_width=True,
                    type="primary",
                    help="點擊開始提速",
                ):
                    st.session_state.run_type = "new"

                    st.session_state.selected_department = selected_department
                    st.session_state.selected_clamping = selected_clamping
                    st.session_state.selected_floor = selected_floor
                    st.session_state.selected_machine = selected_machine
                    st.session_state.selected_date_version = selected_date_version
                    st.session_state.selected_folder = selected_folder
                    st.session_state.baseline_display = baseline_display

                    st.session_state.project_code_name = project_code_name
                    try:
                        st.session_state.df_product_master = pd.read_excel(
                            f"../app/{st.session_state.selected_department}/simulation_master/{st.session_state.selected_clamping}/product_master.xlsx",
                            engine="openpyxl",
                        )
                        st.session_state.df_tools = pd.read_excel(
                            f"../app/{st.session_state.selected_department}/simulation_master/{st.session_state.selected_clamping}/tools.xlsx",
                            engine="openpyxl",
                        )
                    except UnicodeDecodeError as e:
                        st.error(f"文件編碼錯誤：{e}")
                        st.error("請確保Excel文件使用正確的編碼格式")
                        return
                    except Exception as e:
                        st.error(f"讀取Excel文件時發生錯誤：{e}")
                        return

                    # 清除舊的 tab_results
                    st.session_state.tab_results = {}

                    st.session_state.current_page = 0
                    st.rerun()

            with col11:

                if st.button(
                    "上傳代碼",
                    use_container_width=True,
                    help="若找不到機台最新的代碼，可以從這裡將代碼上傳至服務器",
                ):
                    st.session_state.current_page = "upload"
                    st.rerun()

        with col2:
            st.markdown(
                """
                <div style='text-align: center; padding: 20px; background-color: rgba(147,112,219,0.1); border-radius: 10px; cursor: pointer;'>
                    <h2>CT優化方案反饋迭代</h2>
                    <div style='font-size: 48px;'>📦</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")

            scenario_options = conf_init.load_landing_scenario_options()
            df_scenario_options = pd.DataFrame(
                scenario_options, columns=["scenario_name"]
            )
            df_scenario_options["department"] = (
                df_scenario_options["scenario_name"].str.split("/").str[0]
            )
            df_scenario_options["clamping"] = (
                df_scenario_options["scenario_name"]
                .str.split("/")
                .str[1]
                .str.split("_")
                .str[0]
            )
            df_scenario_options["clampingname"] = (
                df_scenario_options["clamping"].str.split("-").str[-1]
            )
            df_scenario_options["product"] = (
                df_scenario_options["scenario_name"]
                .str.split("/")
                .str[1]
                .str.split("-CNC")
                .str[0]
            )
            df_scenario_options["department_clamping"] = (
                df_scenario_options["department"]
                + "/"
                + df_scenario_options["clamping"]
            )
            df_scenario_options["baseline"] = (
                df_scenario_options["scenario_name"]
                .str.split("/")
                .str[1]
                .str.split("_")
                .str[1]
            )
            df_scenario_options["floor"] = (
                df_scenario_options["baseline"].str.split("#").str[0]
            ).fillna("製工標準")
            df_scenario_options["machine"] = (
                df_scenario_options["baseline"].str.split("#").str[3]
            ).fillna("製工標準")

            # 使用apply處理分割後可能不足的情況
            df_scenario_options["line"] = df_scenario_options["baseline"].apply(
                lambda x: x.split("#")[3][0] if len(x.split("#")) > 3 else "製工標準"
            )
            df_scenario_options["date_version"] = (
                df_scenario_options["baseline"].str.split("#").str[4]
            ).fillna("製工標準")

            df_scenario_options["full_scenario_name"] = (
                df_scenario_options["scenario_name"]
                .str.split("/")
                .str[1]
                .str.split("_")
                .str[2:]
                .str.join("_")
            )
            df_scenario_options["last_scenario_name"] = (
                df_scenario_options["scenario_name"]
                .str.split("/")
                .str[1]
                .str.split("_")
                .str[-2]
            )
            df_scenario_options["last_scenario_datetime"] = (
                df_scenario_options["scenario_name"]
                .str.split("/")
                .str[1]
                .str.split("_")
                .str[-1]
            )

            if not scenario_options:  # 處理空選項情況
                st.warning("暫無可用專項代號，請先創建新方案")
                selected_scenario_base = None
            else:
                # 選擇歷史專案的部門和夾位
                st.markdown(
                    "##### <span style='color:yellow'>1. 選擇夾位</span>",
                    unsafe_allow_html=True,
                )
                col11, col22, col33 = st.columns(3)
                with col11:
                    selected_department = st.selectbox(
                        "選擇部門",
                        df_scenario_options["department"].unique(),
                        key="rerun_department",
                    )
                with col22:
                    filtered_df_scenario_options = df_scenario_options[
                        df_scenario_options["department"] == selected_department
                    ]
                    product_options = sorted(
                        filtered_df_scenario_options["product"].unique()
                    )
                    selected_productname = st.selectbox(
                        "選擇機種", product_options, key="rerun_productname"
                    )
                with col33:
                    filtered_df_scenario_options = df_scenario_options[
                        df_scenario_options["product"] == selected_productname
                    ]
                    clamping_options = sorted(
                        filtered_df_scenario_options["clampingname"].unique()
                    )
                    selected_scenario_clampingname = st.selectbox(
                        "選擇夾位", clamping_options, key="rerun_clampingname"
                    )

                selected_scenario_base = f"{selected_department}/{selected_productname}-{selected_scenario_clampingname}"

                filtered_df_scenario_options = filtered_df_scenario_options[
                    df_scenario_options["department_clamping"] == selected_scenario_base
                ]

                # 選擇歷史專案版本
                st.markdown(
                    "##### <span style='color:yellow'>2. 選擇歷史提速專案</span>",
                    unsafe_allow_html=True,
                )
                # 选择楼层线机台日期版本
                col11, col22, col33, col44 = st.columns(4)
                with col11:
                    selected_floor = st.selectbox(
                        "樓層",
                        filtered_df_scenario_options["floor"].unique(),
                        index=0,
                        key="rerun_floor",
                    )
                    filtered_df_scenario_options = filtered_df_scenario_options[
                        filtered_df_scenario_options["floor"] == selected_floor
                    ]
                with col22:
                    selected_line = st.selectbox(
                        "線",
                        filtered_df_scenario_options["line"].unique(),
                        index=0,
                        key="rerun_line",
                    )
                    filtered_df_scenario_options = filtered_df_scenario_options[
                        filtered_df_scenario_options["line"] == selected_line
                    ]
                with col33:
                    selected_machine = st.selectbox(
                        "機台",
                        filtered_df_scenario_options["machine"].unique(),
                        index=0,
                        key="rerun_machine",
                    )
                    filtered_df_scenario_options = filtered_df_scenario_options[
                        filtered_df_scenario_options["machine"] == selected_machine
                    ]
                with col44:
                    selected_date_version = st.selectbox(
                        "提速前版本日期",
                        filtered_df_scenario_options["date_version"].unique(),
                        index=0,
                        key="rerun_date_version",
                    )
                    filtered_df_scenario_options = filtered_df_scenario_options[
                        filtered_df_scenario_options["date_version"]
                        == selected_date_version
                    ]
                filtered_df_scenario_options = filtered_df_scenario_options.sort_values(
                    "last_scenario_datetime", ascending=False
                )

                selected_scenario_name = st.selectbox(
                    "歷史提速專案名稱",
                    filtered_df_scenario_options["full_scenario_name"].unique(),
                    key="rerun_scenario_name",
                )
                selected_scenario_base = filtered_df_scenario_options[
                    filtered_df_scenario_options["full_scenario_name"]
                    == selected_scenario_name
                ]["scenario_name"].to_list()[0]
                # st.info(f"選擇的歷史提速專案名稱為：{selected_scenario_base}")

                # 選擇繼續提速優化或遷移提速策略
                st.markdown(
                    "##### <span style='color:yellow'>3. 選擇 繼續提速優化 或 遷移提速策略</span>",
                    unsafe_allow_html=True,
                )
                sel_method = st.radio(
                    "選擇迭代方式",
                    ("基於當前代碼版本反饋迭代", "將當前策略遷移到其他代碼版本"),
                    horizontal=True,
                )
                # 只在有選擇時執行配置準備
                if selected_scenario_base:
                    prepare_config_section_rerun(selected_scenario_base, sel_method)
                    if sel_method == "基於當前代碼版本反饋迭代":
                        st.success(
                            f"繼續優化{st.session_state.baseline_display}版本代碼"
                        )
                    else:
                        st.success(
                            f"遷移優化策略到{st.session_state.baseline_display}版本代碼"
                        )
                else:
                    st.session_state.baseline_display = "無可用基準版本"

                # 這裡加一個textinput，作為迭代的suffix，這個後綴會添加到senarioname裡去，默認用今天的YYYYMMDDHHMM
                st.markdown(
                    "##### <span style='color:yellow'>4. 輸入迭代實驗標記</span>",
                    unsafe_allow_html=True,
                )
                st.session_state.iteration_suffix = st.text_input(
                    "輸入迭代實驗後標記", value="迭代優化"
                )

                col11, col22 = st.columns(2)
                with col11:
                    if st.button(
                        "查看結果", type="secondary", use_container_width=True
                    ):
                        if (
                            not selected_scenario_base
                            or "/" not in selected_scenario_base
                        ):
                            st.error("請選擇有效的專項代號名")
                        else:
                            # 讀取excels for展示
                            prepare_content_section_rerun()
                            st.session_state.selected_department = (
                                selected_scenario_base.split("/")[0]
                            )
                            st.session_state.scenario_name = (
                                selected_scenario_base.split("/")[-1]
                            )
                            st.session_state.current_page = 2
                            st.write(list(st.session_state.code_lines.keys()))
                            st.rerun()
                with col22:
                    if st.button(
                        "開始迭代",
                        key="iteration",
                        use_container_width=True,
                        type="primary",
                    ):
                        if (
                            not selected_scenario_base
                            or "/" not in selected_scenario_base
                        ):
                            st.error("請選擇有效的專項代號名")
                        else:
                            prepare_content_section_rerun()
                            st.session_state.current_page = "iteration"
                            st.rerun()

        # 添加豎線分隔符
        with divider_col:
            st.markdown(
                """
                <style>
                .column-divider {
                    height: calc(100% - 40px);
                    width: 1px;
                    background-color: rgba(255,255,255,0.2);
                    margin: 20px auto;
                    border-radius: 1px;
                    min-height: 500px;
                }
                </style>
                <div class='column-divider'></div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div style='text-align: center; padding: 20px; background-color: rgba(255,140,0,0.1); border-radius: 10px; cursor: pointer;'>
                    <h2>NC代碼仿真</h2>
                    <div style='font-size: 48px;'>🔧</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")

            st.session_state.simulation_clamping_name = st.text_input(
                "輸入仿真任務名", value="請輸入夾位名稱"
            )

            col11, col22 = st.columns(2)
            with col22:
                if st.button(
                    "新建仿真任務",
                    key="new_simulation",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.current_page = "create_simulation"
                    st.session_state.templates_generated = False
                    st.session_state.validation_completed = False
                    st.rerun()
            with col11:
                if st.button(
                    "查看任務進度",
                    key="simulation_progress",
                    use_container_width=True,
                    type="secondary",
                ):
                    st.session_state.current_page = "simulation_dashboard"
                    st.rerun()

        # 添加返回主頁按鈕
        if st.button("返回CNC360主頁", use_container_width=True):
            st.session_state.current_page = "homepage"
            st.rerun()

    elif st.session_state.current_page == "upload":
        render_upload_page()
        
    elif st.session_state.current_page == "data_maintenance_login":
        render_data_maintenance_login()
        
    elif st.session_state.current_page == "data_maintenance":
        render_data_maintenance_page()

    elif st.session_state.current_page == "simulation_dashboard":
        render_simulation_dashboard()

    elif st.session_state.current_page == "create_simulation":
        render_create_simulation()

    elif st.session_state.current_page == "simulation":
        # 確保simulation_clamping存在於session_state中
        if "simulation_clamping" not in st.session_state:
            st.error("未找到夾位信息，請返回上一頁重新選擇")
            if st.button("返回創建頁面"):
                st.session_state.current_page = "create_simulation"
                st.rerun()
        else:
            if "simulation_config" in st.session_state:
                print(
                    f"simulation_config['clamping_name']: {st.session_state.simulation_config.get('clamping_name', None)}"
                )

            # 確保simulation_config使用正確的夾位名稱
            if (
                "simulation_config" in st.session_state
                and "simulation_clamping" in st.session_state
            ):
                st.session_state.clamping_name = st.session_state.simulation_clamping
                st.session_state.simulation_config["clamping_name"] = (
                    st.session_state.simulation_clamping
                )

            # 渲染仿真頁面
            render_simulation()

    elif st.session_state.current_page == "iteration":
        st.header("CT方案反饋迭代")

        # Ensure all necessary session state variables are initialized
        required_vars = [
            "selected_department",
            "selected_scenario",
            "selected_clamping",
        ]
        if not all(var in st.session_state for var in required_vars):
            st.error("缺少必要的配置信息，請返回上一頁重新選擇")
            if st.button("返回首頁"):
                st.session_state.current_page = "landing"
                st.rerun()
            return

        # Ensure rerun_tab_results is initialized before rendering config section
        if "rerun_tab_results" not in st.session_state:
            try:
                prepare_content_section_rerun()
            except Exception as e:
                st.error(f"初始化配置時發生錯誤：{e}")
                if st.button("返回首頁"):
                    st.session_state.current_page = "landing"
                    st.rerun()
                return

        render_config_section_rerun()
        if st.session_state.config_saved:  # When Save button is clicked
            st.session_state.current_page = 1
            st.rerun()

    # Original pages (數字導航頁面)
    elif st.session_state.current_page in [0, 1, 2]:
        # Create two columns: navigation and content
        nav_col, content_col = st.columns([1.2, 5])

        with nav_col:
            # Add "Return to Home" button at the top
            if st.button("🏠 返回CNC360 V1首頁", use_container_width=True):
                st.session_state.current_page = "landing"
                st.rerun()

            st.markdown('<div class="vertical-nav">', unsafe_allow_html=True)
            st.markdown('<div class="vertical-line"></div>', unsafe_allow_html=True)

            # Navigation items with icons and better spacing
            nav_items = [
                ("⚙️", "1. 設定優化配置"),
                ("📊", "2. 執行優化分析"),
                ("🎯", "3. 優化方案演示"),
            ]

            for i, (icon, text) in enumerate(nav_items):
                # Determine if button should be disabled
                current_page = st.session_state.current_page
                button_disabled = (
                    current_page == 2 and i < 2
                ) or (  # If on page 3, disable pages 1&2
                    current_page == 1 and i == 0
                )  # If on page 2, disable page 1

                if st.button(
                    f"{icon}  {text}",
                    key=f"nav_{i}",
                    use_container_width=True,
                    type="secondary" if i != current_page else "primary",
                    disabled=button_disabled,
                ):
                    st.session_state.current_page = i
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # Content area
        with content_col:

            # Page 0: Configuration
            if st.session_state.current_page == 0:
                st.session_state["active_tab"] = 0
                render_config_section()

                if st.session_state.config_saved:  # When Save button is clicked
                    st.session_state.current_page = 1
                    st.rerun()

            # Page 1: Analysis
            elif st.session_state.current_page == 1:

                st.header("執行優化分析")
                st.markdown(
                    f"##### <span style='color:white'>{st.session_state.selected_clamping} - 基於{st.session_state.baseline_display}代碼提升</span>",
                    unsafe_allow_html=True,
                )

                with st.expander("點擊展開配置詳情"):
                    st.markdown("### 基本信息")
                    st.markdown(f"- Scenario類型: {st.session_state.run_type}")
                    st.markdown(f"- 事業部: {st.session_state.selected_department}")
                    st.markdown(f"- 夾位: {st.session_state.selected_clamping}")
                    st.markdown(f"- 樓層: {st.session_state.selected_floor}")
                    st.markdown(f"- 機台: {st.session_state.selected_machine}")
                    st.markdown(f"- 提升前代碼: {st.session_state.selected_folder}")

                    st.markdown("### 程式單:")
                    st.dataframe(
                        st.session_state.df_product_master, use_container_width=True
                    )
                    st.markdown("### 刀具表:")
                    st.dataframe(st.session_state.df_tools, use_container_width=True)

                    st.markdown("### 參數配置:")
                    st.json(st.session_state.hyper_params_dict)
                    st.markdown("### 子程式配置")
                    st.dataframe(
                        st.session_state.sub_programs_df, use_container_width=True
                    )
                    st.markdown("### 禁用代碼段")
                    st.dataframe(st.session_state.ban_n_df, use_container_width=True)
                    st.markdown("### 禁用代碼行")
                    st.dataframe(st.session_state.ban_row_df, use_container_width=True)

                col11, col22 = st.columns(2)
                with col11:
                    # 加一个返回上页设置页面的按钮
                    if st.button("返回上一頁", use_container_width=True):
                        st.session_state.current_page = 0
                        st.rerun()
                with col22:
                    if st.button("開始分析", use_container_width=True, type="primary"):

                        # (
                        #     st.session_state.code_lines,
                        #     st.session_state.old_codes,
                        #     st.session_state.df_load_analysis,
                        # ) = process_optimization()
                        # st.write(st.session_state.out_df.head())

                        # Use async optimization
                        (
                            code_lines,
                            old_codes,
                            df_load_analysis,
                        ) = process_optimization_async()

                        # If results are immediately available (already completed)
                        if code_lines is not None:
                            st.session_state.code_lines = code_lines
                            st.session_state.old_codes = old_codes
                            st.session_state.df_load_analysis = df_load_analysis
                            st.session_state.current_page = 2
                            st.rerun()

                # Only render optimization status if we have a current optimization
                if (
                    hasattr(st.session_state, "current_optimization_scenario")
                    and st.session_state.current_optimization_scenario
                    and hasattr(st.session_state, "selected_department")
                    and st.session_state.selected_department
                ):

                    # Always render optimization status to handle polling
                    with col22:
                        render_optimization_status()

                        # Check if optimization just completed and advance to results page
                        if (
                            is_optimization_complete()
                            and not st.session_state.get("optimization_polling", False)
                            and "code_lines" not in st.session_state
                        ):

                            # Load the completed results
                            results = load_optimization_result()
                            if results and results.get("success", False):
                                st.session_state.code_lines = {
                                    k.replace("O", ""): v
                                    for k, v in results["new_codes"].items()
                                }
                                st.session_state.old_codes = {
                                    k.replace("O", ""): v
                                    for k, v in results["old_codes"].items()
                                }
                                st.session_state.df_load_analysis = results["out_df"]
                                st.session_state.current_page = 2
                                st.success("✅ 優化完成！正在跳轉到結果頁面...")
                                st.rerun()

            # Page 2: Results
            elif st.session_state.current_page == 2:
                st.header("優化方案演示")
                st.markdown(
                    f"##### <span style='color:white'>已保存Scenario - {st.session_state.scenario_name}</span>",
                    unsafe_allow_html=True,
                )

                tabs = st.tabs(
                    ["期望結果匯總", "優化結果可視化", "NC代碼輸出", "NC代碼分析"]
                )

                with tabs[0]:
                    render_summary_results()

                with tabs[1]:
                    render_optimization_results()

                with tabs[2]:
                    render_nc_code()

                with tabs[3]:
                    render_nc_analysis()

    # 如果沒有匹配到任何頁面，返回到homepage
    else:
        st.warning(f"未知頁面: {st.session_state.current_page}")
        st.session_state.current_page = "homepage"
        st.rerun()


if __name__ == "__main__":
    main()
