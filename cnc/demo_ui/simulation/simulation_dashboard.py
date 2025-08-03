import streamlit as st
import subprocess
import glob
import os
import pandas as pd
from pathlib import Path, PurePath
import re
import datetime
import json
import platform

from cnc_genai.demo_ui.simulation.cad_viewer import display_stl
from cnc_genai.src.simulation.utils import SimulationStatusReader


def render_simulation_dashboard():
    st.header("CNC代碼仿真平台")

    st.subheader("現有仿真任務進度")

    df = get_simulation_tasks()

    # 初始化會話狀態
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False
    if "tasks_to_delete" not in st.session_state:
        st.session_state.tasks_to_delete = []
    if "selected_clamping" not in st.session_state:
        st.session_state.selected_clamping = "請選擇"
    if "confirm_continue" not in st.session_state:
        st.session_state.confirm_continue = False
    if "continue_task_info" not in st.session_state:
        st.session_state.continue_task_info = None
    if "continue_action_type" not in st.session_state:
        st.session_state.continue_action_type = None

    # 配置DataFrame顯示
    edited_df = st.data_editor(
        df,
        column_config={
            "任務進度": st.column_config.ProgressColumn(
                "任務進度",
                help="仿真任務進度",
                format="%.0f%%",
                min_value=0,
                max_value=100,
            ),
            "任務狀態": st.column_config.TextColumn(
                "任務狀態", help="任務當前狀態", width="medium"
            ),
            "進程ID": st.column_config.TextColumn(
                "進程ID", help="後台仿真進程的系統ID", width="small"
            ),
            "進程狀態": st.column_config.TextColumn(
                "進程狀態", help="後台進程運行狀態", width="small"
            ),
            "終止任務": st.column_config.CheckboxColumn(
                "終止任務", help="選擇您要終止的任務"
            ),
            "cmd": None,  # 隱藏 cmd 列
        },
        hide_index=True,
        use_container_width=True,
        height=len(df) * 35 + 38,  # 每行35像素 + 表頭38像素
        disabled=[
            "事業處",
            "用戶",
            "夾位",
            "仿真精度",
            "開始時間",
            "完成時間",
            "耗時(小時)",
            "任務數",
            "任務進度",
            "任務狀態",
            "進程ID",
            "進程狀態",
            "cmd",
        ],
        key="simulation_table",
    )

    # Store edited_df in session state for access by other functions
    st.session_state.edited_df = edited_df

    # 增加CSS樣式使表格字體更大
    st.markdown(
        """
    <style>
    .stDataFrame {
        font-size: 1.2rem !important;
    }
    .stDataFrame th {
        font-size: 1.3rem !important;
        font-weight: bold !important;
    }
    .stDataFrame td {
        font-size: 1.2rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("返回CNC360 V1首頁", use_container_width=True):
            st.session_state.current_page = "landing"
            st.rerun()
    with col2:
        if st.button("刷新任務進度", use_container_width=True):
            st.rerun()
    with col3:
        selected_tasks = edited_df[edited_df["終止任務"] == True]

        # 顯示終止任務按鈕和確認流程
        if not st.session_state.confirm_delete:
            if st.button("終止並刪除選中的任務", use_container_width=True):
                if not selected_tasks.empty:
                    st.session_state.tasks_to_delete = []
                    for _, row in selected_tasks.iterrows():
                        department = row["事業處"]
                        clamping = row["夾位"]
                        process_id = row["進程ID"]
                        username = row["用戶"]
                        st.session_state.tasks_to_delete.append(
                            (department, clamping, process_id, username)
                        )

                    st.session_state.confirm_delete = True
                    st.rerun()
                else:
                    st.warning("請先選擇要終止的任務")
        else:
            # 顯示確認信息
            st.warning("⚠️ 警告：此操作將永久刪除所選夾位的仿真數據，且無法恢復！")

            # 顯示要刪除的任務列表
            for dept, clamp, process_id, username in st.session_state.tasks_to_delete:
                st.error(
                    f"即將刪除：{dept} 的 {clamp} 夾位，進程ID: {process_id}，用戶: {username}"
                )

            # 確認按鈕和取消按鈕
            col_cancel, col_confirm = st.columns(2)
            with col_cancel:
                if st.button("取消操作", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.session_state.tasks_to_delete = []
                    st.rerun()

            with col_confirm:
                if st.button("確認刪除", type="primary", use_container_width=True):
                    for (
                        dept,
                        clamp,
                        process_id,
                        username,
                    ) in st.session_state.tasks_to_delete:
                        if username != st.session_state.username:
                            st.error("您無權終止其他用戶的任務")
                            return
                        else:
                            print(
                                f"終止任務：事業處={dept}, 夾位={clamp}，進程ID={process_id}，用戶={username}"
                            )
                            # 終止對應進程 kill PID
                            try:
                                subprocess.run(
                                    ["kill", "-9", f"{process_id}"],
                                    check=True,
                                )
                            except:
                                pass

                            # 執行刪除操作
                            subprocess.run(
                                [
                                    "rm",
                                    "-rf",
                                    f"../app/{dept}/simulation_master/{clamp}",
                                ],
                                check=True,
                            )

                    st.session_state.confirm_delete = False
                    st.session_state.tasks_to_delete = []
                    st.success("所選任務已成功終止！")
                    st.rerun()

    with col4:
        selected_tasks_continue = edited_df[edited_df["終止任務"] == True]

        # 顯示繼續任務按鈕和確認流程
        if not st.session_state.confirm_continue:
            if st.button("繼續選中的任務", use_container_width=True):
                # 如果選中的任務大於一個，請報錯，提示用戶僅選擇一個任務
                if len(selected_tasks_continue) > 1:
                    st.error("請僅選擇一個任務")
                    return
                # 如果選中的任務小於一個，請報錯，提示用戶請選擇一個任務
                if len(selected_tasks_continue) < 1:
                    st.error("請選擇一個任務")
                    return

                # 如果選中的任務用戶不是當前用戶，請報錯，提示用戶無權繼續其他用戶的任務
                if selected_tasks_continue["用戶"].iloc[0] != st.session_state.username:
                    st.error("您無權繼續其他用戶的任務")
                    return

                # 如果選中的任務等於一個，則判斷該任務進度狀態
                task_status = selected_tasks_continue["任務狀態"].iloc[0]
                st.session_state.continue_task_info = selected_tasks_continue

                if "已完成" in task_status or "運行中" in task_status:
                    if "已完成" in task_status:
                        st.warning("該任務已完成，請問是否要重新啟動仿真？")
                    elif "運行中" in task_status:
                        st.warning("該任務正在運行，請問是否要重新啟動仿真？")

                    st.session_state.continue_action_type = "restart"
                    st.session_state.confirm_continue = True
                    st.rerun()
                elif "異常終止" in task_status:
                    st.session_state.continue_action_type = "resume"
                    st.session_state.confirm_continue = True
                    st.rerun()
                else:
                    st.error("該任務狀態無法繼續")
        else:
            # 顯示確認信息
            task_info = st.session_state.continue_task_info
            task_status = task_info["任務狀態"].iloc[0]

            if st.session_state.continue_action_type == "restart":
                if task_status == "已完成":
                    st.warning("該任務已完成，請問是否要重新啟動仿真？")
                elif task_status == "運行中":
                    st.warning("該任務正在運行，請問是否要重新啟動仿真？")

                # 兩個按鈕"重新仿真"and"取消"
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("重新仿真", use_container_width=True):
                        # 讀取到之前的cmd，並且直接啟動，將新的PID更新到process_info.json
                        setup_task_run(task_info, check_point=None)
                        st.session_state.confirm_continue = False
                        st.session_state.continue_task_info = None
                        st.session_state.continue_action_type = None
                        st.rerun()
                with col2:
                    if st.button("取消", use_container_width=True):
                        st.session_state.confirm_continue = False
                        st.session_state.continue_task_info = None
                        st.session_state.continue_action_type = None
                        st.rerun()

            elif st.session_state.continue_action_type == "resume":
                checkpoint = task_status.split("到")[1]
                st.warning(f"是否繼續從{checkpoint}子程式開始仿真？")

                # 三個按鈕，"重新從頭仿真"，"從異常終止處仿真"，"取消"
                col1, col2, col3 = st.columns(3)
                result = False
                with col1:
                    if st.button("從頭開始", use_container_width=True):
                        result = setup_task_run(task_info, check_point=None)
                        st.session_state.confirm_continue = False
                        st.session_state.continue_task_info = None
                        st.session_state.continue_action_type = None
                with col2:
                    if st.button("從斷點處繼續", use_container_width=True):
                        result = setup_task_run(task_info, check_point=checkpoint)
                        st.session_state.confirm_continue = False
                        st.session_state.continue_task_info = None
                        st.session_state.continue_action_type = None
                with col3:
                    if st.button("取消", use_container_width=True):
                        st.session_state.confirm_continue = False
                        st.session_state.continue_task_info = None
                        st.session_state.continue_action_type = None
                        st.rerun()
                if result:
                    if checkpoint is None:
                        st.success(
                            "仿真任務已重新發佈，仿真任務耗時數小時，請刷新查看進度仿真任務已發佈，進程ID: {st.session_state['process_pid']}"
                        )
                    else:
                        st.success(
                            f"仿真任務已重新發佈，從斷點{checkpoint}處繼續，仿真任務耗時數小時，請返回主頁點擊查看進度，進程ID: {st.session_state['process_pid']}"
                        )

    st.divider()

    # CAD三維視圖校驗部分
    render_cad_validation_section()


def render_cad_validation_section():
    """Render the CAD validation section with 3D visualization."""
    st.subheader("仿真輸出CAD三維視圖校驗")

    # Get completed tasks
    completed_tasks = st.session_state.edited_df[
        st.session_state.edited_df["任務狀態"] == "已完成"
    ]
    completed_clampings = completed_tasks["夾位"].tolist()

    # Add default option to dropdown
    clamping_options = ["請選擇"] + completed_clampings

    # Show the dropdown
    selected_clamping = st.selectbox(
        "選擇要校驗的夾位",
        options=clamping_options,
        index=0,
        key="cad_validation_select",
    )

    # Show the button after selection
    if st.button("顯示CAD三維視圖", use_container_width=True, key="show_cad_button"):
        if selected_clamping != "請選擇":
            try:
                # Get selected task info
                selected_task = completed_tasks[
                    completed_tasks["夾位"] == selected_clamping
                ].iloc[0]

                # Build ZST file path
                zst_dir = f"../app/{selected_task['事業處']}/simulation_master/{selected_task['夾位']}/simulation/latest/"

                if not os.path.exists(zst_dir):
                    st.error(f"目錄不存在: {zst_dir}")
                    return

                zst_files = glob.glob(f"{zst_dir}/*.zst")
                if not zst_files:
                    st.error(f"在目錄中找不到ZST文件: {zst_dir}")
                    return

                zst_path = zst_files[0]

                # Display in columns
                left_col, center_col, right_col = st.columns([1, 2, 1])
                with center_col:
                    display_stl(zst_path, debug_mode=False)
            except Exception as e:
                st.error("3D圖示異常")


def get_simulation_tasks():
    """獲取所有仿真任務的狀態資訊"""
    # 掃描所有仿真任務
    simulation_master_dirs = glob.glob("../app/*/simulation_master/*")

    # 收集所有任務數據
    simulation_data = []
    for sim_path in simulation_master_dirs:
        try:
            # 使用 SimulationStatusReader 讀取狀態
            status_reader = SimulationStatusReader(sim_path)

            # 檢查任務是否有效
            if not status_reader.is_task_valid():
                continue

            # 獲取狀態資訊
            status_info = status_reader.get_status_info()
            if status_info is None:
                continue

            # 添加到結果中
            simulation_data.append(
                {
                    "事業處": status_info["department"],
                    "用戶": status_info["username"],
                    "夾位": status_info["clamping"],
                    "仿真精度": status_info["precision"],
                    "開始時間": status_info["start_time"],
                    "完成時間": status_info["finish_time"],
                    "耗時(小時)": status_info["elapsed_time"],
                    "任務數": status_info["task_progress"],
                    "任務進度": status_info["progress_percentage"],
                    "任務狀態": status_info["task_status"],
                    "進程ID": status_info["process_id"],
                    "進程狀態": status_info["process_status"],
                    "終止任務": False,
                    "cmd": status_info["cmd"],
                }
            )

        except Exception as e:
            print(f"[get_simulation_tasks] 處理 {sim_path} 時發生錯誤: {e}")
            continue

    # 創建DataFrame
    df = pd.DataFrame(simulation_data)
    return df


def setup_task_run(selected_tasks, check_point=None):

    if selected_tasks["cmd"].iloc[0] is None:
        st.error("仿真任務指令未記錄，無法繼續")
        return

    base_cmd = selected_tasks["cmd"].iloc[0].split(" ")

    if check_point is None:
        pass
    else:
        base_cmd.append(f"--check_point")
        base_cmd.append(check_point)

    try:
        log_file = f"../app/{selected_tasks['事業處'].iloc[0]}/simulation_master/{selected_tasks['夾位'].iloc[0]}/simulation/latest/rerun_simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        if platform.system() == "Windows":
            # Windows 平台：確保 UTF-8 編碼並創建獨立進程組
            with open(log_file, "w", encoding="utf-8") as log_f:
                # 設置環境變數確保 Python 輸出 UTF-8
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                process = subprocess.Popen(
                    base_cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    cwd=os.getcwd(),
                    env=env,  # 傳遞修改過的環境變數
                    encoding="utf-8",  # 確保子進程使用 UTF-8
                )
                st.session_state["process_pid"] = process.pid
        else:
            # Linux/macOS 平台：通常預設就是 UTF-8
            with open(log_file, "w", encoding="utf-8") as log_f:
                # 設置環境變數（雖然 Linux 通常不需要，但為了一致性）
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                process = subprocess.Popen(
                    base_cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    cwd=os.getcwd(),
                    env=env,
                    encoding="utf-8",
                )
                st.session_state["process_pid"] = process.pid
    except Exception as e:
        st.error(f"啟動仿真任務失敗: {str(e)}")
        st.session_state["simulating"] = False
        return False

    if st.session_state.get("process_pid"):
        try:
            process_info_file = f"../app/{selected_tasks['事業處'].iloc[0]}/simulation_master/{selected_tasks['夾位'].iloc[0]}/simulation/latest/process_info.json"
            # 讀取process_info.json
            with open(process_info_file, "r", encoding="utf-8") as f:
                process_info = json.load(f)
            # 更新process_info.json
            process_info["process_id"] = st.session_state["process_pid"]
            process_info["start_time"] = datetime.datetime.now().strftime(
                "%Y/%m/%d %H:%M:%S"
            )
            process_info["start_timestamp"] = datetime.datetime.now().timestamp()
            process_info["finish_flag"] = False
            process_info["finish_time"] = None
            process_info["finish_timestamp"] = None
            process_info["cmd"] = selected_tasks["cmd"].iloc[0]
            # 寫回process_info.json
            with open(process_info_file, "w", encoding="utf-8") as f:
                json.dump(process_info, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.warning(f"無法記錄進程信息: {str(e)}")
            return False
