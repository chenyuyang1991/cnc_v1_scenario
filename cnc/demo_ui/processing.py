import warnings
import json
import subprocess
import platform
import joblib
import time

warnings.filterwarnings("ignore")
from cnc_genai.src.v1_algo.generate_nc_code import run_generate_nc_code
from cnc_genai.src.utils import utils
from datetime import datetime, timezone, timedelta
import os
import pandas as pd
import streamlit as st


def load_config_v1_ui():

    base_config = utils.load_yaml_config("cnc_genai/conf/v1_config.yaml")

    # customized with selected clamping
    base_config["department"] = st.session_state.selected_department
    base_config["clamping_name"] = st.session_state.selected_clamping
    make_scenario_name()
    base_config["scenario_name"] = st.session_state.scenario_name
    base_config["path"]["dir_app"] = base_config["path"]["dir_app"].format(
        department=base_config["department"]
    )
    base_config["path"]["dir_machine_data"] = base_config["path"][
        "dir_machine_data"
    ].format(
        department=st.session_state.selected_department,
        folder=st.session_state.selected_folder,
    )

    # load precision from process_info.json
    process_info_path = f"{base_config['path']['dir_app']}/{base_config['clamping_name']}/{base_config['path']['dir_simulation']}/process_info.json"
    with open(process_info_path, "r") as f:
        process_info = json.load(f)
    base_config["precision"] = process_info.get("precision", 4)

    # customized with configured settings
    specialized_config = utils.process_v1_sp_config(
        df_sub_prog=st.session_state["sub_programs_df"],
        ban_n_dict=st.session_state["ban_n_df"]
        .groupby("sub_program")["ban_n"]
        .apply(list)
        .to_dict(),
        ban_row_dict=st.session_state["ban_row_df"]
        .groupby("sub_program")["ban_row"]
        .apply(list)
        .to_dict(),
        hyper_params_dict=st.session_state["hyper_params_dict"],
    )

    # base_config = utils.merge_configs(base_config, st.session_state["input_path"])
    conf = utils.merge_configs(base_config, specialized_config)
    return conf


def save_scenario_config(
    department: str,
    scenario_name: str,
    hyper_params: pd.DataFrame,
    sub_programs: pd.DataFrame,
    ban_n: pd.DataFrame,
    ban_row: pd.DataFrame,
    bboxes_n: dict,
    bboxes_row: dict,
    out_df: pd.DataFrame,
):
    """
    Save configuration to Excel file in a scenario folder.

    Args:
        scenario_name (str): Name of the scenario
        hyper_params (pd.DataFrame): Hyperparameters DataFrame
        sub_programs (list): List of sub-programs
        ban_n (pd.DataFrame): Ban_n DataFrame

    Returns:
        str: Path to the created Excel file
    """
    # Create scenario folder in a cross-platform way
    scenario_folder = f"../app/{department}/scenario/{scenario_name}"
    os.makedirs(scenario_folder, exist_ok=True)

    # Create Excel file path
    excel_path = os.path.join(scenario_folder, f"{scenario_name}.xlsx")

    # Create Excel writer object
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        header_row = pd.DataFrame(["Configuration"]).T

        # Save hyper_params sheet
        header_row.to_excel(writer, sheet_name="hyper_params", index=False)
        hyper_params.to_excel(
            writer, sheet_name="hyper_params", startrow=1, index=False
        )

        # Save sub_programs sheet
        header_row.to_excel(writer, sheet_name="sub_program", index=False)
        pd.DataFrame(sub_programs).to_excel(
            writer, sheet_name="sub_program", startrow=1, index=False
        )

        # Save ban_n sheet
        header_row.to_excel(writer, sheet_name="ban_n", index=False)
        ban_n = ban_n.explode("ban_n")
        ban_n = ban_n[~pd.isna(ban_n["ban_n"])]
        ban_n.to_excel(writer, sheet_name="ban_n", startrow=1, index=False)

        # Save ban_row sheet
        header_row.to_excel(writer, sheet_name="ban_row", index=False)
        ban_row.to_excel(writer, sheet_name="ban_row", startrow=1, index=False)

        # Save bboxes_n sheet
        header_row.to_excel(writer, sheet_name="bboxes_n", index=False)
        if "bboxes_n_prev" in st.session_state:
            bboxes_n_df = pd.concat(
                [
                    st.session_state.bboxes_n_prev,
                    convert_bbox_to_df(bboxes_n, scenario_name),
                ],
                ignore_index=True,
                axis=0,
            )
        else:
            bboxes_n_df = convert_bbox_to_df(bboxes_n, scenario_name)
        bboxes_n_df.to_excel(writer, sheet_name="bboxes_n", startrow=1, index=False)

        # Save bboxes_row sheet
        header_row.to_excel(writer, sheet_name="bboxes_row", index=False)
        if "bboxes_row_prev" in st.session_state:
            bboxes_row_df = pd.concat(
                [
                    st.session_state.bboxes_row_prev,
                    convert_bbox_to_df(bboxes_row, scenario_name),
                ],
                ignore_index=True,
                axis=0,
            )
        else:
            bboxes_row_df = convert_bbox_to_df(bboxes_row, scenario_name)
        bboxes_row_df.to_excel(writer, sheet_name="bboxes_row", startrow=1, index=False)

    out_df.to_excel(os.path.join(scenario_folder, f"load_analysis.xlsx"), index=False)


def convert_bbox_to_df(bbox_data, scenario_name):
    """
    將邊界框數據轉換為結構化的DataFrame

    Args:
        bbox_data (dict): 邊界框數據
        scenario_name (str): 場景名稱
    Returns:
        pd.DataFrame: 轉換後的DataFrame
    """
    # 處理邊界框數據
    all_records = []
    if bbox_data:
        for sub_program, z_data in bbox_data.items():
            for z_value, data in z_data.items():
                for i, bbox in enumerate(data.get("bboxes", [])):
                    label = (
                        data.get("labels", [])[i]
                        if i < len(data.get("labels", []))
                        else None
                    )
                    all_records.append(
                        {
                            "子程序": sub_program,
                            "Z坐標(mm)": z_value,
                            "邊界框索引": i + 1,
                            "左上X": bbox[0],
                            "左上Y": bbox[1],
                            "右下X": bbox[2],
                            "右下Y": bbox[3],
                            "標籤": label,
                            "場景": scenario_name,
                        }
                    )

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def ensure_session_state_for_optimization():
    """Ensure all required session state variables are available for optimization"""

    # Default values if missing
    defaults = {
        "run_type": "new",
        "selected_folder": "製工標準",
        "project_code_name": "default_project",
    }

    for key, default_value in defaults.items():
        if not hasattr(st.session_state, key) or not getattr(st.session_state, key):
            st.session_state[key] = default_value
            st.warning(f"Session state變數 '{key}' 未設置，使用默認值: {default_value}")


def make_scenario_name():

    # 使用 UTC 时间加 8 小时取得台灣時間
    timestamp = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime(
        "%Y-%m-%d %H%M"
    )
    if st.session_state.run_type == "new":
        scenario_name = (
            f"{st.session_state.selected_clamping}_"
            + f"{st.session_state.selected_folder}_"
            + f"{st.session_state.project_code_name}_{timestamp}"
        )
    elif st.session_state.run_type == "existing":
        # 确保包含时间戳和迭代序号
        scenario_name = (
            f"{st.session_state.selected_scenario}_"
            + f"{st.session_state.iteration_suffix}_"
            + f"{timestamp}"
        )
    else:
        scenario_name = "Unknown"

    st.session_state.scenario_name = scenario_name


def process_optimization():

    conf = load_config_v1_ui()
    if st.session_state.selected_folder == "製工標準":
        conf["path"]["dir_machine_folder"] = "製工標準"
    else:
        conf["path"]["dir_machine_folder"] = "機台代碼"

    # st.write(conf)

    new_codes, old_codes, out_df = run_generate_nc_code(conf)

    # convert back to df
    hyper_params_df = pd.DataFrame([st.session_state.hyper_params_dict])

    save_scenario_config(
        department=st.session_state.selected_department,
        scenario_name=st.session_state.scenario_name,
        hyper_params=hyper_params_df,
        sub_programs=st.session_state.sub_programs_df,
        ban_n=st.session_state.ban_n_df,
        ban_row=st.session_state.ban_row_df,
        bboxes_n=(
            st.session_state.bbox_data_ban_n
            if "bbox_data_ban_n" in st.session_state
            else {}
        ),
        bboxes_row=(
            st.session_state.bbox_data_ban_row
            if "bbox_data_ban_row" in st.session_state
            else {}
        ),
        out_df=out_df,
    )

    new_codes = {k.replace("O", ""): v for k, v in new_codes.items()}
    old_codes = {k.replace("O", ""): v for k, v in old_codes.items()}

    return new_codes, old_codes, out_df


def get_scenario_folder_path():
    """Get the path to the current scenario folder"""
    # Check if required session state variables exist
    if not hasattr(st.session_state, "selected_department") or not hasattr(
        st.session_state, "scenario_name"
    ):
        return None
    if not st.session_state.selected_department or not st.session_state.scenario_name:
        return None
    return f"../app/{st.session_state.selected_department}/scenario/{st.session_state.scenario_name}"


def get_optimization_paths():
    """Get paths for optimization config and result files"""
    scenario_folder = get_scenario_folder_path()
    if scenario_folder is None:
        return None, None, None, None

    config_path = os.path.join(scenario_folder, "optimization_config.json")
    result_path = os.path.join(scenario_folder, "optimization_result.joblib")
    log_path = os.path.join(scenario_folder, "optimization_log.txt")
    process_info_path = os.path.join(scenario_folder, "optimization_process_info.json")

    return config_path, result_path, log_path, process_info_path


def is_optimization_running():
    """Check if optimization is currently running"""
    # Use current optimization scenario if available, otherwise fall back to general check
    if (
        hasattr(st.session_state, "current_optimization_scenario")
        and st.session_state.current_optimization_scenario
    ):
        scenario_folder = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.current_optimization_scenario}"
        process_info_path = os.path.join(
            scenario_folder, "optimization_process_info.json"
        )
    else:
        _, _, _, process_info_path = get_optimization_paths()

    if process_info_path is None or not os.path.exists(process_info_path):
        return False

    try:
        with open(process_info_path, "r", encoding="utf-8") as f:
            process_info = json.load(f)

        # Check if process is marked as finished
        if process_info.get("finish_flag", False):
            return False

        # Check if process is still running (basic check)
        process_id = process_info.get("process_id")
        if process_id:
            try:
                # Check if process exists (cross-platform)
                if platform.system() == "Windows":
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {process_id}"],
                        capture_output=True,
                        text=True,
                    )
                    return str(process_id) in result.stdout
                else:
                    result = subprocess.run(
                        ["ps", "-p", str(process_id)], capture_output=True, text=True
                    )
                    return result.returncode == 0
            except:
                pass

        return False
    except:
        return False


def is_optimization_complete():
    """Check if optimization is complete and result file exists"""
    # Use current optimization scenario if available, otherwise fall back to general check
    if (
        hasattr(st.session_state, "current_optimization_scenario")
        and st.session_state.current_optimization_scenario
    ):
        scenario_folder = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.current_optimization_scenario}"
        result_path = os.path.join(scenario_folder, "optimization_result.joblib")
    else:
        _, result_path, _, _ = get_optimization_paths()

    if result_path is None:
        return False
    return os.path.exists(result_path)


def load_optimization_result():
    """Load optimization result from joblib file"""
    # Use current optimization scenario if available, otherwise fall back to general check
    if (
        hasattr(st.session_state, "current_optimization_scenario")
        and st.session_state.current_optimization_scenario
    ):
        scenario_folder = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.current_optimization_scenario}"
        result_path = os.path.join(scenario_folder, "optimization_result.joblib")
    else:
        _, result_path, _, _ = get_optimization_paths()

    if result_path is None or not os.path.exists(result_path):
        return None

    try:
        results = joblib.load(result_path)

        # Convert out_df back to DataFrame if it exists
        if results.get("out_df") is not None:
            results["out_df"] = pd.DataFrame(results["out_df"])

        return results
    except Exception as e:
        # st.error(f"Error loading optimization results: {str(e)}")
        return None


def start_optimization_async():
    """Start optimization process asynchronously"""

    # Ensure required session state variables are available
    ensure_session_state_for_optimization()

    # Debug session state to see what's missing
    debug_info = []

    # Check required variables for make_scenario_name()
    base_required_vars = ["selected_department", "run_type"]
    missing_vars = []

    # Check base variables first
    for var in base_required_vars:
        if not hasattr(st.session_state, var):
            missing_vars.append(f"{var} (not set)")
        elif not getattr(st.session_state, var):
            missing_vars.append(f"{var} (empty)")
        else:
            debug_info.append(f"{var}: {getattr(st.session_state, var)}")

    # Check run_type specific variables
    if hasattr(st.session_state, "run_type") and st.session_state.run_type:
        if st.session_state.run_type == "new":
            scenario_vars = [
                "selected_clamping",
                "selected_folder",
                "project_code_name",
            ]
        elif st.session_state.run_type == "existing":
            scenario_vars = ["selected_scenario", "iteration_suffix"]
        else:
            scenario_vars = []

        for var in scenario_vars:
            if not hasattr(st.session_state, var):
                missing_vars.append(f"{var} (not set)")
            elif not getattr(st.session_state, var):
                missing_vars.append(f"{var} (empty)")
            else:
                debug_info.append(f"{var}: {getattr(st.session_state, var)}")

    if missing_vars:
        st.error(f"錯誤：缺少必要的session state變數: {', '.join(missing_vars)}")
        with st.expander("Debug Info"):
            st.write("Available variables:")
            for info in debug_info:
                st.write(f"- {info}")
            st.write("Missing variables:")
            for var in missing_vars:
                st.write(f"- {var}")
        return False, None

    # Use the scenario name that's already been set
    if (
        not hasattr(st.session_state, "current_optimization_scenario")
        or not st.session_state.current_optimization_scenario
    ):
        st.error("錯誤：無法獲取scenario名稱，請確保已生成scenario")
        return False, None

    unique_scenario_name = st.session_state.current_optimization_scenario

    # Check if scenario folder path can be created
    scenario_folder = (
        f"../app/{st.session_state.selected_department}/scenario/{unique_scenario_name}"
    )
    if not st.session_state.selected_department:
        st.error("錯誤：無法獲取scenario資訊，請確保已選擇部門和scenario")
        return False, None

    conf = load_config_v1_ui()
    if st.session_state.selected_folder == "製工標準":
        conf["path"]["dir_machine_folder"] = "製工標準"
    else:
        conf["path"]["dir_machine_folder"] = "機台代碼"

    # Create scenario folder
    os.makedirs(scenario_folder, exist_ok=True)

    # Get paths for this specific optimization run
    config_path = os.path.join(scenario_folder, "optimization_config.json")
    result_path = os.path.join(scenario_folder, "optimization_result.joblib")
    log_path = os.path.join(scenario_folder, "optimization_log.txt")
    process_info_path = os.path.join(scenario_folder, "optimization_process_info.json")

    # Save configuration to JSON file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2, default=str)

    # Remove old result file if exists
    if os.path.exists(result_path):
        os.remove(result_path)

    # Prepare command
    script_path = "cnc_genai/demo_ui/run_optimization_async.py"
    base_cmd = [
        "python",
        "-u",
        script_path,
        "--config_path",
        config_path,
        "--output_path",
        result_path,
    ]

    # Start background process
    try:
        if platform.system() == "Windows":
            # Windows platform
            with open(log_path, "w", encoding="utf-8") as log_f:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                process = subprocess.Popen(
                    base_cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    cwd=os.getcwd(),
                    env=env,
                    encoding="utf-8",
                )
                process_id = process.pid
        else:
            # Linux/macOS platform
            with open(log_path, "w", encoding="utf-8") as log_f:
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
                process_id = process.pid

        # Save process info
        process_info = {
            "process_id": process_id,
            "username": st.session_state.get("username", "unknown"),
            "start_time": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "start_timestamp": datetime.now().timestamp(),
            "finish_flag": False,
            "finish_time": None,
            "finish_timestamp": None,
            "cmd": " ".join(base_cmd),
            "scenario_name": st.session_state.scenario_name,
        }

        with open(process_info_path, "w", encoding="utf-8") as f:
            json.dump(process_info, f, ensure_ascii=False, indent=2)

        st.session_state["optimization_process_id"] = process_id
        st.session_state["optimization_started"] = True

        return True, process_id

    except Exception as e:
        st.error(f"Failed to start optimization: {str(e)}")
        return False, None


def process_optimization_async():
    """
    Async version of process_optimization that starts background processing
    and returns immediately. UI should poll for completion.
    """

    # Use existing scenario name if already generated, or create new one
    if (
        not hasattr(st.session_state, "current_optimization_scenario")
        or not st.session_state.current_optimization_scenario
    ):
        # Generate scenario name only if not already set
        make_scenario_name()
        st.session_state.current_optimization_scenario = st.session_state.scenario_name

    # Check if optimization is already running for THIS scenario
    if is_optimization_running():
        st.warning("優化任務正在運行中，請等待完成")
        return None, None, None

    # Start new optimization with fresh scenario name
    success, process_id = start_optimization_async()

    if success:
        st.success(f"優化任務已啟動，進程ID: {process_id}")
        st.info("優化正在後台運行，請在綫等待進度完成...")

        # Set up automatic rerun every 10 seconds to check for completion
        if not st.session_state.get("optimization_polling", False):
            st.session_state["optimization_polling"] = True
            time.sleep(1)  # Small delay before starting polling
            st.rerun()

    return None, None, None


def render_optimization_status():
    """Render optimization status and handle polling"""

    # Check if we have a current optimization scenario
    if not (
        hasattr(st.session_state, "current_optimization_scenario")
        and st.session_state.current_optimization_scenario
        and hasattr(st.session_state, "selected_department")
        and st.session_state.selected_department
    ):
        return  # Don't render anything if no current optimization

    # Check if we should be polling
    if st.session_state.get("optimization_polling", False):

        if is_optimization_complete():
            # Optimization is complete
            st.session_state["optimization_polling"] = False
            st.success("✅ 優化已完成！正在載入結果...")

            # Load the completed results
            results = load_optimization_result()
            if results and results.get("success", False):
                # Save scenario config with the loaded results
                hyper_params_df = pd.DataFrame([st.session_state.hyper_params_dict])

                # Use current optimization scenario name for saving (ensure consistency)
                scenario_name_to_save = st.session_state.current_optimization_scenario
                # Also update the main scenario_name to match
                st.session_state.scenario_name = scenario_name_to_save

                save_scenario_config(
                    department=st.session_state.selected_department,
                    scenario_name=scenario_name_to_save,
                    hyper_params=hyper_params_df,
                    sub_programs=st.session_state.sub_programs_df,
                    ban_n=st.session_state.ban_n_df,
                    ban_row=st.session_state.ban_row_df,
                    bboxes_n=(
                        st.session_state.bbox_data_ban_n
                        if "bbox_data_ban_n" in st.session_state
                        else {}
                    ),
                    bboxes_row=(
                        st.session_state.bbox_data_ban_row
                        if "bbox_data_ban_row" in st.session_state
                        else {}
                    ),
                    out_df=results["out_df"],
                )

                # Store results in session state for page 2
                st.session_state.code_lines = {
                    k.replace("O", ""): v for k, v in results["new_codes"].items()
                }
                st.session_state.old_codes = {
                    k.replace("O", ""): v for k, v in results["old_codes"].items()
                }
                st.session_state.df_load_analysis = results["out_df"]

                # Advance to results page
                st.session_state.current_page = 2
                st.success("✅ 優化完成！正在跳轉到結果頁面...")
            else:
                error_msg = (
                    results.get("error", "Unknown error")
                    if results
                    else "No results found"
                )
                # st.error(f"優化失敗: {error_msg}")

            time.sleep(1)
            st.rerun()

        elif is_optimization_running():
            # Still running
            if (
                hasattr(st.session_state, "current_optimization_scenario")
                and st.session_state.current_optimization_scenario
            ):
                scenario_folder = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.current_optimization_scenario}"
                log_path = os.path.join(scenario_folder, "optimization_log.txt")
                process_info_path = os.path.join(
                    scenario_folder, "optimization_process_info.json"
                )
            else:
                _, _, log_path, process_info_path = get_optimization_paths()

            # Show progress info
            if os.path.exists(process_info_path):
                # try:
                with open(process_info_path, "r", encoding="utf-8") as f:
                    process_info = json.load(f)

                start_time = process_info.get("start_time", "Unknown")
                elapsed = datetime.now().timestamp() - process_info.get(
                    "start_timestamp", datetime.now().timestamp()
                )
                elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"

                st.info(
                    f"🔄 優化正在運行中... \n開始時間: {start_time} \n已運行: {elapsed_str}"
                )

                # TODO: this is good to have showcase (activate it only if necessary)
                # Show recent log entries if available
                # if os.path.exists(log_path):
                #     try:
                #         with open(log_path, 'r', encoding='utf-8') as f:
                #             log_content = f.readlines()

                #         if log_content:
                #             recent_logs = log_content[-5:]  # Show last 5 lines
                #             st.text("最近日誌:")
                #             for log_line in recent_logs:
                #                 st.text(log_line.strip())
                #     except:
                #         pass

                # except:
                #     st.info("🔄 優化正在運行中...")
            else:
                st.info("🔄 優化正在運行中...")

            # Auto-refresh every 20 seconds
            time.sleep(10)
            st.rerun()

        else:
            # Process stopped but no result - probably failed
            st.session_state["optimization_polling"] = False
            st.error("❌ 優化進程似乎已停止，但未找到結果文件。請檢查日誌。")

            # Show log for debugging
            if (
                hasattr(st.session_state, "current_optimization_scenario")
                and st.session_state.current_optimization_scenario
            ):
                scenario_folder = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.current_optimization_scenario}"
                log_path = os.path.join(scenario_folder, "optimization_log.txt")
            else:
                _, _, log_path, _ = get_optimization_paths()

            if os.path.exists(log_path):
                with st.expander("查看錯誤日誌"):
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            log_content = f.read()
                        st.text(log_content)
                    except:
                        st.text("無法讀取日誌文件")
