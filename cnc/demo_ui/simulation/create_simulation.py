import streamlit as st
import pandas as pd
import os
import re
import shutil
import subprocess
from pathlib import PurePath, Path
import base64

from cnc_genai.src.code_parsing.code_parsing import run_code_parsing
from cnc_genai.src.utils.utils import (
    load_config_v1,
    load_raw_gcodes,
    read_gcode_from_json,
)
from cnc_genai.src.code_parsing.generate_template import (
    generate_sub_program_master,
    generate_macros_needed,
    generate_tools,
    generate_macros_and_tools,
    parse_sub_program_master,
)
from cnc_genai.demo_ui import conf_init
from cnc_genai.src.utils.utils import normalize_program_keys


def render_create_simulation():
    """
    渲染新建專案頁面，包括上傳宏變量表格和代碼目錄的功能
    """
    st.header("新建專案")

    st.markdown(
        "#### <span style='color:yellow'>第1步: 上傳代碼</span>",
        unsafe_allow_html=True,
    )

    # 創建默認配置
    config = load_config_v1("./cnc_genai/conf/v1_config.yaml")
    st.session_state.simulation_config = config

    # 初始化 templates_generated 狀態，確保在頁面重新加載時不丟失
    if "templates_generated" not in st.session_state:
        st.session_state.templates_generated = False

    # 初始化 templates 字典
    templates = {}

    if "gcodes" not in st.session_state:
        st.session_state.gcodes = {}

    # 初始化代碼來源
    if "code_source" not in st.session_state:
        st.session_state.code_source = "手工上傳代碼"

    # 創建左中右布局
    col1, col2, col3 = st.columns(3)

    # 初始化文件列表
    file_names = []

    # 左側欄 - 上傳代碼文件和選擇主程序
    with col1:
        # 上傳代碼文件
        st.markdown(
            "##### 上傳代碼文件 <span style='color:yellow'>[必選]</span>",
            unsafe_allow_html=True,
        )

        st.session_state.code_source = st.radio(
            "選擇代碼來源",
            options=["手工上傳代碼", "讀取機台代碼"],
            key="code_source_radio",
            index=0 if st.session_state.code_source == "手工上傳代碼" else 1,
        )

        if st.session_state.code_source == "手工上傳代碼":
            st.markdown("請上傳TXT格式的CNC代碼文件，可以選擇多個文件")

            col11, col22, col33 = st.columns(3)

            with col11:
                # 使用 session_state 中的 clamping_name 作為默認值
                # 確定默認索引
                default_index = 0
                if (
                    "selected_department" in st.session_state
                    and st.session_state.selected_department in ["mac1", "mac2", "mac3"]
                ):
                    default_index = ["mac1", "mac2", "mac3"].index(
                        st.session_state.selected_department
                    )

                st.session_state.selected_department = st.selectbox(
                    "選擇事業處",
                    options=["mac1", "mac2", "mac3"],
                    key="department_select",
                    index=default_index,
                )
                st.session_state.simulation_config["path"]["dir_app"] = (
                    st.session_state.simulation_config["path"]["dir_app"].format(
                        department=st.session_state.selected_department
                    )
                )

            with col22:
                st.session_state.simulation_productname = st.text_input(
                    "機種名稱",
                    value=st.session_state.get("simulation_productname", ""),
                )

            with col33:
                # 使用 session_state 中的 clamping_name 作為默認值
                st.session_state.simulation_clampingname = st.text_input(
                    "夾位名稱",
                    value=st.session_state.get("simulation_clampingname", ""),
                )
                if st.session_state.simulation_clampingname:
                    if "CNC" not in st.session_state.simulation_clampingname:
                        st.error("夾位名稱必須包含'CNC'")

            st.session_state.simulation_clamping_name = f"{st.session_state.simulation_productname}-{st.session_state.simulation_clampingname}"
            st.session_state.simulation_clamping_name = (
                st.session_state.simulation_clamping_name.replace("_", "-").replace(
                    " ", "-"
                )
            )
            st.session_state.simulation_config["clamping_name"] = (
                st.session_state.simulation_clamping_name
            )

            code_files = st.file_uploader(
                "CNC代碼文件",
                accept_multiple_files=True,
                key="code_files_multi",
            )
            if code_files:
                st.session_state.code_files = code_files
            else:
                st.session_state.code_files = []

            if (
                len(st.session_state.code_files)
                and st.session_state.simulation_clamping_name
            ):

                # 顯示上傳的文件數量
                st.success(f"成功上傳 {len(st.session_state.code_files)} 個CNC代碼文件")

                # 添加刪除所有文件的按鈕
                if st.button(
                    "刪除所有已上傳文件",
                    key="delete_all_files",
                    use_container_width=True,
                ):
                    # 清除session state中的文件
                    if "gcodes" in st.session_state:
                        del st.session_state.gcodes
                    if "code_files_multi" in st.session_state:
                        del st.session_state.code_files_multi
                    st.rerun()

                # 保存文件
                file_names = []

                # 只在內存中存儲文件內容，不立即寫入文件系統
                gcodes = {}

                for file in st.session_state.code_files:
                    # 讀取文件內容
                    content = file.read().decode("utf-8", errors="ignore")
                    gcodes[file.name.replace(".txt", "")] = content
                    gcodes = normalize_program_keys(gcodes)
                    file_names.append(file.name.replace(".txt", ""))

                # 將gcodes存入session state以便後續使用
                st.session_state.gcodes = gcodes

        else:  # code_source == "讀取機台代碼"
            # 讀取機台代碼功能
            # try:
            # 獲取所有夾位-機台選項
            all_options = conf_init.load_landing_product_options(finished=False)

            # 選擇夾位
            col11, col22, col33 = st.columns(3)
            with col11:
                # 使用session_state維護選擇的夾位
                department_options = sorted(
                    list(set([x.split("/")[0] for x in all_options]))
                )
                index = 0
                if st.session_state.selected_department in department_options:
                    index = sorted(list(department_options)).index(
                        st.session_state.selected_department
                    )

                selected_department = st.selectbox(
                    "選擇事業處",
                    department_options,
                    index=index,
                    key="machine_department_select",
                )
                st.session_state.selected_department = selected_department

            with col22:
                product_options = sorted(
                    list(
                        set(
                            [
                                x.split("/")[1].split("-CNC")[0]
                                for x in all_options
                                if x.split("/")[0]
                                == st.session_state.selected_department
                            ]
                        )
                    )
                )
                index = 0
                if st.session_state.simulation_productname in product_options:
                    index = sorted(list(product_options)).index(
                        st.session_state.simulation_productname
                    )

                st.session_state.simulation_productname = st.selectbox(
                    "選擇機種",
                    product_options,
                    index=index,
                    key="machine_product_select",
                )

            with col33:
                clamping_options = sorted(
                    list(
                        set(
                            [
                                x.replace(
                                    f"{st.session_state.selected_department}/{st.session_state.simulation_productname}-",
                                    "",
                                )
                                for x in all_options
                                if x.split("/")[0]
                                == st.session_state.selected_department
                                and x.split("/")[1].split("-CNC")[0]
                                == st.session_state.simulation_productname
                            ]
                        )
                    )
                )
                index = 0
                if st.session_state.simulation_clampingname in clamping_options:
                    index = sorted(list(clamping_options)).index(
                        st.session_state.simulation_clampingname
                    )
                st.session_state.simulation_clampingname = st.selectbox(
                    "選擇夾位",
                    clamping_options,
                    index=index,
                    key="machine_clamping_select",
                )

            st.session_state.selected_clamping_base = f"{st.session_state.simulation_productname}-{st.session_state.simulation_clampingname}"
            st.session_state.selected_clamping_base = (
                st.session_state.selected_clamping_base.replace("_", "-").replace(
                    " ", "-"
                )
            )

            if st.session_state.selected_clamping_base:
                selected_department = st.session_state.selected_department
                selected_clamping = st.session_state.selected_clamping_base

                # 載入機台數據
                if "machine_df" in st.session_state:
                    machine_df = st.session_state.machine_df.copy()
                    machine_df = machine_df[
                        machine_df["product_type"]
                        == "-".join(selected_clamping.split("-")[:-1])
                    ]
                    machine_df = machine_df[
                        machine_df["clamping"] == selected_clamping.split("-")[-1]
                    ].sort_values("date", ascending=False)

                    # 創建三列布局
                    col11, col22, col33 = st.columns(3)

                    with col11:
                        # 選擇樓層
                        floor_options = ["製工標準"] + list(
                            machine_df["floor"].unique()
                        )
                        selected_floor = st.selectbox(
                            "選擇樓層", floor_options, key="machine_floor_select"
                        )
                        if selected_floor != "製工標準":
                            machine_df = machine_df[
                                machine_df["floor"] == selected_floor
                            ]

                    with col22:
                        # 選擇機台
                        machine_options = ["製工標準"] + list(
                            machine_df["machine_id"].unique()
                        )
                        selected_machine = st.selectbox(
                            "選擇機台", machine_options, key="machine_id_select"
                        )
                        if selected_machine != "製工標準":
                            machine_df = machine_df[
                                machine_df["machine_id"] == selected_machine
                            ]

                    with col33:
                        # 選擇日期版本
                        date_version_options = ["製工標準"] + list(
                            machine_df["date"].unique()
                        )
                        selected_date_version = st.selectbox(
                            "選擇日期版本",
                            date_version_options,
                            key="date_version_select",
                        )

                    # 確定選擇的文件夾
                    if selected_date_version != "製工標準":
                        selected_folder = machine_df[
                            machine_df["date"] == selected_date_version
                        ].folder_name.to_list()[0]
                        baseline_display = f"{selected_floor}-{selected_machine}機台{selected_date_version}版本"
                    else:
                        selected_folder = "製工標準"
                        baseline_display = "製工標準"

                    st.success(
                        f"已選擇{st.session_state.selected_clamping_base}夾位的{baseline_display}代碼進行仿真"
                    )

                    # 讀取機台代碼
                    try:
                        # 更新配置參數
                        updated_config = st.session_state.simulation_config.copy()
                        updated_config["department"] = selected_department
                        updated_config["clamping_name"] = selected_clamping
                        updated_config["path"]["dir_app"] = updated_config["path"][
                            "dir_app"
                        ].format(department=selected_department)

                        if selected_folder != "製工標準":
                            # 設置機台文件夾路徑
                            machine_code_path = os.path.join(
                                "機台代碼",
                                selected_folder,
                                "programs.txt",  # 假設機台代碼是以JSON格式存儲的
                            )
                            updated_config["path"][
                                "dir_machine_folder"
                            ] = selected_folder
                            updated_config["path"]["dir_machine_data"] = os.path.join(
                                updated_config["path"]["dir_app"],
                                selected_clamping,
                                machine_code_path,
                            )
                        else:
                            updated_config["path"]["dir_machine_folder"] = "製工標準"

                        # 使用load_raw_gcodes函數讀取G碼文件
                        gcodes = load_raw_gcodes(updated_config)
                        file_names = list(gcodes.keys())

                        # 定義gcode目錄路徑
                        st.session_state.gcodes = gcodes

                    except Exception as e:
                        st.error(f"讀取G碼文件時出錯：{str(e)}")
                        file_names = []

                    # 仿真任務的名稱
                    st.session_state.simulation_clamping_name = st.text_input(
                        "仿真任務夾位名稱",
                        value=st.session_state.get("selected_clamping_base", ""),
                    )
                    st.session_state.simulation_clamping_name = (
                        st.session_state.simulation_clamping_name.replace(
                            "_", "-"
                        ).replace(" ", "-")
                    )
                    st.session_state.simulation_config["clamping_name"] = (
                        st.session_state.simulation_clamping_name
                    )

            # except Exception as e:
            #     st.error(f"讀取機台代碼時出錯：{str(e)}")
            #     file_names = []

            st.session_state.code_files = []  # 保持與原代碼兼容的變量名

    # 添加主程序選擇下拉菜單
    with col2:
        st.markdown(
            "##### 選擇主程序 <span style='color:yellow'>[必選]</span>",
            unsafe_allow_html=True,
        )
        st.markdown("請從上傳的文件中選擇一個作為主程序")
        file_names = [name for name in file_names if re.match(r"^O\d{4}$", name)]

        # 計算默認索引
        default_index = None
        if file_names:
            main_program = st.session_state.get("main_program")
            if main_program and main_program in file_names:
                default_index = file_names.index(main_program)
            else:
                default_index = 0

        st.session_state.main_program = st.selectbox(
            "主程序",
            options=file_names,
            index=default_index,
            format_func=lambda x: x,
            key=f"main_program_{st.session_state.simulation_clamping_name}",
        )

        if st.session_state.main_program is not None:
            st.success(f"已選擇 {st.session_state.main_program} 作為主程序")

    # 右側欄 - 上傳宏變量表格
    with col3:
        # 上傳宏變量表格
        st.markdown("##### 上傳宏變量表格 [可選]")
        st.markdown("請上傳包含宏變量的Excel表格文件（.xlsx或.csv格式）")

        macro_file = st.file_uploader(
            "宏變量表格", type=["xlsx", "csv"], key="macro_file"
        )

        if (
            macro_file is not None
            and "simulation_clamping_name" in st.session_state
            and st.session_state.simulation_clamping_name
        ):
            try:
                if macro_file.name.endswith(".csv"):
                    init_macro_df = pd.read_csv(macro_file, header=None)
                else:
                    init_macro_df = pd.read_excel(macro_file, header=None)

                # 檢查第一行是否為表頭（包含"宏變量"關鍵字）
                if len(init_macro_df) > 0 and "宏變量" in str(init_macro_df.iloc[0, 0]):
                    # 第一行是表頭，跳過第一行並設置列名
                    init_macro_df = init_macro_df.iloc[1:].reset_index(drop=True)
                    init_macro_df.columns = ["宏變量", "取值"]
                else:
                    # 第一行不是表頭，直接設置列名
                    init_macro_df.columns = ["宏變量", "取值"]

                st.success(f"成功上傳宏變量表格：{macro_file.name}")

                # 保存宏變量表格，但不立即創建目錄
                macro_dir = os.path.join(
                    st.session_state.simulation_config["path"]["dir_app"],
                    st.session_state.simulation_clamping_name,
                    st.session_state.simulation_config["path"]["machine_macro_path"],
                )
                # 將宏變量表格存儲在內存中，等待按鈕處理時再保存
                st.session_state.init_macro_df = init_macro_df
                st.session_state.macro_dir = macro_dir

            except Exception as e:
                st.error(f"無法讀取表格文件：{str(e)}")
        else:
            init_macro_df = pd.DataFrame()

    # 生成程式單等模版按鈕
    col1, col2 = st.columns(2)
    with col1:
        if st.button("返回CNC360 V1首頁", use_container_width=True):
            st.session_state.current_page = "landing"
            st.rerun()
    with col2:
        if st.button("生成程式單模版", use_container_width=True, type="primary"):

            # 如果使用的是"讀取機台代碼"模式，確保使用正確的夾位名稱
            if (
                st.session_state.code_source == "讀取機台代碼"
                and "created_clamping_name" in st.session_state
            ):
                st.session_state.simulation_clamping_name = (
                    st.session_state.created_clamping_name
                )

            # 現在使用最新的clamping_name設置clamping_dir
            clamping_dir = os.path.join(
                st.session_state.simulation_config["path"]["dir_app"],
                st.session_state.simulation_clamping_name,
            )

            # 如果夾位名稱為空，則顯示錯誤訊息
            if not len(st.session_state.simulation_clamping_name):
                st.error("請填寫夾位名稱")
            # 如果沒有上傳CNC代碼文件，則顯示錯誤訊息
            elif not len(file_names):
                st.error("請上傳CNC代碼文件")
            # 如果沒有選擇主程序，則顯示錯誤訊息
            elif "main_program" not in st.session_state:
                st.error("請選擇主程序")
            # 如果夾位已存在，則顯示警告訊息
            elif os.path.exists(clamping_dir):
                # 使用session_state來跟踪用戶的選擇
                st.session_state.simulation_exists_choice = None

                # 執行刪除操作
                def delete_clamping():
                    try:
                        # shutil.rmtree(simulation_dir)
                        subprocess.run(["rm", "-rf", clamping_dir], check=True)
                        st.session_state.simulation_exists_choice = "delete"
                        st.session_state.show_delete_confirmation = False
                        st.session_state.templates_generated = False
                        # st.write(f"已刪除舊的仿真夾位目錄：{clamping_dir}，現在您可以新建名為{st.session_state.clamping_name}的仿真了")
                    except Exception as e:
                        st.error(f"刪除目錄時出錯：{str(e)}")

                # 顯示確認刪除對話框
                st.markdown("### ⚠️ 確認刪除?")
                st.warning(
                    f"該夾位 '{st.session_state.simulation_clamping_name}' 的仿真已經存在！您要刪除它的仿真嗎？注意：此操作不可逆！"
                )
                confirm_container = st.container()
                with confirm_container:
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button(
                            "✓ 刪除",
                            key="confirm_delete_btn",
                            on_click=delete_clamping,
                            use_container_width=True,
                        ):
                            pass

                    with col_cancel:
                        if st.button(
                            "✗ 取消",
                            key="cancel_delete_btn",
                            use_container_width=True,
                        ):
                            st.session_state.simulation_exists_choice = "cancel"
                            st.session_state.show_delete_confirmation = False
                            st.rerun()

                # 在選項顯示後，如果用戶還沒有做出選擇，暫停執行
                if st.session_state.simulation_exists_choice != "delete":
                    st.stop()  # 停止執行，等待用戶選擇

            # 正常情況下，繼續執行
            else:
                pass

            # 定義 gcode_dir 變數
            gcode_dir = os.path.join(
                st.session_state.simulation_config["path"]["dir_app"],
                st.session_state.simulation_clamping_name,
                st.session_state.simulation_config["path"]["dir_gcode"],
            )

            # 現在將內存中的文件內容寫入文件系統
            os.makedirs(gcode_dir, exist_ok=True)
            gcodes = st.session_state.gcodes
            for file_name, content in gcodes.items():
                file_path = os.path.join(gcode_dir, file_name)
                if not file_path.endswith(".txt"):
                    file_path = file_path + ".txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # 獲取主程序文件路徑
            main_program_path = os.path.join(
                gcode_dir, f"{st.session_state.main_program}.txt"
            )

            # 創建模板目錄
            template_dir = os.path.join(
                st.session_state.simulation_config["path"]["dir_app"],
                st.session_state.simulation_clamping_name,
                st.session_state.simulation_config["path"]["dir_templates"],
            )
            os.makedirs(template_dir, exist_ok=True)

            # 確保配置中的夾位名稱是當前選擇的夾位
            st.session_state.simulation_config["clamping_name"] = (
                st.session_state.simulation_clamping_name
            )

            # 確保目錄存在
            os.makedirs(os.path.dirname(gcode_dir), exist_ok=True)
            os.makedirs(gcode_dir, exist_ok=True)

            # 調用三個函數生成模板
            with st.spinner("正在生成子程序模板..."):
                st.session_state.sub_program_df_origin, st.session_state.tool_mapper = (
                    generate_sub_program_master(main_program_path)
                )

                # 使用init_macro_df前檢查它是否存在於session_state中
                if "init_macro_df" in st.session_state:
                    macro_df_to_use = st.session_state.init_macro_df
                    # 如果已經保存了宏變量表格的路徑，現在創建目錄並保存文件
                    if "macro_dir" in st.session_state:
                        os.makedirs(
                            os.path.dirname(st.session_state.macro_dir), exist_ok=True
                        )
                        st.session_state.init_macro_df.to_excel(
                            st.session_state.macro_dir, index=False
                        )
                else:
                    macro_df_to_use = init_macro_df

                st.session_state.macros_df, st.session_state.tools_df = (
                    generate_macros_and_tools(
                        st.session_state.simulation_config,
                        macros=macro_df_to_use,
                        main_func=main_program_path,
                        funcs=gcode_dir,
                        tool_mapper=st.session_state.tool_mapper,
                    )
                )

                # 只存儲模板文件到template_dir，不存儲到最終路徑
                st.session_state.template_product_master_path = os.path.join(
                    template_dir, "product_master_template.xlsx"
                )
                st.session_state.sub_program_df_origin.to_excel(
                    st.session_state.template_product_master_path,
                    index=False,
                    header=False,
                )

            with st.spinner("正在生成刀具模板..."):
                st.session_state.template_tools_path = os.path.join(
                    template_dir, "tools_template.xlsx"
                )

                # 嘗試從刀具信息庫補充刀具信息
                # st.session_state.tools_df = query_tools_info(st.session_state.tools_df)

                # 更新模板文件
                st.session_state.tools_df.to_excel(
                    st.session_state.template_tools_path, index=False, header=False
                )

            with st.spinner("正在生成宏變量模板..."):
                st.session_state.template_macros_path = os.path.join(
                    template_dir, "macros_template.xlsx"
                )
                st.session_state.macros_df.to_excel(
                    st.session_state.template_macros_path, index=False, header=False
                )

            st.session_state.templates_generated = True

    # 確認是否有必需的數據可以進行驗證
    can_validate = (
        st.session_state.templates_generated
        and hasattr(st.session_state, "sub_program_df")
        and st.session_state.sub_program_df is not None
        and hasattr(st.session_state, "tools_df")
        and st.session_state.tools_df is not None
    )

    # 確保僅在以下情況下顯示模板預覽部分：
    # 1. 這是一個新夾位，不存在（simulation_exists_choice不存在於session_state）
    # 2. 這是一個舊夾位，但用戶點擊了刪除（simulation_exists_choice為"delete"）
    if st.session_state.templates_generated:
        st.markdown("---")

        # 顯示模板預覽和上傳下載
        st.markdown(
            "#### <span style='color:yellow'>第2步: 上傳程式單和刀具信息</span>",
            unsafe_allow_html=True,
        )

        # 從現有程式單更新模板
        # 初始化用戶選擇狀態
        if "user_template_choice" not in st.session_state:
            st.session_state.user_template_choice = "從代碼解析生成模板"

        # 計算當前選擇的索引
        choice_options = ["從代碼解析生成模板", "從程式單提取更新模板"]
        current_index = (
            choice_options.index(st.session_state.user_template_choice)
            if st.session_state.user_template_choice in choice_options
            else 0
        )

        update_template_choice = st.radio(
            "請選擇",
            choice_options,
            index=current_index,
            key="update_template_choice",
        )

        # 保存用戶選擇
        st.session_state.user_template_choice = update_template_choice
        if update_template_choice == "從程式單提取更新模板":
            st.session_state.existing_sub_program = st.file_uploader(
                "上傳現有程式單自動解析",
                type=["xlsx"],
                key="upload_existing_sub_program",
            )
            if st.session_state.existing_sub_program is not None:
                # if st.button("自動解析上傳的程式單", use_container_width=True):
                try:
                    existing_df = pd.read_excel(st.session_state.existing_sub_program)
                    st.session_state.sub_program_df = parse_sub_program_master(
                        existing_df, st.session_state.sub_program_df_origin.copy()
                    )
                    st.session_state.sub_program_df.to_excel(
                        st.session_state.template_product_master_path,
                        index=False,
                        header=False,
                    )
                except Exception as e:
                    st.error(f"無法讀取現有程式單：{str(e)}")
        else:
            st.session_state.sub_program_df = (
                st.session_state.sub_program_df_origin.copy()
            )
            st.session_state.sub_program_df["sub_program"] = (
                st.session_state.sub_program_df["sub_program"].astype(str).str.zfill(4)
            )
            st.session_state.sub_program_df.to_excel(
                st.session_state.template_product_master_path,
                index=False,
                header=False,
            )

        # 創建三列布局用於預覽、下載和上傳
        col1, col2, col3 = st.columns(3)

        # 子程序模板預覽、下載和上傳
        with col1:
            st.markdown(
                "##### 程式單 <span style='color:yellow'>[必選]</span>",
                unsafe_allow_html=True,
            )
            st.markdown("請根據提示補全程式單信息")

            # 修改預覽，只顯示前5行
            # preview_df = st.session_state.sub_program_df.copy()
            st.session_state.sub_program_df["sub_program"] = (
                st.session_state.sub_program_df["sub_program"].astype(str).str.zfill(4)
            )  # 確保子程序號碼格式正確
            st.dataframe(
                st.session_state.sub_program_df,
                use_container_width=True,
                hide_index=True,
            )

            # 設定最終路徑但不立即存儲
            st.session_state.product_master_path = os.path.join(
                st.session_state.simulation_config["path"]["dir_app"],
                st.session_state.simulation_clamping_name,
                st.session_state.simulation_config["path"]["master_path"],
            )
            # 去除標註is_supporting的子程序
            if "is_supporting" in st.session_state.sub_program_df.columns:
                # st.session_state.sub_program_df["is_supporting"] = (
                #     st.session_state.sub_program_df["is_supporting"].fillna(
                #         "子程式文件未找到"
                #     )
                # )
                # st.session_state.sub_program_df = st.session_state.sub_program_df[
                #     ~st.session_state.sub_program_df["is_supporting"]
                #     .astype(str)
                #     .isin(["輔助子程式", "1"])
                # ]
                sub_programs = st.session_state.sub_program_df["sub_program"].to_list()
                st.session_state.sub_program_df["sub_program_key"] = [
                    f"{str(idx+1).zfill(2)}-{str(sub_programs[idx]).zfill(4)}"
                    for idx in range(len(sub_programs))
                ]

            # 子程序模板下載按鈕
            with open(st.session_state.template_product_master_path, "rb") as file:
                st.download_button(
                    label="下載程式單模板",
                    data=file,
                    file_name="sub_program_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_sub_program",
                    use_container_width=True,
                )

            # 子程序模板上傳
            sub_program_upload = st.file_uploader(
                "上傳已填寫的程式單模板",
                type=["xlsx"],
                key="upload_sub_program",
            )

            if sub_program_upload is not None:
                try:
                    df = pd.read_excel(sub_program_upload)
                    if "sub_program" in df.columns:
                        pass  # 第一行作為 header
                    else:
                        df = pd.read_excel(
                            sub_program_upload, header=1
                        )  # 第二行作為 header

                    st.success(
                        f"成功上傳程式單模板{sub_program_upload.name}，將在驗證通過後保存"
                    )
                    df["sub_program"] = (
                        df["sub_program"].astype(str).str.zfill(4)
                    )  # 確保子程序號碼格式正確
                    templates["product_master"] = df
                    st.session_state.sub_program_df = df  # 更新預覽表格顯示的數據

                    if "is_supporting" in df.columns:
                        # df["is_supporting"] = df["is_supporting"].fillna(
                        #     "子程式文件未找到"
                        # )
                        # df = df[
                        #     ~df["is_supporting"].astype(str).isin(["輔助子程式", "1"])
                        # ]
                        sub_programs = df["sub_program"].to_list()
                        df["sub_program_key"] = [
                            f"{str(idx+1).zfill(2)}-{str(sub_programs[idx]).zfill(4)}"
                            for idx in range(len(sub_programs))
                        ]

                        def set_default_rot_center(string):
                            if "@" in str(string):
                                return str(string)
                            else:
                                return f"{string}@0/0/0"

                        df["rotation_4th_axis"] = df["rotation_4th_axis"].apply(
                            set_default_rot_center
                        )
                        df["rotation_0.5_axis"] = df["rotation_0.5_axis"].apply(
                            set_default_rot_center
                        )

                        st.session_state.sub_program_df = (
                            df  # 更新session_state中的數據
                        )

                except Exception as e:
                    st.error(f"無法讀取程式單模板：{str(e)}")

        # 刀具模板預覽、下載和上傳
        with col2:
            st.markdown(
                "##### 刀具信息 <span style='color:yellow'>[必選]</span>",
                unsafe_allow_html=True,
            )
            st.markdown("請根據提示填寫刀具信息")
            if st.session_state.tools_df is not None:
                st.dataframe(
                    st.session_state.tools_df,
                    use_container_width=True,
                    hide_index=True,
                )

                # 設定最終路徑但不立即存儲
                st.session_state.tools_path = os.path.join(
                    st.session_state.simulation_config["path"]["dir_app"],
                    st.session_state.simulation_clamping_name,
                    st.session_state.simulation_config["path"]["tool_path"],
                )

                # 刀具模板下載按鈕（從模板文件下載）
                with open(st.session_state.template_tools_path, "rb") as file:
                    st.download_button(
                        label="下載刀具模板",
                        data=file,
                        file_name="tools_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_tools",
                        use_container_width=True,
                    )

                # 刀具模板上傳
                tools_upload = st.file_uploader(
                    "上傳已填寫的刀具模板", type=["xlsx"], key="upload_tools"
                )

                if tools_upload is not None:
                    try:
                        df = pd.read_excel(tools_upload)
                        if "刀號" in df.columns:
                            pass  # 第一行作為 header
                        else:
                            df = pd.read_excel(
                                tools_upload, header=1
                            )  # 第二行作為 header

                        st.success(
                            f"成功上傳刀具模板{tools_upload.name}，將在驗證通過後保存"
                        )
                        templates["tools"] = df
                        st.session_state.tools_df = df
                        # st.session_state.tools_df = query_tools_info(st.session_state.tools_df)

                    except Exception as e:
                        st.error(f"無法讀取刀具模板：{str(e)}")
            else:
                st.info("未生成刀具模板")

        # 宏變量模板預覽、下載和上傳
        with col3:
            st.markdown("##### 宏變量[可選]")
            st.markdown("請填寫缺失值的宏變量取值")
            if st.session_state.macros_df is not None:
                st.dataframe(
                    st.session_state.macros_df,
                    use_container_width=True,
                    hide_index=True,
                )
                # 設定最終路徑但不立即存儲
                st.session_state.macros_path = os.path.join(
                    st.session_state.simulation_config["path"]["dir_app"],
                    st.session_state.simulation_clamping_name,
                    st.session_state.simulation_config["path"]["machine_macro_path"],
                )

                # 宏變量模板下載按鈕（從模板文件下載）
                with open(st.session_state.template_macros_path, "rb") as file:
                    st.download_button(
                        label="下載宏變量模板",
                        data=file,
                        file_name="macros_template.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_macros",
                        use_container_width=True,
                    )
            else:
                st.info("未生成宏變量模板")

            # 宏變量模板上傳
            macros_upload = st.file_uploader(
                "上傳已填寫的宏變量模板", type=["xlsx", "csv"], key="upload_macros"
            )

            if macros_upload is not None:
                try:
                    df = pd.read_excel(macros_upload)
                    if "macro" in df.columns:
                        pass  # 第一行作為 header
                    else:
                        df = pd.read_excel(macros_upload, header=1)  # 第二行作為 header

                    st.success(
                        f"成功上傳宏變量模板{macros_upload.name}，將在驗證通過後保存"
                    )
                    st.session_state.macros_df = df
                    templates["macros"] = df
                except Exception as e:
                    st.error(f"無法讀取宏變量模板：{str(e)}")

    if st.session_state.templates_generated and can_validate:

        # 初始化驗證狀態
        if "validation_completed" not in st.session_state:
            st.session_state.validation_completed = False
        if "validation_results" not in st.session_state:
            st.session_state.validation_results = {}

        # 創建兩列布局用於刷新和驗證按鈕
        col_refresh, col_validate = st.columns(2)

        with col_refresh:
            if st.button(
                "刷新預覽表格", use_container_width=True, key="refresh_preview"
            ):
                st.rerun()

        with col_validate:
            validate_clicked = st.button(
                "驗證表格",
                use_container_width=True,
                key="simulation_validate_tables",
                type="primary",
            )

        if validate_clicked:

            col1, col2, col3 = st.columns(3)
            with col1:
                valid_sub_program, errors, warnings = validate_upload_sub_program_df(
                    st.session_state.sub_program_df
                )
                if valid_sub_program:
                    os.makedirs(
                        os.path.dirname(st.session_state.product_master_path),
                        exist_ok=True,
                    )
                    st.session_state.sub_program_df.to_excel(
                        st.session_state.product_master_path, index=False
                    )
                    st.success(
                        f"程式單表格驗證通過，並保存到{st.session_state.product_master_path}"
                    )
                else:
                    st.error("程式單表格驗證失敗")
                with st.expander("驗證細節"):
                    for each in errors:
                        st.error(each)
                    for each in warnings:
                        st.warning(each)
            with col2:
                valid_tools, errors, warnings = validate_upload_tools_df(
                    st.session_state.tools_df
                )
                if valid_tools:
                    os.makedirs(
                        os.path.dirname(st.session_state.tools_path), exist_ok=True
                    )
                    st.session_state.tools_df.to_excel(
                        st.session_state.tools_path, index=False
                    )
                    st.success(
                        f"刀具表格驗證通過，並保存到{st.session_state.tools_path}"
                    )
                else:
                    st.error("刀具表格驗證失敗")
                with st.expander("驗證細節"):
                    for each in errors:
                        st.error(each)
                    for each in warnings:
                        st.warning(each)
            with col3:
                valid_macro, errors, warnings = validate_upload_macro_df(
                    st.session_state.macros_df
                )
                if valid_macro:
                    os.makedirs(
                        os.path.dirname(st.session_state.macros_path), exist_ok=True
                    )
                    st.session_state.macros_df.to_excel(
                        st.session_state.macros_path, index=False
                    )
                    st.success(
                        f"宏變量表格驗證通過，並保存到{st.session_state.macros_path}"
                    )
                else:
                    st.error("宏變量表格驗證失敗")
                with st.expander("驗證細節"):
                    for each in errors:
                        st.error(each)
                    for each in warnings:
                        st.warning(each)

            # 保存驗證結果到session_state
            st.session_state.validation_results = {
                "valid_sub_program": valid_sub_program,
                "valid_tools": valid_tools,
                "valid_macro": valid_macro,
            }
            st.session_state.validation_completed = True

        # 顯示進入仿真按鈕（不嵌套在驗證按鈕內）
        if (
            st.session_state.validation_completed
            and st.session_state.validation_results.get("valid_sub_program", False)
            and st.session_state.validation_results.get("valid_tools", False)
            and st.session_state.validation_results.get("valid_macro", False)
        ):

            if st.button(
                "下一步 - 上傳毛坯",
                use_container_width=True,
                key="simulation_button",
                type="primary",
            ):

                # 在跳轉之前確保所有相關變量都被正確設定
                st.session_state["simulation_clamping"] = (
                    st.session_state.simulation_clamping_name
                )
                # 確保simulation_config也使用正確的夾位名稱
                st.session_state.simulation_config["clamping_name"] = (
                    st.session_state.simulation_clamping_name
                )
                print(
                    f"跳轉前診斷: clamping_name={st.session_state.simulation_clamping_name}, simulation_clamping={st.session_state.simulation_clamping}"
                )
                st.session_state.current_page = "simulation"
                st.rerun()

        if st.button(
            "返回CNC360 V1首頁",
            use_container_width=True,
            help="return_homepage_create_simulation",
        ):
            st.session_state.current_page = "landing"
            st.rerun()


def validate_upload_macro_df(macro_df):

    if "宏變量" in macro_df["宏變量"].to_list():
        macro_df = macro_df.iloc[2:]

    res = True
    errors = []
    warnings = []
    df_NaN = macro_df[macro_df["取值"].isna()]
    for idx, row in df_NaN.iterrows():
        warnings.append(
            f"宏變量{row['宏變量']}未定義，可能造成仿真失敗或錯誤，請盡量提供..."
        )
    return res, errors, warnings


def validate_upload_sub_program_df(sub_program_df):

    if "sub_program" in sub_program_df["sub_program"].to_list():
        sub_program_df = sub_program_df.iloc[2:]

    # 檢查sub_program_df的sub_program_key是否唯一，找出重複的子程式號
    errors = []
    warnings = []
    df_duplicated = sub_program_df[sub_program_df["sub_program_key"].duplicated()]
    res = True
    for each in df_duplicated["sub_program"]:
        warnings.append(
            f"子程式序號{each}重複，請確認子程式序號是否唯一，系統只會對第一次執行子程式進行仿真"
        )
    # 檢查sub_program_df的sub_program是否為四位數字
    for each in sub_program_df["sub_program"]:
        if not (str(each).isdigit() and len(str(each)) == 4):
            errors.append(f"子程式序號必須為四位數字，{each}不符合規則，請修改")
            res = False

    # 檢查sub_program是否都存在於gcodes中
    if hasattr(st.session_state, "gcodes") and st.session_state.gcodes:
        for each in sub_program_df["sub_program"]:
            # 將子程序號轉換為標準格式 O####
            sub_program = f"O{str(each).zfill(4)}"
            if sub_program not in st.session_state.gcodes.keys():
                errors.append(
                    f"子程式 {each} 對應的代碼文件 {sub_program} 未找到，請確認已上傳相應的代碼文件，或檢查程式單是否匹配"
                )
                res = False
    else:
        errors.append("無法獲取已上傳的代碼文件信息，請重新上傳代碼文件")
        res = False
    return res, errors, warnings


def validate_upload_tools_df(tools_df, hint=True):
    if "刀號" in tools_df["刀號"].to_list():
        tools_df = tools_df.iloc[2:]

    # 檢查tools_df的刀號是否唯一，找出重複的刀號
    df_duplicated = tools_df[tools_df["刀號"].duplicated()]
    res = True
    errors = []
    warnings = []
    for each in df_duplicated["刀號"]:
        warnings.append(
            f"刀號{each}重複，請確認刀號是否唯一，系統將會默認使用第一次出現的刀具"
        )
    # 檢查刀具的diameter和height是否為大於0的數字
    for idx, row in tools_df.iterrows():
        # 檢查刀頭直徑
        try:
            diameter_value = float(row["刀頭直徑"])
            if diameter_value <= 0:
                errors.append(
                    f"刀具{row['刀號']}的直徑{row['刀頭直徑']}必須為大於0的數字，請檢查"
                )
                res = False
            if diameter_value > 30:
                warnings.append(
                    f"刀具{row['刀號']}的直徑{row['刀頭直徑']}大於30mm，請檢查"
                )
        except (ValueError, TypeError):
            errors.append(f"刀具{row['刀號']}的直徑需為數字，請檢查")
            res = False

        # 檢查刀頭高度
        try:
            height_value = float(row["刀頭高度"])
            if height_value <= 0:
                errors.append(
                    f"刀具{row['刀號']}的刃長{row['刀頭高度']}必須為大於0的數字，請檢查"
                )
                res = False
            if height_value > 50:
                warnings.append(
                    f"刀具{row['刀號']}的刃長{row['刀頭高度']}大於50mm，請檢查"
                )
        except (ValueError, TypeError):
            errors.append(f"刀具{row['刀號']}的刃長需為數字，請檢查")
            res = False
    return res, errors, warnings


def query_tools_info(tools_df, tools_info_bank="../cnc_data/tools/all_tools.xlsx"):
    """
    根據刀具的規格型號查詢刀具信息庫，補充或更新刀具信息

    Args:
        tools_df (pd.DataFrame): 刀具資料框，包含刀號、規格型號等信息
        tools_info_bank (str): 刀具信息庫文件路徑

    Returns:
        pd.DataFrame: 更新後的刀具資料框
    """
    import os

    try:
        # 檢查刀具信息庫文件是否存在
        if not os.path.exists(tools_info_bank):
            st.warning(f"刀具信息庫文件不存在：{tools_info_bank}")
            return tools_df

        # 讀取刀具信息庫
        try:
            if tools_info_bank.endswith(".xlsx"):
                tools_bank_df = pd.read_excel(tools_info_bank)
            elif tools_info_bank.endswith(".csv"):
                tools_bank_df = pd.read_csv(tools_info_bank)
            else:
                st.warning(f"不支持的文件格式：{tools_info_bank}")
                return tools_df
        except Exception as e:
            st.warning(f"讀取刀具信息庫文件失敗：{str(e)}")
            return tools_df

        # 檢查必要的列是否存在
        if "規格型號" not in tools_bank_df.columns:
            st.warning("刀具信息庫中缺少'規格型號'列，無法進行匹配")
            return tools_df

        if "規格型號" not in tools_df.columns:
            st.warning("刀具表中缺少'規格型號'列，無法進行匹配")
            return tools_df

        # 使用 merge 進行匹配，保留原始數據的優先級
        merged_df = tools_df.merge(
            tools_bank_df, on="規格型號", how="left", suffixes=("", "_bank")
        )

        # 統計匹配成功的數量
        match_count = (
            merged_df[merged_df.columns[merged_df.columns.str.endswith("_bank")]]
            .notna()
            .any(axis=1)
            .sum()
        )

        # 用信息庫的數據填充空值（只在原數據為空時更新）
        for col in tools_bank_df.columns:
            if col != "規格型號" and col in tools_df.columns:
                bank_col = f"{col}_bank"
                if bank_col in merged_df.columns:
                    # 只在原數據為空或0時用信息庫數據替換
                    mask = (
                        (merged_df[col].isna())
                        | (merged_df[col] == "")
                        | (merged_df[col] == 0)
                    )
                    merged_df.loc[mask, col] = merged_df.loc[mask, bank_col]

        # 刪除帶 _bank 後綴的列
        result_df = merged_df[tools_df.columns]

        if match_count > 0:
            st.success(f"成功從刀具信息庫匹配了 {match_count} 個刀具的信息")
        else:
            st.info("未找到匹配的刀具信息，請檢查規格型號是否一致")

        return result_df

    except Exception as e:
        st.error(f"查詢刀具信息庫時發生錯誤：{str(e)}")
        return tools_df
