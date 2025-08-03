# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import zipfile
import json
import tempfile
from pathlib import Path
import pandas as pd
import subprocess
from datetime import datetime, timedelta
import shutil
from cnc_genai.demo_ui import conf_init


def render_upload_page():
    """渲染代碼上傳頁面"""
    st.markdown(
        "<h1 style='text-align: center; color: #FFFFFF; margin-bottom: 30px;'>生技代碼上傳</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 選擇代碼文件...")
        uploaded_files = st.file_uploader(
            "選擇要上傳的代碼文件",
            accept_multiple_files=True,
            type=["nc", "txt", "zip", "rar"],
            help="支持.nc, .txt, .zip, .rar文件格式",
        )

        # 顯示選擇的文件
        if uploaded_files:
            st.info(f"已選擇 {len(uploaded_files)} 個文件")

    with col2:
        st.subheader("🔧 上傳到...")

        # 讀取parquet文件數據
        data_list = pd.read_parquet("../cnc_data/input/equCramp.parquet")["data"][0]
        # 處理帶有轉義雙引號的JSON字符串
        parsed_data = []
        for item in data_list:
            if isinstance(item, str):
                # 替換轉義的雙引號
                clean_json = item.replace('""', '"')
                parsed_data.append(json.loads(clean_json))
            else:
                parsed_data.append(item)

        machine_df = pd.DataFrame(parsed_data)
        machine_df["equaddress"] = machine_df["equaddress"] + "F"

        # clamping_options = conf_init.load_landing_product_options()
        # selected_clamping = st.selectbox(
        #     "選擇夾位", clamping_options, key="clamping_select"
        # )

        col21, col22, col23 = st.columns(3)
        with col21:
            selected_clamping_dept = st.selectbox(
                "選擇事業部", ["mac1", "mac3"], key="clamping_select_dept"
            )
        with col22:
            # 獲取所有機種選項
            all_clamping_options = conf_init.load_landing_product_options()
            product_options = sorted(
                list(
                    set(
                        [
                            x.split("/")[1].split("-CNC")[0]
                            for x in all_clamping_options
                            if x.startswith(selected_clamping_dept)
                        ]
                    )
                )
            ) + ["其他機種..."]

            # 選擇或輸入機種
            selected_temp = st.selectbox(
                "選擇/輸入機種",
                options=product_options,
                index=None,
                placeholder="請選擇機種...",
                key="product_select",
            )

            # 處理自定義輸入
            if selected_temp == "其他機種...":
                custom_product = st.text_input(
                    "輸入新機種",
                    value="",
                    placeholder="輸入新機種名稱",
                    key="custom_product_input",
                )
                selected_productname = custom_product
            else:
                selected_productname = selected_temp

        with col23:
            # 預設夾位選項
            clamping_options = [f"CNC{i}" for i in range(1, 8)] + ["其他夾位..."]

            # 選擇或輸入機種
            selected_temp = st.selectbox(
                "選擇/輸入夾位",
                options=clamping_options,
                index=None,
                placeholder="請選擇夾位...",
                key="clamping_select",
            )

            # 處理自定義輸入
            if selected_temp == "其他夾位...":
                custom_clamping = st.text_input(
                    "輸入新夾位",
                    value="",
                    placeholder="輸入新夾位名稱",
                    key="custom_clamping_input",
                )
                selected_clamping_name = custom_clamping
            else:
                selected_clamping_name = selected_temp

    if selected_productname and selected_clamping_name:
        with col2:
            selected_clamping = (
                selected_clamping_dept
                + "/"
                + selected_productname
                + "-"
                + selected_clamping_name
            )
            st.success(f"已選擇夾位{selected_clamping}")

            col22, col33, col44, col55 = st.columns(4)
            with col22:
                floor_options = list(machine_df["equaddress"].unique())
                selected_floor = st.selectbox("選擇樓層", floor_options)
                machine_df = machine_df[machine_df["equaddress"] == selected_floor]
            with col33:
                line_options = list(machine_df["equcode"].str[0].unique())
                selected_line = st.selectbox("選擇線", sorted(line_options))
                machine_df = machine_df[
                    machine_df["equcode"].str.startswith(selected_line)
                ]
            with col44:
                machine_options = list(machine_df["equcode"].unique())
                selected_machine = st.selectbox("選擇機台", machine_options)
                machine_df = machine_df[machine_df["equcode"] == selected_machine]
            with col55:
                selected_date = st.date_input("選擇日期", value=datetime.now().date())

            # 加一個自由輸入的文本作為後綴，默認"生技上傳"
            input_suffix = st.text_input(
                "請輸入版本後綴，避免版本衝突...",
                value=st.session_state.get("username", "Unknown User"),
                placeholder="請輸入版本後綴，避免版本衝突...",
            )

            # 檢查輸入是否為ASCII字符
            def is_ascii_only(text):
                try:
                    text.encode("ascii")
                    return True
                except UnicodeEncodeError:
                    return False

            # 驗證輸入後綴是否為ASCII
            if input_suffix:
                if not is_ascii_only(input_suffix):
                    st.error(
                        "❌ 版本後綴只能包含ASCII字符（英文字母、數字、符號），請重新輸入！"
                    )
                    st.info("💡 建議使用英文字母、數字或符號作為版本後綴")
                    input_suffix = ""  # 清空非法輸入

            # 檢查路徑是否存在並提示覆蓋
            new_string = re.sub(
                r"^([^-]+-[^-]+)-", r"\1#", selected_clamping.split("/")[1]
            )
            target_path = f"../cnc_data/nc_code/{selected_floor}#{new_string}#{selected_machine}#{selected_date}{input_suffix}"

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            # 返回按鈕
            if st.button("返回CNC360 V1首頁", use_container_width=True):
                st.session_state.current_page = "landing"
                st.rerun()

        with col2:
            # 上傳按鈕
            if input_suffix:
                if st.button("開始上傳", use_container_width=True, type="primary"):

                    # 判断target_path是否存在，如果存在请用户二次确认是否删除
                    if os.path.exists(target_path):
                        # 使用session_state來跟踪用戶的選擇
                        st.session_state.upload_exists_choice = None

                        # 執行刪除操作
                        def delete_target_path():
                            try:
                                import subprocess

                                subprocess.run(["rm", "-rf", target_path], check=True)
                                st.session_state.upload_exists_choice = "delete"
                                st.session_state.show_delete_confirmation = False
                            except Exception as e:
                                st.error(f"刪除目錄時出錯：{str(e)}")

                        # 顯示確認刪除對話框
                        st.markdown("### ⚠️ 確認刪除?")
                        st.warning(
                            f"目標路徑 '{target_path}' 已經存在！您要刪除它嗎？注意：此操作不可逆！"
                        )
                        confirm_container = st.container()
                        with confirm_container:
                            col_confirm, col_cancel = st.columns(2)
                            with col_confirm:
                                if st.button(
                                    "✓ 刪除",
                                    key="upload_confirm_delete_btn",
                                    on_click=delete_target_path,
                                    use_container_width=True,
                                ):
                                    pass

                            with col_cancel:
                                if st.button(
                                    "✗ 取消",
                                    key="upload_cancel_delete_btn",
                                    use_container_width=True,
                                ):
                                    st.session_state.upload_exists_choice = "cancel"
                                    st.session_state.show_delete_confirmation = False
                                    st.rerun()

                        # 在選項顯示後，如果用戶還沒有做出選擇，暫停執行
                        if st.session_state.upload_exists_choice != "delete":
                            st.stop()  # 停止執行，等待用戶選擇

                    # 創建cache目錄，如果已存在則先清空
                    cache_dir = f"../cnc_data/cache_uploaded_nc/{selected_floor}#{new_string}#{selected_machine}#{selected_date}{input_suffix}"
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)  # 徹底刪除目錄及其內容
                    os.makedirs(cache_dir, exist_ok=True)  # 重新創建乾淨的目錄

                    # 保存上傳文件到cache目錄
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(cache_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # 執行轉換命令
                    subprocess.run(
                        [
                            "python",
                            "cnc_genai/src/utils/convert_txt_to_json.py",
                            "-i",
                            cache_dir,
                            "-o",
                            target_path,
                        ]
                    )
                    st.success(f"NC代碼已成功上傳並轉換，存儲路徑: {target_path}")

                    # 删除cache_dir
                    try:
                        shutil.rmtree(cache_dir)
                    except Exception as e:
                        st.warning(f"清理緩存目錄時出現問題：{str(e)}")

                    st.session_state.machine_df = (
                        conf_init.parse_folder_names_to_dataframe(
                            f"../cnc_data/nc_code/"
                        )
                    )

    else:
        with col2:
            st.error("請選擇機種和夾位")
