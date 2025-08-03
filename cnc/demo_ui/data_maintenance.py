# -*- coding: utf-8 -*-
import streamlit as st
import yaml
import os
import pandas as pd
from pathlib import Path
from cnc_v0.model.agent.langchain.src.utils.auth import verify_password


def load_user_credentials(config_path=None):
    """從YAML文件中加載用戶憑據"""
    try:
        # 如果没有指定路径，使用data_path.yaml中的dir作为前缀
        if config_path is None:
            data_path = get_data_path()
            config_path = os.path.join(data_path, "cnc_cred/auth_users.yaml")
        
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        credentials = {}
        acc_data = data.get("account_name", {})
        for k in acc_data:
            username = k
            password = acc_data[k].get("password", None)
            access_level = acc_data[k].get("access_level", [])
            data_admin = acc_data[k].get("data_admin", [])
            credentials[username] = [password, access_level, data_admin]
        return credentials
    except Exception as e:
        st.error(f"加載用戶憑據時出錯: {str(e)}")
        return {}


def render_data_maintenance_login():
    """渲染數據維護登入頁面"""
    # 重置數據維護相關的session狀態
    if "data_maintenance_logged_in" in st.session_state:
        del st.session_state.data_maintenance_logged_in
    if "data_maintenance_username" in st.session_state:
        del st.session_state.data_maintenance_username
    if "data_maintenance_data_admin" in st.session_state:
        del st.session_state.data_maintenance_data_admin
    
    # 重置代碼管理相關的session狀態
    # 删除所有复杂的session state管理
    
    st.markdown(
        """
        <style>
        div[data-testid="stForm"] {
            background-color: transparent;
            border: none;
            padding: 0;
        }
        div[data-testid="stForm"] > div:first-child {
            background-color: transparent;
            border: none;
            padding: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='max-width: 1000px; margin: 0 auto;'>
            <div style='text-align: center; padding: 20px; background-color: rgba(255,0,0,0.1); border-radius: 10px; margin-bottom: 30px;'>
                <h2>數據維護管理系統</h2>
                <h3>請輸入您的用戶名和密碼</h3>
                <p>僅限mac1和mac3用戶訪問</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 創建一個居中的容器
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 創建登入表單
        with st.form("data_maintenance_login_form"):
            username = st.text_input("用戶名")
            password = st.text_input("密碼", type="password")
            submit_button = st.form_submit_button("登入", use_container_width=True)

            # 從YAML文件加載用戶憑據
            valid_users = load_user_credentials()

            if valid_users:
                if submit_button:
                    if username not in valid_users:
                        st.error("用戶名錯誤，請重試")
                        return
                    else:
                        try:
                            stored_password = valid_users[username][0]
                            if password == stored_password:  # 直接比較密碼
                                # 檢查是否有data_admin權限
                                data_admin = valid_users[username][2]
                                if data_admin and (isinstance(data_admin, list) and ("mac1" in data_admin or "mac3" in data_admin)):
                                    st.session_state.data_maintenance_logged_in = True
                                    st.session_state.data_maintenance_username = username
                                    st.session_state.data_maintenance_data_admin = data_admin
                                    st.session_state.current_page = "data_maintenance"
                                    st.rerun()
                                else:
                                    st.error("您沒有數據維護權限，僅限mac1和mac3用戶訪問")
                            else:
                                st.error("密碼錯誤，請重試")
                        except Exception as e:
                            st.error(f"其他錯誤: {e}")

    # 添加返回按鈕
    if st.button("返回主頁", use_container_width=True):
        st.session_state.current_page = "homepage"
        st.rerun()


def get_data_path():
    """獲取數據路徑配置"""
    try:
        # 嘗試多個可能的路徑（基于Docker环境）
        possible_paths = [
            "cnc_genai/conf/data_path.yaml",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r") as file:
                    data = yaml.safe_load(file)
                return data.get("dir", "../cnc_data/")
        
        # 如果都找不到，返回默認路徑
        st.warning("無法找到data_path.yaml配置文件，使用默認路徑")
        return "../cnc_data/"
    except Exception as e:
        st.error(f"加載數據路徑配置時出錯: {str(e)}")
        return "../cnc_data/"


def list_folder_contents(folder_path):
    """列出文件夾內容"""
    try:
        if not os.path.exists(folder_path):
            return []
        
        contents = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                # 獲取文件大小
                file_size = os.path.getsize(item_path)
                file_size_str = format_file_size(file_size)
                contents.append({
                    "name": item, 
                    "type": "file", 
                    "path": item_path,
                    "size": file_size_str,
                    "size_bytes": file_size
                })
            elif os.path.isdir(item_path):
                contents.append({"name": item, "type": "directory", "path": item_path})
        return contents
    except Exception as e:
        st.error(f"讀取文件夾內容時出錯: {str(e)}")
        return []


def format_file_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"


def render_data_maintenance_page():
    """渲染數據維護頁面"""
    # 檢查登入狀態
    if not st.session_state.get("data_maintenance_logged_in", False):
        st.error("請先登入數據維護系統")
        if st.button("返回登入頁面"):
            st.session_state.current_page = "data_maintenance_login"
            st.rerun()
        return

    # 檢查權限
    data_admin = st.session_state.get("data_maintenance_data_admin", [])
    if not data_admin or not any(admin in data_admin for admin in ["mac1", "mac3"]):
        st.error("您沒有數據維護權限")
        if st.button("返回主頁"):
            st.session_state.current_page = "landing"
            st.rerun()
        return

    st.markdown(
        "<h1 style='text-align: center; color: #FFFFFF; margin-bottom: 30px;'>數據維護管理系統</h1>",
        unsafe_allow_html=True,
    )

    # 創建兩列佈局
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📁 數據管理")
        
        # 獲取數據路徑
        data_path = get_data_path()
        
        # 根據用戶權限顯示對應的文件夾
        available_folders = []
        for admin in data_admin:
            if admin in ["mac1", "mac3"]:
                folder_path = os.path.join(data_path, "cnc_master_manual", admin)
                if os.path.exists(folder_path):
                    available_folders.append({"name": admin, "path": folder_path})
        
        if available_folders:
            # 第一部分：選擇數據類型
            # 直接使用第一個可用的文件夾（用戶權限對應的文件夾）
            selected_folder_info = available_folders[0]
            selected_folder = selected_folder_info["name"]
            folder_contents = list_folder_contents(selected_folder_info["path"])
            
            # 獲取子文件夾列表
            sub_folders = []
            for item in folder_contents:
                if item["type"] == "directory":
                    sub_folders.append(item)
            
            if sub_folders:
                # 初始化session state用于多级选择
                if "data_management_path_stack" not in st.session_state:
                    st.session_state.data_management_path_stack = []
                if "data_management_current_path" not in st.session_state:
                    st.session_state.data_management_current_path = selected_folder_info["path"]
                
                # 递归函数：处理多级文件夹选择
                def handle_folder_selection(current_path, level=0):
                    folder_contents = list_folder_contents(current_path)
                    folders = [item for item in folder_contents if item["type"] == "directory"]
                    files = [item for item in folder_contents if item["type"] == "file"]
                    
                    # 显示当前路径
                    if level > 0:
                        st.info(f"📁 当前路径: {os.path.relpath(current_path, selected_folder_info['path'])}")
                    
                    # 如果有文件夹，显示文件夹选择
                    if folders:
                        folder_options = [f["name"] for f in folders]
                        selected_folder_name = st.selectbox(
                            f"選擇第{level+1}級數據類型",
                            options=folder_options,
                            format_func=lambda x: f"📁 {x}",
                            key=f"folder_select_{level}"
                        )
                        
                        # 获取选中的文件夹路径
                        current_selected_folder_info = next(f for f in folders if f["name"] == selected_folder_name)
                        selected_folder_path = current_selected_folder_info["path"]
                        
                        # 递归处理下一级
                        return handle_folder_selection(selected_folder_path, level + 1)
                    
                    # 如果没有文件夹，显示文件选择
                    elif files:
                        file_options = [f["name"] for f in files]
                        selected_file = st.selectbox(
                            "選擇數據項",
                            options=file_options,
                            format_func=lambda x: f"📄 {x}",
                            key=f"file_select_{level}"
                        )
                        
                        # 下载和上传按钮
                        col_download, col_upload = st.columns(2)
                        
                        with col_download:
                            st.markdown("**下载文件**")
                            if selected_file:
                                file_path = os.path.join(current_path, selected_file)
                                if os.path.exists(file_path):
                                    with open(file_path, "rb") as f:
                                        file_data = f.read()
                                    st.download_button(
                                        label="📥 下載文件",
                                        data=file_data,
                                        file_name=selected_file,
                                        mime="application/octet-stream",
                                        key=f"download_btn_{level}",
                                        use_container_width=True
                                    )
                        
                        with col_upload:
                            st.markdown("**上传文件**")
                            uploaded_files = st.file_uploader(
                                f"上傳文件到當前文件夾",
                                type=None,  # 接受所有文件类型，包括无扩展名文件
                                key=f"upload_{level}",
                                label_visibility="collapsed",
                                accept_multiple_files=True
                            )
                            
                            if uploaded_files:
                                try:
                                    saved_files = []
                                    for uploaded_file in uploaded_files:
                                        # 保存上傳的文件
                                        file_path = os.path.join(current_path, uploaded_file.name)
                                        with open(file_path, "wb") as f:
                                            f.write(uploaded_file.getbuffer())
                                        saved_files.append(uploaded_file.name)
                                    
                                    # 顯示上傳成功信息
                                    st.success(f"✅ 已成功上傳 {len(saved_files)} 個文件到當前文件夾")
                                    st.success(f"📄 文件列表: {', '.join(saved_files)}")
                                except Exception as e:
                                    st.error(f"❌ 上傳失敗: {str(e)}")
                        
                        return current_path
                    
                    else:
                        st.info("當前文件夾沒有文件")
                        
                        # 只有上傳功能
                        st.markdown("**上传文件**")
                        uploaded_files = st.file_uploader(
                            "上傳文件到當前文件夾",
                            type=None,  # 接受所有文件类型，包括无扩展名文件
                            key=f"upload_empty_{level}",
                            label_visibility="collapsed",
                            accept_multiple_files=True
                        )
                        
                        if uploaded_files:
                            try:
                                saved_files = []
                                for uploaded_file in uploaded_files:
                                    # 保存上傳的文件
                                    file_path = os.path.join(current_path, uploaded_file.name)
                                    with open(file_path, "wb") as f:
                                        f.write(uploaded_file.getbuffer())
                                    saved_files.append(uploaded_file.name)
                                
                                # 顯示上傳成功信息
                                st.success(f"✅ 已成功上傳 {len(saved_files)} 個文件到當前文件夾")
                                st.success(f"📄 文件列表: {', '.join(saved_files)}")
                            except Exception as e:
                                st.error(f"❌ 上傳失敗: {str(e)}")
                        
                        return current_path
                
                # 开始递归处理
                final_path = handle_folder_selection(selected_folder_info["path"])
            else:
                st.info(f"{selected_folder} 文件夾沒有子文件夾")
        else:
            st.warning("沒有可用的文件夾")

    with col2:
        st.markdown("### 💻 代碼管理")
        
        # 獲取數據路徑
        data_path = get_data_path()

        # 輸入機種和夾位信息
        col_machine, col_clamping, col_confirm = st.columns([1, 1, 1])
        with col_machine:
            machine_type = st.text_input("機種", key="machine_type_input")
        with col_clamping:
            clamping = st.text_input("夾位", key="clamping_input")
        with col_confirm:
            confirm_clicked = st.button("確認", use_container_width=True)
        
        # 如果點擊了確認按鈕，設置狀態
        if confirm_clicked and machine_type and clamping:
            st.session_state.code_confirmed = True
            st.session_state.machine_type = machine_type
            st.session_state.clamping = clamping
            st.rerun()
        
        # 如果已確認，顯示文件夾檢查和上傳區域
        if st.session_state.get("code_confirmed", False):
            machine_type = st.session_state.get("machine_type", "")
            clamping = st.session_state.get("clamping", "")
            
            if machine_type and clamping:
                # 構建目標文件夾路徑
                target_folder_name = f"{machine_type}#{clamping}"
                
                # 根據用戶權限確定正確的路徑
                user_admin = st.session_state.get("data_maintenance_data_admin", [])
                user_folder = None
                for admin in user_admin:
                    if admin in ["mac1", "mac3"]:
                        user_folder = admin
                        break
                
                if user_folder:
                    # 構建用戶權限對應的路徑
                    user_nc_code_path = os.path.join(data_path, "cnc_master_manual", user_folder, "nc_code")
                    target_folder_path = os.path.join(user_nc_code_path, target_folder_name)
                    
                    # 檢查文件夾是否存在
                    if os.path.exists(target_folder_path):
                        # 文件夾存在，詢問是否覆蓋
                        st.warning(f"⚠️ 發現已存在的文件夾: {target_folder_name}")
                        st.warning("⚠️ 目標文件夾已存在，是否覆蓋當前文件內容？")
                        
                        # 覆蓋按鈕
                        if st.button("是，覆蓋", key="overwrite_yes"):
                            st.session_state.overwrite_confirmed = True
                            st.rerun()
                        
                        # 取消按鈕
                        if st.button("否，取消", key="overwrite_no"):
                            st.session_state.code_confirmed = False
                            st.rerun()
                    else:
                        # 文件夾不存在，顯示新建信息
                        st.info(f"📁 將創建新文件夾: {target_folder_name}")
                else:
                    st.error("❌ 無法確定用戶權限路徑")
            else:
                st.error("❌ 請填寫機種和夾位信息")
        
        # 始終顯示文件上傳區域（但根據狀態控制行為）
        if st.session_state.get("code_confirmed", False):
            machine_type = st.session_state.get("machine_type", "")
            clamping = st.session_state.get("clamping", "")
            
            if machine_type and clamping:
                target_folder_name = f"{machine_type}#{clamping}"
                user_admin = st.session_state.get("data_maintenance_data_admin", [])
                user_folder = None
                for admin in user_admin:
                    if admin in ["mac1", "mac3"]:
                        user_folder = admin
                        break
                
                if user_folder:
                    user_nc_code_path = os.path.join(data_path, "cnc_master_manual", user_folder, "nc_code")
                    target_folder_path = os.path.join(user_nc_code_path, target_folder_name)
                    
                    # 根據是否覆蓋來決定上傳行為
                    if st.session_state.get("overwrite_confirmed", False):
                        st.info("請選擇要上傳的文件（覆蓋模式）")
                        uploaded_code_files = st.file_uploader(
                            "上傳代碼文件（覆蓋模式）",
                            type=None,
                            accept_multiple_files=True,
                            help="支持所有文件类型，包括NC代码文件（无扩展名）"
                        )
                    else:
                        uploaded_code_files = st.file_uploader(
                            "上傳代碼文件",
                            type=None,
                            accept_multiple_files=True,
                            help="支持所有文件类型，包括NC代码文件（无扩展名）"
                        )
                    
                    # 處理文件上傳
                    if uploaded_code_files:
                        try:
                            # 如果是覆蓋模式，直接保存；如果是新建模式，先創建文件夾
                            if not st.session_state.get("overwrite_confirmed", False):
                                os.makedirs(target_folder_path, exist_ok=True)
                            
                            saved_files = []
                            for uploaded_code_file in uploaded_code_files:
                                # 保存文件
                                file_path = os.path.join(target_folder_path, uploaded_code_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_code_file.getbuffer())
                                saved_files.append(uploaded_code_file.name)
                            
                            # 顯示上傳成功信息
                            if st.session_state.get("overwrite_confirmed", False):
                                st.success(f"✅ 已成功上傳 {len(saved_files)} 個文件")
                            else:
                                st.success(f"✅ 已創建文件夾 {target_folder_name} 並保存 {len(saved_files)} 個文件")
                            st.success(f"📄 文件列表: {', '.join(saved_files)}")
                            
                            # 重置狀態
                            st.session_state.code_confirmed = False
                            st.session_state.overwrite_confirmed = False
                            # st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ 保存失敗: {str(e)}")
        


    # 添加返回按鈕
    st.markdown("---")
    col_return1, col_return2, col_return3 = st.columns([1, 1, 1])
    with col_return1:
        if st.button("返回登入頁面", use_container_width=True):
            st.session_state.current_page = "data_maintenance_login"
            st.rerun()
    with col_return2:
        if st.button("返回主頁", use_container_width=True):
            st.session_state.current_page = "homepage"
            st.rerun()
    with col_return3:
        if st.button("登出", use_container_width=True):
            st.session_state.data_maintenance_logged_in = False
            st.session_state.current_page = "homepage"
            st.rerun() 