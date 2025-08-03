import streamlit as st
import hashlib
import yaml
import pandas as pd
from cnc_genai.demo_ui import conf_init
from cnc_v0.model.agent.langchain.src.utils.auth import verify_password


def load_user_credentials(config_path="../cnc_data/cnc_cred/auth_users.yaml"):
    """
    從Excel文件中加載用戶憑據
    """
    try:

        # 本地方法
        # df = pd.read_excel("../cnc_data/access_control_user_data.xlsx")
        # credentials = {}
        # for _, row in df.iterrows():
        #     username = row["username"]
        #     hashed_password = hashlib.sha256(str(row["password"]).encode()).hexdigest()
        #     access_level = row["access_level"].split(";")
        #     credentials[username] = [hashed_password, access_level]

        # pipeline build method
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        credentials = {}
        acc_data = data.get("account_name", {})
        for k in acc_data:
            username = k
            hashed_password = acc_data[k].get("password", None)
            access_level = acc_data[k].get("access_level", [])
            credentials[username] = [hashed_password, access_level]

        return credentials
    except Exception as e:
        st.error(f"加載用戶憑據時出錯: {str(e)}")
        return {}


def render_login_page():
    """渲染登入頁面"""
    # 添加自定義 CSS 來移除表單的白色邊框和背景
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
            <div style='text-align: center; padding: 20px; background-color: rgba(0,255,255,0.1); border-radius: 10px; margin-bottom: 30px;'>
                <h2>歡迎使用 CNC360 V1</h2>
                <h3>CT時間優化助手</h3>
                <p>請輸入您的用戶名和密碼</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 創建一個居中的容器，寬度為600px
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 創建登入表單
        with st.form("login_form"):
            username = st.text_input("用戶名")
            password = st.text_input("密碼", type="password")
            submit_button = st.form_submit_button("登入", use_container_width=True)

            # 從Excel文件加載用戶憑據
            valid_users = load_user_credentials()

            if valid_users:
                if submit_button:
                    if username not in valid_users:
                        st.error(
                            "用戶名錯誤，請重試，注意B05-3F的账号名为b05-3，请检查是否为大写"
                        )
                        return
                    else:
                        try:
                            check_password_res = verify_password(
                                password, valid_users[username][0]
                            )
                            if check_password_res:
                                st.session_state.logged_in = True
                                st.session_state.username = username
                                st.session_state.current_page = "landing"
                                st.session_state.available_clampings = (
                                    get_available_clamping(
                                        floor_list=valid_users[username][1]
                                    )
                                )
                                st.rerun()
                            else:
                                st.error("密碼錯誤，請重試")
                        except Exception as e:
                            st.error(f"其他錯誤: {e}")

    # 添加幫助信息
    st.markdown(
        """
        <div style='max-width: 600px; margin: 0 auto; margin-top: 50px; text-align: center; color: #888;'>
            <p>請使用您的帳號密碼登入系統</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_available_clamping(floor_list=["admin"]):

    # 这个文件每周更新，不采用
    # mapper_path = '../cnc_data/input/equCramp.parquet'
    # df_mapper = pd.read_parquet(mapper_path)
    # data = list(df_mapper.data[0])
    # df = pd.DataFrame(data)
    # df['available_clamping'] = df['modelname'] + '-' + df['campingname']

    # 使用沙哥拉到的历史代码，补正系统口径，保持一致
    st.session_state.machine_df = conf_init.parse_folder_names_to_dataframe(
        f"../cnc_data/nc_code/"
    )
    df = st.session_state.machine_df.copy()
    df["available_clamping"] = df["product_type"] + "-" + df["clamping"]

    floor_list = [
        f"{x}F" if (x != "admin" and not x.endswith("F")) else x for x in floor_list
    ]

    # 为管理员用户添加app目录下的夹位
    if "admin" in floor_list:
        clamping_list = [x.replace(" ", "-") for x in df["available_clamping"].unique()]

        # 额外添加app目录下的夹位
        import glob
        from pathlib import Path

        # 获取所有符合模式的simulation_master目录
        master_paths = glob.glob("../app/*/simulation_master/*")

        # 遍历每个simulation_master目录
        for path in master_paths:
            # 解析路径: ../app/mac1/simulation_master/X2867-CNC2
            path_parts = Path(path).parts
            if len(path_parts) >= 4:
                department = path_parts[-3]  # mac1
                product = path_parts[-1]  # X2867-CNC2

                # 添加为 mac1/X2867-CNC2
                app_clamping = f"{department}/{product}"
                clamping_list.append(app_clamping)

        return clamping_list
    else:
        df = df[df["floor"].isin(floor_list)]
        clamping_list = [x.replace(" ", "-") for x in df["available_clamping"].unique()]
        return clamping_list
