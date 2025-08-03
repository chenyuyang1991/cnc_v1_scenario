import streamlit as st
import pandas as pd
from cnc_genai.demo_ui import conf_init
from cnc_genai.demo_ui.image_annotation import image_annotation
from cnc_genai.src.utils.utils import check_intersections
from cnc_genai.src.v1_algo.predict_feed_rate import (
    load_clamping_ml_input,
    FeedRateRegressor,
)


def render_parameter_settings_tab(hyper_params_config) -> dict:
    """Render the first tab with parameter settings"""
    hyper_params = {}
    col1, col2 = st.columns(2)

    with col1:
        hyper_params["percentile_threshold"] = st.number_input(
            "主軸負載目標百分位 (Percentile Threshold)",
            value=float(hyper_params_config["percentile_threshold"]),
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="以各子程序内負載的多少百分位作為目標均衡負載優化（注：不是機台的負載%），數值越高代表越激进",
        )

        hyper_params["multiplier_max"] = st.number_input(
            "最大倍率 (Max Multiplier)",
            value=float(hyper_params_config["multiplier_max"]),
            min_value=1.0,
            max_value=5.0,
            step=0.1,
            format="%.1f",
            help="設定進給速率的最大倍率",
        )

        hyper_params["multiplier_min"] = st.number_input(
            "最小倍率 (Min Multiplier)",
            value=float(hyper_params_config["multiplier_min"]),
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            format="%.1f",
            help="設定進給速率的最小倍率",
        )

    with col2:
        hyper_params["multiplier_air"] = st.number_input(
            "空切倍率 (Air Cut Multiplier)",
            value=float(hyper_params_config["multiplier_air"]),
            min_value=1.0,
            max_value=5.0,
            step=0.1,
            format="%.1f",
            help="設定空切時的進給速率倍率",
        )

        hyper_params["apply_finishing"] = st.selectbox(
            "區分精修/開粗 (Apply Finishing)",
            options=["不區分", "區分"],
            index=1 if hyper_params_config["apply_finishing"] == 1 else 0,
            help="是否區分精修/開粗 (1:區分, 0:不區分)",
        )
        if hyper_params["apply_finishing"] == "區分":
            hyper_params["apply_finishing"] = 1
        else:
            hyper_params["apply_finishing"] = 0

        hyper_params["multiplier_finishing"] = st.number_input(
            "精修段最大倍率 (Finishing Multiplier)",
            value=float(hyper_params_config["multiplier_finishing"]),
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            format="%.1f",
            help="設定精修段的進給速率倍率",
        )

    return hyper_params


def render_get_n_from_image_annotation(df_ban_n):

    st.session_state["bbox_data_ban_n"] = image_annotation(
        st.session_state.product_image,
        st.session_state.product_image_origin,
        key_suffix="ban_n",
        precision=st.session_state.precision,
    )
    if st.button(
        "確認標註",
        key="confirm_annotation_block_n",
        use_container_width=True,
        type="primary",
    ):
        df_load_analysis = st.session_state.df_load_analysis
        code_snippets_to_ban = []
        for sub_program, v in st.session_state["bbox_data_ban_n"].items():
            for coord_label, vv in v.items():
                for selected_mm, vvv in vv.items():
                    defect_list = vv[selected_mm]["bboxes"]
                    df_filtered = df_load_analysis.copy()
                    # 垂直切削平面的道路不考虑
                    if coord_label == "Z":
                        df_filtered = df_filtered[
                            ~(
                                (df_filtered["X_prev_pixel"] == df_filtered["X_pixel"])
                                & (
                                    df_filtered["Y_prev_pixel"]
                                    == df_filtered["Y_pixel"]
                                )
                            )
                        ]
                    elif coord_label == "Y":
                        df_filtered = df_filtered[
                            ~(
                                (df_filtered["X_prev_pixel"] == df_filtered["X_pixel"])
                                & (
                                    df_filtered["Z_prev_pixel"]
                                    == df_filtered["Z_pixel"]
                                )
                            )
                        ]
                    elif coord_label == "X":
                        df_filtered = df_filtered[
                            ~(
                                (df_filtered["Y_prev_pixel"] == df_filtered["Y_pixel"])
                                & (
                                    df_filtered["Z_prev_pixel"]
                                    == df_filtered["Z_pixel"]
                                )
                            )
                        ]
                    else:
                        raise ValueError(f"Invalid coord_label: {coord_label}")
                    # 只考虑特定子程序
                    if sub_program != "所有子程式":
                        df_filtered = df_filtered[
                            df_filtered["sub_program"].astype(str).str.zfill(4)
                            == str(sub_program).zfill(4)
                        ]
                    df_filtered["is_defect"] = df_filtered.apply(
                        lambda x: check_intersections(x, defect_list, coord_label),
                        axis=1,
                    )
                    code_snippet_to_ban = (
                        df_filtered[df_filtered["is_defect"].apply(len) > 0]
                        .groupby(["sub_program", "function", "N"])
                        .size()
                        .reset_index()
                    )
                    code_snippets_to_ban.append(code_snippet_to_ban)
                    # st.write(f'{sub_program} | {coord_label} | {selected_mm} | {len(code_snippet_to_ban)}')

        if code_snippets_to_ban:
            to_ban_from_image = pd.concat(
                code_snippets_to_ban, ignore_index=True
            ).drop_duplicates()
            to_ban_from_image = (
                to_ban_from_image.explode("N")
                .reset_index(drop=True)
                .rename(
                    columns={
                        "sub_program": "sub_program",
                        "function": "function",
                        "N": "ban_n",
                    }
                )[["sub_program", "function", "ban_n"]]
            )

            # Update df_ban_n in session state
            df_ban_n = pd.concat(
                [
                    df_ban_n,
                    to_ban_from_image,
                ]
            ).drop_duplicates()
    if "rerun_tab_results" not in st.session_state:
        st.session_state.rerun_tab_results = {}
    st.session_state.rerun_tab_results["ban_n_df"] = df_ban_n
    return df_ban_n


def render_get_row_from_image_annotation(df_ban_row):
    st.session_state["bbox_data_ban_row"] = image_annotation(
        st.session_state.product_image,
        st.session_state.product_image_origin,
        key_suffix="ban_row",
        precision=st.session_state.precision,
    )

    # get additional ban n from annotation
    if st.button(
        "確認標註",
        key="st_keys.confirm_annotation_block_row",
        use_container_width=True,
        type="primary",
    ):
        df_load_analysis = st.session_state.df_load_analysis
        dfs = [df_ban_row]

        for sub_program, v in st.session_state["bbox_data_ban_row"].items():
            for coord_label, vv in v.items():
                for selected_mm, vvv in vv.items():
                    defect_list = vvv["bboxes"]
                    df_filtered = df_load_analysis.copy()
                    # 垂直切削平面的道路不考虑
                    if coord_label == "Z":
                        df_filtered = df_filtered[
                            ~(
                                (df_filtered["X_prev_pixel"] == df_filtered["X_pixel"])
                                & (
                                    df_filtered["Y_prev_pixel"]
                                    == df_filtered["Y_pixel"]
                                )
                            )
                        ]
                    elif coord_label == "Y":
                        df_filtered = df_filtered[
                            ~(
                                (df_filtered["X_prev_pixel"] == df_filtered["X_pixel"])
                                & (
                                    df_filtered["Z_prev_pixel"]
                                    == df_filtered["Z_pixel"]
                                )
                            )
                        ]
                    elif coord_label == "X":
                        df_filtered = df_filtered[
                            ~(
                                (df_filtered["Y_prev_pixel"] == df_filtered["Y_pixel"])
                                & (
                                    df_filtered["Z_prev_pixel"]
                                    == df_filtered["Z_pixel"]
                                )
                            )
                        ]
                    # 只考虑特定子程序
                    if sub_program != "所有子程式":
                        df_filtered = df_filtered[
                            df_filtered["sub_program"].astype(str).str.zfill(4)
                            == str(sub_program).zfill(4)
                        ]
                    df_filtered["is_defect"] = df_filtered.apply(
                        lambda x: check_intersections(x, defect_list, coord_label),
                        axis=1,
                    )
                    code_snippet_to_ban = (
                        df_filtered[df_filtered["is_defect"].apply(len) > 0][
                            ["sub_program", "function", "N", "row_id", "src"]
                        ]
                        .rename({"row_id": "ban_row"}, axis=1)
                        .drop_duplicates()
                    )
                    dfs.append(code_snippet_to_ban)
                    # st.write(f'{sub_program} | {coord_label} | {selected_mm} | {len(code_snippet_to_ban)}')
        df_ban_row = pd.concat(dfs, axis=0)
        # todo: add tool and tool_spec
        df_ban_row["tool"] = None
        df_ban_row["tool_spec"] = None
    st.session_state["ban_row_df"] = df_ban_row  # 保存到 session_state
    return df_ban_row


def render_sub_program_settings_tab(
    df=conf_init.load_sub_program_init(),
) -> pd.DataFrame:
    """Render the second tab with program settings"""

    df["sub_program"] = df["sub_program"].astype(str).str.zfill(4)
    st.markdown('<div class="sub-program-config">', unsafe_allow_html=True)

    st.markdown(
        """
    <style>
    .sub-program-config .stDataFrame {
        font-size: 14px;
    }
    .sub-program-config div[data-testid="stHorizontalBlock"] {
        gap: 0rem;
        padding: 0rem;
        align-items: center;
    }
    .sub-program-config div[data-testid="stNumberInput"],
    .sub-program-config div[data-testid="stSelectbox"] {
        margin-top: 0rem;
    }
    .sub-program-config select {
        margin-top: 0rem;
    }
    /* 添加表格樣式 */
    .program-table-container {
        width: 100%;
        overflow-x: auto;
        margin-bottom: 20px;
        border-collapse: collapse;
    }
    .program-table {
        width: 100%;
        min-width: 800px;
        border-collapse: collapse;
    }
    .program-table th {
        background-color: #262730;
        padding: 10px;
        text-align: left;
        font-weight: bold;
    }
    .program-table td {
        padding: 10px;
        border-bottom: 1px solid #4e4e5e;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 開始表格容器
    st.markdown('<div class="sub-program-table-container">', unsafe_allow_html=True)

    # 只有當 use_cnc_knowledge_base 從 False 變為 True 時才進行計算
    if (
        st.session_state.use_cnc_knowledge_base
        and st.session_state.use_cnc_knowledge_base_changed
    ):
        ml_input = load_clamping_ml_input(
            st.session_state.selected_department, st.session_state.selected_clamping
        )
        ml_input["clamping"] = st.session_state.selected_clamping
        st.session_state.regressor = FeedRateRegressor()
        st.session_state.regressor.load_model()
        st.session_state.regressor.train_data_set = None
        st.session_state.regressor.eval_data_set = {
            st.session_state.selected_clamping: ml_input
        }
        st.session_state.regressor.data_preprocess()
        st.session_state.regressor.eval_data_df["F_pred"] = (
            st.session_state.regressor.predict(
                st.session_state.regressor.X_eval, target_col="F"
            )
        )
        st.session_state.regressor.eval_data_df["FoS_pred"] = (
            st.session_state.regressor.predict(
                st.session_state.regressor.X_eval, target_col="FoS"
            )
        )
        st.session_state.regressor.eval_data_df["F_pred_constraint"] = (
            st.session_state.regressor.eval_data_df["S"]
            * st.session_state.regressor.eval_data_df["FoS_pred"]
        )
        st.session_state.regressor.eval_data_df["F_pred_final"] = (
            st.session_state.regressor.eval_data_df[
                ["F_pred_constraint", "F_pred"]
            ].min(axis=1)
        )

        st.session_state.regressor.eval_data_df["F_multiplier_pred_final"] = (
            st.session_state.regressor.eval_data_df["F_pred_final"]
            / st.session_state.regressor.eval_data_df["F"]
        )
        # st.session_state.regressor.eval_data_df["F_multiplier_pred_final"] = (
        #     st.session_state.regressor.eval_data_df["F_multiplier_pred_final"].clip(
        #         lower=1.0, upper=1.5
        #     )
        # )
        ml_res = st.session_state.regressor.calc_mape(
            st.session_state.regressor.eval_data_df, input_col="F_multiplier_pred_final"
        )
        ml_res = ml_res.reset_index()[
            ["sub_program", "F_multiplier_pred_wavg_subprogram"]
        ].drop_duplicates()
        ml_res["sub_program"] = ml_res["sub_program"].astype(str)
        ml_res["F_multiplier_pred_wavg_subprogram"] = ml_res[
            "F_multiplier_pred_wavg_subprogram"
        ].fillna(1.0)
        st.session_state.sub_program_default_multiplier = dict(
            zip(ml_res["sub_program"], ml_res["F_multiplier_pred_wavg_subprogram"])
        )
    elif not st.session_state.use_cnc_knowledge_base:
        st.session_state.sub_program_default_multiplier = {}

    # 表頭
    if "sub_program_default_multiplier" in st.session_state and (
        st.session_state.sub_program_default_multiplier != {}
    ):
        cols_width = [0.5, 1.5, 1, 1, 1, 1, 1, 1, 0.5, 0.2]
    else:
        cols_width = [0.5, 1.5, 1, 1, 1, 1, 1, 1, 0.2]
    cols = st.columns(cols_width)
    cols[0].markdown("**子程式**")
    cols[1].markdown("**加工內容**")
    cols[2].markdown("**刀號**")
    # cols[3].markdown("**刀具規格型號**")
    cols[3].markdown("**是否是精修子程序**")
    cols[4].markdown("**空切優化**")
    cols[5].markdown("**切削優化**")
    cols[6].markdown("**轉角優化**")
    cols[7].markdown("**最大倍率**")
    if "sub_program_default_multiplier" in st.session_state and (
        st.session_state.sub_program_default_multiplier != {}
    ):
        cols[8].markdown("**提示**")

    sub_programs = {}
    for i, row in df.iterrows():
        cols = st.columns(cols_width)
        cols[0].markdown(f"**{row['sub_program']}**")

        # 修改處理加工內容的邏輯
        function = str(row["function"]) if not pd.isna(row["function"]) else ""
        if function:
            cols[1].markdown(str(function).replace("nan", ""))
        else:
            cols[1].markdown("")  # 空值時顯示空白

        # 添加刀號和刀具規格型號輸入欄位
        cols[2].markdown("")
        # cols[3].markdown("")

        finishing = cols[3].toggle(
            "是",
            value=bool(row["finishing"]),
            key=f"st_keys.finishing_{row['sub_program']}_{i}",
        )

        apply_air = cols[4].toggle(
            "開啟",
            value=bool(row["apply_air"]),
            key=f"st_keys.air_{row['sub_program']}_{i}",
        )

        apply_afc = cols[5].toggle(
            "開啟",
            value=bool(row["apply_afc"]),
            key=f"st_keys.afc_{row['sub_program']}_{i}",
        )

        apply_turning = cols[6].toggle(
            "開啟",
            value=bool(row.get("apply_turning", 1)) if apply_afc else False,
            disabled=not apply_afc,
            key=f"st_keys.turning_{row['sub_program']}_{i}",
        )

        # 計算最小和最大值限制
        min_value = st.session_state.temp_hyper_params_dict["multiplier_min"]
        max_value = st.session_state.temp_hyper_params_dict["multiplier_max"]

        # 如果是精修程序且精修優化開關打開，最大值不超過精修倍率
        if finishing and (
            st.session_state.temp_hyper_params_dict["apply_finishing"] == 1
        ):
            max_value = min(
                max_value,
                st.session_state.temp_hyper_params_dict["multiplier_finishing"],
            )

        # 獲取初始值並確保在範圍內
        initial_value = (
            st.session_state.sub_program_default_multiplier[str(row["sub_program"])]
            if "sub_program_default_multiplier" in st.session_state
            and str(row["sub_program"])
            in st.session_state.sub_program_default_multiplier
            else row.get("multiplier_max", 1.0)
        )
        # 裁剪初始值確保在範圍內
        initial_value = max(min_value, min(max_value, initial_value))

        multiplier = cols[7].number_input(
            "",
            value=initial_value,
            min_value=min_value,
            max_value=max_value,
            step=0.01,
            format="%.2f",
            key=f"st_keys.mult_{row['sub_program']}_{i}",
            label_visibility="collapsed",
            help="hint to develop",
        )

        # 如果sub_program_default_multiplier不為空，則顯示提示
        if "sub_program_default_multiplier" in st.session_state and (
            st.session_state.sub_program_default_multiplier != {}
        ):
            cols[8].markdown("", help="hint to develop")

        sub_programs[str(row["sub_program"])] = {
            "function": function,  # 使用處理過的 function 值
            "tool": "",  # 使用用戶輸入的刀號
            "tool_spec": "",  # 使用用戶輸入的刀具規格型號
            "finishing": int(finishing),
            "apply_afc": int(apply_afc),
            "apply_air": int(apply_air),
            "apply_turning": int(apply_turning),
            "multiplier_max": multiplier,
        }

    # 處理空字典的情況
    if sub_programs:
        sub_programs_df = pd.DataFrame.from_dict(
            sub_programs, orient="index"
        ).reset_index()
        sub_programs_df.columns = [
            "sub_program",
            "function",
            "tool",
            "tool_spec",
            "finishing",
            "apply_afc",
            "apply_air",
            "apply_turning",
            "multiplier_max",
        ]
    else:
        # 如果沒有子程序，返回空的 DataFrame 且具有正確的列結構
        sub_programs_df = pd.DataFrame(
            columns=[
                "sub_program",
                "function",
                "tool",
                "tool_spec",
                "finishing",
                "apply_afc",
                "apply_air",
                "apply_turning",
                "multiplier_max",
            ]
        )
    st.markdown("</div>", unsafe_allow_html=True)

    return sub_programs_df


def render_block_disable_tab(
    ban_n_df=pd.DataFrame(
        columns=["sub_program", "function", "tool", "tool_spec", "ban_n"]
    )
) -> pd.DataFrame:
    """Render the third tab for disabling specific blocks"""

    ban_n_df["sub_program"] = ban_n_df["sub_program"].astype(str).str.zfill(4)

    # 添加CSS來實現水平滾動
    st.markdown(
        """
    <style>
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        min-width: max-content;
        overflow-x: auto;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="block-disable-config">', unsafe_allow_html=True)

    df_n = conf_init.load_all_nc_block(
        st.session_state.selected_department, st.session_state.selected_clamping
    )
    df_n["sub_program"] = df_n["sub_program"].astype(str).str.zfill(4)

    # Get unique sub_programs in original order
    unique_sub_programs = df_n["sub_program"].unique()
    # Group while preserving order
    grouped = df_n.groupby("sub_program", sort=False)

    ban_n_dict = {}

    # 初始化清空標誌
    st.session_state.clear_ban_n = False
    # 添加一鍵清空按鈕
    if st.button("一鍵清空所有禁用N段", type="secondary", use_container_width=True):
        st.session_state.clear_ban_n = True

    cols = st.columns([1, 2, 1, 1, 2])
    with cols[0]:
        st.markdown(f"**子程式**")
    with cols[1]:
        st.markdown(f"**加工內容**")  # Display the function name
    with cols[2]:
        st.markdown(f"**刀號**")
    with cols[3]:
        st.markdown(f"**刀具規格型號**")
    with cols[4]:
        st.markdown(f"**禁止提速N段**")

    # Iterate through sub_programs in original order
    for i, sub_program in enumerate(unique_sub_programs):
        group = grouped.get_group(sub_program)
        preselected_bans = ban_n_df[
            ban_n_df["sub_program"].astype(str).str.zfill(4)
            == str(sub_program).zfill(4)
        ]["ban_n"].tolist()
        to_be_selected_bans = group["ban_n"].tolist()

        # Filter preselected_bans to ensure all values are in to_be_selected_bans
        preselected_bans = [
            ban for ban in preselected_bans if ban in to_be_selected_bans
        ]

        # Get the function name for the current sub_program
        function_name = group["function"].iloc[
            0
        ]  # Assuming all rows have the same function
        cols = st.columns([1, 2, 1, 1, 2])
        with cols[0]:
            st.markdown(f"{sub_program}")
        with cols[1]:
            st.markdown(
                str(function_name).replace("nan", "")
            )  # Display the function name
        with cols[2]:
            st.markdown("")  # Display the function name
        with cols[3]:
            st.markdown("")  # Display the function name
        with cols[4]:
            ban_n_dict[str(int(sub_program))] = {
                "function": function_name,
                "tool": "",
                "tool_spec": "",
                "bans": st.multiselect(
                    "",
                    options=to_be_selected_bans,
                    default=[] if st.session_state.clear_ban_n else preselected_bans,
                    key=f"st_keys.ban_n_{str(int(sub_program))}_{i}",
                    label_visibility="collapsed",
                ),
            }
    # 處理空字典的情況
    if ban_n_dict:
        ban_n_df = pd.DataFrame.from_dict(ban_n_dict, orient="index").reset_index()
        ban_n_df.columns = [
            "sub_program",
            "function",
            "tool",
            "tool_spec",
            "ban_n",
        ]
        ban_n_df["sub_program"] = ban_n_df["sub_program"].astype(str).str.zfill(4)
        ban_n_df = ban_n_df.explode("ban_n").reset_index(drop=True)
        ban_n_df = ban_n_df[~pd.isna(ban_n_df["ban_n"])]
    else:
        # 如果沒有子程序，返回空的 DataFrame 且具有正確的列結構
        ban_n_df = pd.DataFrame(
            columns=[
                "sub_program",
                "function",
                "tool",
                "tool_spec",
                "ban_n",
            ]
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 重置清空標誌
    if st.session_state.clear_ban_n:
        st.session_state.clear_ban_n = False

    # for _, row in ban_n_df.iterrows():
    #     st.write(f"O{row['sub_program']} - {row['function']}: {row['ban_n']}")

    return ban_n_df


def render_row_disable_tab(ban_row_df):
    # 添加CSS來實現水平滾動
    st.markdown(
        """
    <style>
    .element-container div.stDataFrame {
        width: 100%;
        overflow-x: auto;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    ban_row_df["sub_program"] = ban_row_df["sub_program"].astype(str).str.zfill(4)
    st.markdown("#### 關閉優化的程式碼行:")

    # 確保DataFrame有必需的列
    if len(ban_row_df) > 0:
        # 添加一個操作列，用戶可以通過勾選來選擇要刪除的行
        if "選擇刪除" not in ban_row_df.columns:
            ban_row_df["選擇刪除"] = False

        # 使用 data_editor 來創建互動式表格
        edited_df = st.data_editor(
            ban_row_df,
            hide_index=True,
            column_config={
                "選擇刪除": st.column_config.CheckboxColumn(
                    "選擇刪除",
                    help="勾選要刪除的行",
                    default=False,
                ),
                "sub_program": st.column_config.TextColumn(
                    "子程序",
                    help="子程序編號",
                    disabled=True,
                ),
                "function": st.column_config.TextColumn(
                    "子程序名稱",
                    help="子程序名稱",
                    disabled=True,
                ),
                "tool": st.column_config.TextColumn(
                    "刀號",
                    help="刀號",
                    disabled=True,
                ),
                "tool_spec": st.column_config.TextColumn(
                    "刀具規格型號",
                    help="刀具規格型號",
                    disabled=True,
                ),
                "N": st.column_config.TextColumn(
                    "N",
                    help="子程序N",
                    disabled=True,
                ),
                "ban_row": st.column_config.NumberColumn(
                    "行號",
                    help="程式行號",
                    disabled=True,
                ),
                "src": st.column_config.TextColumn(
                    "原程式",
                    help="原程式",
                    disabled=True,
                ),
            },
            use_container_width=True,
            num_rows="dynamic",
        )

        # 添加刪除按鈕
        if st.button("刪除選中行", type="primary"):
            # 過濾出未被選中刪除的行
            ban_row_df = edited_df[~edited_df["選擇刪除"]].reset_index(drop=True)
            # 移除選擇刪除列，以便下次顯示時重新添加
            ban_row_df = ban_row_df.drop(columns=["選擇刪除"])
            st.session_state["ban_row_df"] = ban_row_df  # 保存到 session_state
            st.success("已刪除選中的行")

        st.write(f"總共 **{len(ban_row_df)}** 行被標記為關閉")

        # 移除"選擇刪除"列後返回數據
        if "選擇刪除" in ban_row_df.columns:
            ban_row_df = ban_row_df.drop(columns=["選擇刪除"])

        return ban_row_df
    return pd.DataFrame(
        columns=[
            "sub_program",
            "function",
            "tool",
            "tool_spec",
            "N",
            "ban_row",
            "src",
        ]
    )


def render_advanced_settings_tab(
    default_params={
        "short_threshold": 0.2,
        "ae_thres": 0.1,
        "ap_thres": 0.1,
        "target_pwc_strategy": "按刀具",
        "min_air_speed": 0.0,
        "max_increase_step": 2000.0,
        "max_air_speed": 15000.0,
    }
):
    """Render the fourth tab with advanced settings"""
    adv_params = {}
    col1, col2 = st.columns(2)

    with col1:
        adv_params["short_threshold"] = st.number_input(
            "連續空切時間閾值 (Short Threshold)",
            value=float(default_params["short_threshold"]),
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            format="%.1f",
            help="優化多少秒以上的空切段",
        )

        adv_params["ae_thres"] = st.number_input(
            "徑向切寬精修閾值 (AE Threshold)",
            value=float(default_params["ae_thres"]),
            min_value=0.1,
            max_value=2.0,
            step=0.1,
            format="%.1f",
            help="徑向切寬(ae)多少mm以下判定為精修",
        )

        adv_params["min_air_speed"] = st.number_input(
            "最小空切速度 (Min Air Speed)",
            value=float(default_params["min_air_speed"]),
            min_value=0.0,
            max_value=10000.0,
            step=1000.0,
            format="%.0f",
            help="最小空切速度",
        )

    with col2:
        adv_params["ap_thres"] = st.number_input(
            "軸向切深精修閾值 (AP Threshold)",
            value=float(default_params["ap_thres"]),
            min_value=0.05,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="軸向切深(ap)多少mm以下判定為精修",
        )

        # 新增目標功率策略選項
        adv_params["target_pwc_strategy"] = "按刀具"
        # adv_params["target_pwc_strategy"] = st.selectbox(
        #     "目標功率計算策略",
        #     options=["按刀具", "按刀具組"],
        #     index=0,  # 預設選第一個選項
        #     help="選擇目標功率的計算策略模式",
        # )

        adv_params["max_air_speed"] = st.number_input(
            "最大空切速度 (Max Air Speed)",
            value=float(default_params["max_air_speed"]),
            min_value=0.0,
            max_value=99999.0,
            step=1000.0,
            format="%.0f",
            help="最大空切速度",
        )

        adv_params["max_increase_step"] = st.number_input(
            "F最大增量 (Max Increase Step)",
            value=float(default_params["max_increase_step"]),
            min_value=0.0,
            max_value=99999.0,
            step=1000.0,
            format="%.0f",
            help="進給速度最大增量",
        )

    return adv_params
