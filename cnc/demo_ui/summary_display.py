import streamlit as st
import pandas as pd
import json
import numpy as np
from io import BytesIO
import zipfile
import datetime
import os
import re
import subprocess
import shutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events
from cnc_genai.demo_ui.config_section_rerun import (
    get_image_path,
    resize_image_to_target_width,
)
from cnc_genai.demo_ui import conf_init
from cnc_genai.demo_ui.image_annotation import (
    numpy_nearest_resize,
    visualize_code_area,
)
from cnc_genai.src.simulation.utils import load_from_zst
from cnc_genai.demo_ui.code_diff_view import show_code_diff
from cnc_genai.src.v1_algo.generate_nc_code import validate_codes


def render_summary_results() -> None:
    """
    Render a summary table of optimization results.

    Args:
        df_load_analysis: DataFrame containing optimization results
    """
    # Get aggregated summary

    df_load_analysis = st.session_state.df_load_analysis
    df_load_analysis["time_physical_improved_air"] = (
        df_load_analysis["time_physical_improved"] * df_load_analysis["apply_air"]
    )

    summary_df = (
        df_load_analysis.groupby(["sub_program_key", "function"])
        .agg(
            {
                "real_ct": "min",
                "time_physical_improved_air": "sum",
                "time_physical_improved": "sum",
            }
        )
        .reset_index()
    )

    # Rename columns for better display
    summary_df.columns = [
        "子程式",
        "加工內容",
        "標準CT (秒)",
        "空切降低CT (秒)",
        "總降低CT (秒)",
    ]

    # Calculate improvement percentage
    summary_df["改善幅度"] = (
        summary_df["總降低CT (秒)"] / summary_df["標準CT (秒)"] * 100
    ).round(1).astype(str) + "%"

    # Format numeric columns to 2 decimal places
    summary_df["標準CT (秒)"] = summary_df["標準CT (秒)"].round(2)
    summary_df["空切降低CT (秒)"] = summary_df["空切降低CT (秒)"].round(2)
    summary_df["總降低CT (秒)"] = summary_df["總降低CT (秒)"].round(2)
    summary_df["AFC降低CT (秒)"] = (
        summary_df["總降低CT (秒)"] - summary_df["空切降低CT (秒)"]
    ).round(2)

    summary_df = summary_df[
        [
            "子程式",
            "加工內容",
            "標準CT (秒)",
            "空切降低CT (秒)",
            "AFC降低CT (秒)",
            "總降低CT (秒)",
            "改善幅度",
        ]
    ]
    # Display the table with styling
    st.markdown("### 優化結果總覽")

    total_pct = (
        summary_df["總降低CT (秒)"].sum() / summary_df["標準CT (秒)"].sum() * 100
    ).round(1)

    st.write(f"總體預計改善{total_pct}%")
    st.write(f"總體標準CT{summary_df['標準CT (秒)'].sum().round(0)}(秒)")
    st.write(f"總體預計改善CT{summary_df['總降低CT (秒)'].sum().round(1)}（秒）")
    st.write(f"其中空切CT改善{summary_df['空切降低CT (秒)'].sum().round(1)}（秒）")
    st.write(f"其中非空切CT改善{summary_df['AFC降低CT (秒)'].sum().round(1)}（秒）")

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        height=(len(summary_df) + 1) * 35,
        column_config={
            "提速幅度": st.column_config.TextColumn(
                "提速幅度", help="相較於標準CT的提速百分比"
            )
        },
    )


def render_optimization_results():
    """
    Render the optimization results with interactive filters and charts.
    """
    if "df_load_analysis" not in st.session_state:
        st.warning("No optimization results available. Please run the analysis first.")
        return

    df = st.session_state.df_load_analysis

    summary_df = (
        df.groupby(["sub_program", "function"])
        .agg(
            {
                "time_physical_improved": "sum",
            }
        )
        .reset_index()
    )

    # Create sub-program filter
    sub_programs = df["sub_program"].unique()
    selected_sub_program = st.selectbox(
        "選擇子程序", options=sub_programs, key="sub_program_selector"
    )

    # Filter data
    filtered_df = df[df["sub_program"] == selected_sub_program]
    filtered_df["real_ct"] = filtered_df["real_ct"].fillna(0)
    function = filtered_df["function"].to_list()[0]

    pct = (
        filtered_df["time_physical_improved"].sum()
        / filtered_df["real_ct"].astype(int).min()
        * 100
    ).round(1)
    txt1 = f"總標準CT時間{filtered_df['real_ct'].astype(int).min()}(秒)"
    txt2 = (
        f"CT時間降低{filtered_df['time_physical_improved'].sum().round(1)}(秒) & {pct}%"
    )

    # Option 2: With separator and emphasis (same color)
    st.markdown(
        f"""<div style="display: flex; align-items: center; gap: 1rem; font-size: 1.1rem;">
            <span style="font-weight: 700;">{txt1}</span>
            <span style="opacity: 0.5;">|</span>
            <span style="font-weight: 700;">{txt2}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Legend
    st.markdown(
        """
        <style>
            .legend-container {
                display: flex;
                justify-content: space-around;
                margin-top: 10px;
            }
            .legend-item {
                display: flex;
                align-items: center;
                margin-right: 15px;
            }
            .legend-line {
                display: inline-block;
                width: 20px;
                height: 2px;
                margin-right: 5px;
            }
            .green-line { background-color: #2ca02c; }
            .red-dotted-line { background-color: #d62728; border-bottom: 1px dotted; }
            .blue-line { background-color: #1f77b4; }
            .orange-line { background-color: #ff7f0e; }
        </style>
        <div class="legend-container">
            <div class="legend-item"><span class="legend-line green-line"></span><b>預測功率</b></div>
            <div class="legend-item"><span class="legend-line red-dotted-line"></span><b>目標功率</b></div>
            <div class="legend-item"><span class="legend-line blue-line"></span><b>F</b></div>
            <div class="legend-item"><span class="legend-line orange-line"></span><b>優化F</b></div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"子程序 {selected_sub_program} - {function} 的功率對比",
            f"子程序 {selected_sub_program} - {function} 的F對比",
        ),
    )

    # First subplot - Power comparison
    trace1 = go.Scatter(
        x=filtered_df["time_physical_acc"].astype(float).to_list(),
        y=filtered_df["power_pc"].astype(float).to_list(),
        name="功率",
        line=dict(color="#2ca02c"),
        mode="lines+markers",
        customdata=filtered_df.index,
        selectedpoints=[],
        showlegend=False,  # Hide legend
    )
    fig.add_trace(trace1, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=filtered_df["time_physical_acc"].astype(float).to_list(),
            y=filtered_df["target_power_pc"].astype(float).to_list(),
            name="目標功率",
            line=dict(color="#d62728", dash="dot"),
            showlegend=False,  # Hide legend
        ),
        row=1,
        col=1,
    )

    # Second subplot - Force comparison
    # First add F_adjusted (will be underneath)
    fig.add_trace(
        go.Scatter(
            x=filtered_df["time_physical_acc"].astype(float).to_list(),
            y=filtered_df["F_adjusted"].astype(float).to_list(),
            name="優化F",
            line=dict(color="#ff7f0e"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Then add F on top (will cover F_adjusted where values are same)
    trace3 = go.Scatter(
        x=filtered_df["time_physical_acc"].astype(float).to_list(),
        y=filtered_df["F"].astype(float).to_list(),
        name="F",
        line=dict(color="#1f77b4"),
        mode="lines+markers",
        customdata=filtered_df.index,
        selectedpoints=[],
        showlegend=False,
    )
    fig.add_trace(trace3, row=2, col=1)

    # Update layout
    fig.update_layout(
        xaxis_title="累積時間 (秒)",
        xaxis2_title="累積時間 (秒)",
        showlegend=False,  # Hide legend
        height=800,
        width=1200,
        template="plotly_dark",
        dragmode="select",
        margin=dict(t=150, b=50, r=50, l=50),
    )

    # Display the plot
    selected_points = plotly_events(
        fig,
        click_event=False,
        select_event=True,
        override_height=800,
        override_width="100%",
        key="plot",
    )

    # Automatically display selected data without button
    if len(selected_points) > 0:
        selected_times = [point["x"] for point in selected_points]
        display_df = filtered_df[
            filtered_df["time_physical_acc"]
            .round(6)
            .isin([round(x, 6) for x in selected_times])
        ]
        st.write("選擇的代碼行:")

        display_df["MRR"] = display_df["MRR"].round(0)  # Round MRR to 1 decimal
        display_df["ap_mm"] = display_df["ap_mm"].round(1)
        display_df["ae_mm"] = display_df["ae_mm"].round(1)
        display_df = display_df[
            [
                "sub_program",
                "function",
                "N",
                "src",
                "row_id",
                "ap_mm",
                "ae_mm",
                "MRR",
                "time_physical",
                "is_valid",
                "is_finishing",
                "F",
                "F_adjusted",
            ]
        ].rename(
            columns={
                "is_valid": "是否切割",
                "is_finishing": "是否精修",
                "MRR": "材料去除率mm3/min",
                "time_physical": "切削時間s",
                "sub_program": "子程式",
                "function": "描述",
                "row_id": "行",
                "F_adjusted": "F優化",
            }
        )

        display_df["是否切割"] = display_df["是否切割"].map({1: True, 0: False})

        st.dataframe(display_df)

        # 添加按鈕來顯示程序圖
        if st.button("顯示代碼段刀路"):

            print(st.session_state.scenario_name)

            try:
                hyper_params_dict, sub_programs_df, ban_n_df, ban_row_df = (
                    conf_init.load_rerun_conf(
                        st.session_state.selected_department,
                        st.session_state.scenario_name,
                    )
                )

                # load image from zst path and resize to target and save to session_state
                product_image, product_image_origin = load_from_zst(
                    input_path=get_image_path(hyper_params_dict, sub_programs_df)
                )
                resized_image, resize_scale = resize_image_to_target_width(
                    product_image, target_width=600
                )
                st.session_state.product_image = resized_image
                st.session_state.resize_scale = resize_scale
                st.session_state.product_image_origin = product_image_origin
                st.session_state["bbox_data_ban_n"] = {}
                st.session_state["bbox_data_ban_row"] = {}
                print(
                    f"load {st.session_state.selected_clamping} image from zst path done, reset annotation to empty..."
                )
            except Exception as e:
                print(f"无法从zst文件加载图像: {e}")
                print("创建模拟边界图像...")

            # 读取Excel文件获取代码信息
            st.session_state.df_load_analysis = pd.read_excel(
                f"../app/{st.session_state.selected_department}/scenario/{st.session_state.scenario_name}/load_analysis.xlsx"
            )

            # 獲取所有唯一的子程序和N值組合
            unique_programs = display_df["子程式"].unique()

            # 對每個子程序和N值生成圖片
            for program_id in unique_programs:
                # 獲取當前子程序的所有N值
                program_df = display_df[display_df["子程式"] == program_id]
                unique_N = program_df["N"].unique()

                for n in unique_N:
                    # 顯示子程序和N值信息
                    st.write(f"子程序: {program_id}, N段: {n}")

                    # 創建兩列佈局
                    col1, col2 = st.columns([1.5, 1])

                    # 獲取該子程序中所有具有相同N值的行
                    if pd.isna(n):
                        selected_code = st.session_state.df_load_analysis[
                            (
                                st.session_state.df_load_analysis["sub_program"].astype(
                                    str
                                )
                                == str(program_id)
                            )
                            & (st.session_state.df_load_analysis["N"].isna())
                        ]
                    else:
                        selected_code = st.session_state.df_load_analysis[
                            (
                                st.session_state.df_load_analysis["sub_program"].astype(
                                    str
                                )
                                == str(program_id)
                            )
                            & (st.session_state.df_load_analysis["N"] == n)
                        ]

                    # 篩選移動代碼行
                    move_code_rows = selected_code[
                        selected_code["move_code"].isin(["G01", "G02", "G03"])
                    ]

                    if not move_code_rows.empty:
                        # 獲取刀具半徑
                        tool_radius = (
                            move_code_rows["tool_diameter"].iloc[0] / 2
                            if "tool_diameter" in move_code_rows.columns
                            else 0
                        )

                        # 獲取X和Y坐標的範圍，並加上刀具半徑
                        x_min = move_code_rows["X_pixel"].min() - tool_radius
                        x_max = move_code_rows["X_pixel"].max() + tool_radius
                        y_min = move_code_rows["Y_pixel"].min() - tool_radius
                        y_max = move_code_rows["Y_pixel"].max() + tool_radius
                        z_max = move_code_rows["Z_pixel"].max()

                        # 構建四個角點的坐標
                        bbox_coords = {
                            "x_coords": [x_min, x_max],
                            "y_coords": [y_min, y_max],
                            "z_coords": z_max,
                            "tool_radius": tool_radius,  # 添加刀具半徑信息
                        }

                        with col1:

                            # 將移動代碼行傳遞給visualize_code_area函數，用於繪製路徑
                            visualize_code_area(
                                product_image=st.session_state.product_image,
                                scale=st.session_state.resize_scale,
                                bbox_coords=bbox_coords,
                                selected_code=selected_code,  # 修改這一行，傳遞完整的selected_code
                            )

                        with col2:
                            # 顯示代碼內容
                            # 獲取所有行號
                            all_line_numbers = selected_code["row_id"].tolist()
                            # 獲取移動代碼的行號
                            move_line_numbers = move_code_rows["row_id"].tolist()

                            # 找到第一個和最後一個移動代碼的位置
                            first_move = min(move_line_numbers)
                            last_move = max(move_line_numbers)

                            # 構建顯示的代碼
                            display_lines = []
                            for idx, row in selected_code.iterrows():
                                if (
                                    row["row_id"] <= first_move + 2
                                    or row["row_id"] >= last_move - 2
                                ):
                                    # 添加行號和源代碼
                                    display_lines.append(
                                        f"{row['row_id']}: {row['src']}"
                                    )
                                elif (
                                    len(display_lines) > 0
                                    and "..." not in display_lines[-1]
                                ):
                                    # 添加省略號
                                    display_lines.append("...")

                            # 顯示代碼
                            st.code("\n".join(display_lines))

                    # 添加分隔線
                    st.divider()


def render_nc_code():
    """Render NC code with subprogram filtering."""

    df = st.session_state.df_load_analysis
    code_lines = st.session_state.code_lines
    old_codes = st.session_state.old_codes

    # 驗證子程序
    failed_sub_programs = []
    for sub_program in code_lines.keys():
        if not validate_codes(sub_program, code_lines, old_codes):
            failed_sub_programs.append(sub_program)
    if failed_sub_programs:
        st.error(f"子程序 {', '.join(failed_sub_programs)} 驗證失敗")
    else:
        st.success("所有子程序驗證成功")

    # 添加标记
    new_codes = {}
    for k in st.session_state.code_lines.keys():
        lines = st.session_state.code_lines[k].split("\n")
        if lines:
            if st.session_state.selected_folder != "製工標準":
                tag = f"(GENERATED BY CNC360 BASED ON {st.session_state.selected_floor}#{st.session_state.selected_machine}#{st.session_state.selected_date_version})"
                if tag not in lines[0]:
                    lines[0] = lines[0] + tag
            else:
                tag = "(GENERATED BY CNC360 BASED ON ZHI GONG BAN BEN)"
                if tag not in lines[0]:
                    lines[0] = lines[0] + tag
            new_codes[k] = "\n".join(lines)

    # Create sub-program filter
    sub_programs = df["sub_program"].unique()
    selected_subprogram = st.selectbox(
        "選擇子程序", options=sub_programs, key="sub_program_selector_code"
    )

    # CNC机台操作系统选择
    st.session_state.cnc_os = st.radio(
        "CNC機台操作系統",
        options=["Fanuc", "Mitsubishi"],
        key="cnc_os_selector",
        horizontal=True,
    )

    # Create download button for all programs
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for program_number in new_codes.keys():
            # 過濾條件，只處理屬於當前產品的子程序
            if str(program_number) not in sub_programs:
                continue
            if st.session_state.cnc_os == "Mitsubishi":
                zip_file.writestr(
                    f"{program_number}", new_codes[program_number].encode("utf-8")
                )
            else:  # Fanuc
                zip_file.writestr(
                    f"O{program_number}.txt", new_codes[program_number].encode("utf-8")
                )
    zip_buffer.seek(0)

    # 保存 zip 文件到指定目錄
    try:
        # 創建目錄結構
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        new_string = re.sub(
            r"^([^-]+-[^-]+)-", r"\1#", st.session_state.selected_clamping
        )
        scenario_path = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.scenario_name}"
        os.makedirs(scenario_path, exist_ok=True)

        # 保存 zip 文件
        zip_file_path = os.path.join(scenario_path, "nc_programs.zip")
        with open(zip_file_path, "wb") as f:
            f.write(zip_buffer.getvalue())

        # 重置 zip_buffer 位置，以便下載按鈕使用
        zip_buffer.seek(0)
    except Exception as e:
        st.warning(f"無法保存 zip 文件到指定目錄: {str(e)}")

    # 将生成的代码额外保存一份json格式
    with open(
        f"../app/{st.session_state.selected_department}/scenario/{st.session_state.scenario_name}/nc_programs.json",
        "w",
    ) as f:
        json.dump(new_codes, f)
    with open(
        f"../app/{st.session_state.selected_department}/scenario/{st.session_state.scenario_name}/nc_programs_old.json",
        "w",
    ) as f:
        json.dump(old_codes, f)

    if not failed_sub_programs:

        st.download_button(
            label="下載所有程序",
            data=zip_buffer,
            file_name="nc_programs.zip",
            mime="application/zip",
            use_container_width=True,
            type="secondary",
        )

        # with st.expander("展開上傳CNC代碼"):
        #     code_files = st.file_uploader(
        #         "上傳CNC代碼文件",
        #         accept_multiple_files=True,
        #         key="code_files_multi_vis",
        #     )
        #     if code_files:
        #         # 將所有上傳文件放入一個cache目錄
        #         today_str = datetime.datetime.now().strftime(
        #             "%Y-%m-%d"
        #         )  # 寫成2025-03-12格式
        #         new_string = re.sub(
        #             r"^([^-]+-[^-]+)-", r"\1#", st.session_state.selected_clamping
        #         )
        #         output_path = f"../cnc_data/{st.session_state.selected_department}/nc_code/{st.session_state.selected_floor}#{new_string}#{st.session_state.selected_machine}TS#{today_str}"

        #         # 創建cache目錄，如果已存在則先清空
        #         cache_dir = "../cnc_data/cache_uploaded_nc"
        #         if os.path.exists(cache_dir):
        #             shutil.rmtree(cache_dir)  # 徹底刪除目錄及其內容
        #         os.makedirs(cache_dir, exist_ok=True)  # 重新創建乾淨的目錄

        #         # 保存上傳文件到cache目錄
        #         for uploaded_file in code_files:
        #             file_path = os.path.join(cache_dir, uploaded_file.name)
        #             with open(file_path, "wb") as f:
        #                 f.write(uploaded_file.getbuffer())

        #         # 執行轉換命令
        #         subprocess.run(
        #             [
        #                 "python",
        #                 "cnc_genai/src/utils/convert_txt_to_json.py",
        #                 "-i",
        #                 cache_dir,
        #                 "-o",
        #                 output_path,
        #             ]
        #         )
        #         st.success(f"NC代碼已成功上傳並轉換，存儲路徑: {output_path}")

    # Display selected subprogram
    if selected_subprogram:
        program_number = str(selected_subprogram)
        st.markdown(f"### 子程序 O{program_number} NC代碼")
        show_code_diff(old_codes[program_number], new_codes[program_number])


def render_nc_analysis():
    st.subheader("NC代碼分析細節數據")

    col1, col2 = st.columns([1, 1])
    with col1:
        # 點擊按鈕請下載這個excel：f"../app/{department}/scenario/{scenario_name}/{scenario_name}.xlsx"
        scenario_path = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.scenario_name}/{st.session_state.scenario_name}.xlsx"

        # 檢查文件是否存在
        if os.path.exists(scenario_path):
            with open(scenario_path, "rb") as file:
                exp_scenario = file.read()

            st.download_button(
                label="點擊下載實驗配置",
                data=exp_scenario,
                file_name=f"{st.session_state.scenario_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.warning(f"找不到實驗配置文件: {scenario_path}")

    with col2:
        details_path = f"../app/{st.session_state.selected_department}/scenario/{st.session_state.scenario_name}/load_analysis.xlsx"

        # 檢查文件是否存在
        if os.path.exists(details_path):
            with open(details_path, "rb") as file:
                exp_details = file.read()

            st.download_button(
                label="點擊下載提速明細",
                data=exp_details,
                file_name="exp_details.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.warning(f"找不到提速明細文件: {details_path}")

    df = st.session_state.df_load_analysis
    # st.write(list(df))
    sub_programs = df["sub_program"].unique()
    selected_sub_program = st.selectbox(
        "選擇子程序", options=sub_programs, key="sub_program_selector_analysis"
    )

    # Add is_valid filter
    is_valid_options = ["全部", "是", "否"]
    selected_is_valid = st.selectbox(
        "是否空切", options=is_valid_options, key="is_valid_selector"
    )

    # Filter data
    filtered_df = df[df["sub_program"] == selected_sub_program]

    # Apply is_valid filter if not "全部"
    if selected_is_valid != "全部":
        is_valid_value = 0 if selected_is_valid == "是" else 1
        filtered_df = filtered_df[filtered_df["is_valid"] == is_valid_value]

    # Display specific columns from df_load_analysis
    display_columns = [
        "sub_program",
        "function",
        "src",
        "row_id",
        "F",
        "ap_mm",
        "ae_mm",
        "MRR",
        "是否空切",
        "F_adjusted",
    ]

    # Create a copy of filtered_df for display
    display_df = filtered_df.copy()
    # Convert is_valid values to "是"/"否"
    display_df["是否空切"] = display_df["is_valid"].map({1: "否", 0: "是"})

    # Create a styled dataframe
    display_df_subset = display_df[display_columns]
    styled_df = (
        display_df_subset.style.format(
            {
                "ap_mm": "{:.1f}",
                "ae_mm": "{:.1f}",
                "F": "{:.0f}",
                "MRR": "{:.0f}",
                "F_adjusted": "{:.0f}",
            }
        )
        .apply(
            lambda df: [
                (
                    "background-color: darkred; color: white"
                    if col == "F_adjusted" and val != df["F"]
                    else ""
                )
                for col, val in df.items()
            ],
            axis=1,
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("padding", "3px"),
                        ("font-size", "11px"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("padding", "3px"),
                        ("font-size", "11px"),
                        ("white-space", "nowrap"),
                    ],
                },
            ]
        )
    )

    st.dataframe(styled_df, hide_index=True, use_container_width=True)
