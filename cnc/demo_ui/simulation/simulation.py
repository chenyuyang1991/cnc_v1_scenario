import streamlit as st
import pandas as pd
import platform
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import time
import glob
import shutil
import subprocess
import threading
import io
import base64
import datetime
import json
from cnc_genai.src.utils.utils import load_config_v1
from cnc_genai.src.simulation.utils import (
    load_from_zst,
    save_to_zst,
    convert_stl_to_numpy,
)

# from cnc_genai.demo_ui.simulation.cad_viewer import display_stl
from cnc_genai.src.code_parsing.code_parsing import run_code_parsing
from cnc_genai.src.simulation.colors import MATERIAL_COLOR, EMPTY_COLOR


# 設置中文字體
def set_matplotlib_chinese_font():
    # 嘗試設置中文字體，按優先順序嘗試
    chinese_fonts = [
        "Arial Unicode MS",
        "Heiti TC",
        "PingFang TC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "STHeiti",
    ]

    # 查找可用的中文字體
    available_font = None
    for font in chinese_fonts:
        if any(f.name == font for f in fm.fontManager.ttflist):
            available_font = font
            break

    if available_font:
        plt.rcParams["font.family"] = [available_font, "sans-serif"]
    else:
        # 如果找不到列表中的字體，使用系統默認字體
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]

    # 解決負號顯示問題
    plt.rcParams["axes.unicode_minus"] = False


def generate_origin_image(origin, array_data=None, precision=4):
    """
    生成原點位置示意圖

    Args:
        origin: 原點位置值，表示該軸的原點設置
        array_data: 工件形狀尺寸 (寬度, 高度)

    Returns:
        plt.figure: 生成的圖像對象
    """
    # 設置中文字體
    set_matplotlib_chinese_font()

    # 創建圖像
    if array_data is None:
        none_input_tag = True
        array_data = np.zeros((1000, 1600, 1, 3))
        array_data[:] = MATERIAL_COLOR
    else:
        none_input_tag = False

    fig, ax = plt.subplots()
    # 設置圖表背景為半透明白色
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(0.05)

    width = array_data.shape[1]
    height = array_data.shape[0]

    # 繪製工件矩形
    ax.imshow(array_data[:, :, -1], cmap="viridis")

    # 計算原點位置（如果是百分比則轉換）
    origin_x, origin_y, origin_z = origin

    # 設置原點在工件上的位置
    origin_point_x = (
        origin_x * 10 ** (precision - 3) if origin_x > 1 else width * origin_x
    )
    origin_point_y = (
        height - origin_y * 10 ** (precision - 3)
        if origin_y > 1
        else height - height * origin_y
    )
    origin_point_z = origin_z * 10 ** (precision - 3)
    # st.write(f"origin_x: {origin_x}, origin_y: {origin_y}")
    # st.write(f"origin_point_x: {origin_point_x}, origin_point_y: {origin_point_y}")

    if none_input_tag and origin_x > 1:
        origin_point_x = 0.5 * width
    if none_input_tag and origin_y > 1:
        origin_point_y = height - 0.5 * height

    # 畫一個紅色圓點作為原點標記
    circle_radius = 1
    circle = plt.Circle(
        (origin_point_x, origin_point_y),
        radius=circle_radius,
        color="red",
        fill=True,
        alpha=0.7,
    )
    ax.add_patch(circle)
    ax.text(
        origin_point_x + 20,
        origin_point_y + 60,
        "工件坐標係原點",
        color="red",
        fontsize=12,
        fontweight="bold",
    )

    # 繪製X軸原點箭頭（從左側到原點，與原點在同一高度）
    arrow_start_x = 0
    arrow_start_y = origin_point_y  # 修改為與原點同高度

    # 計算箭頭長度，使箭頭尖剛好碰到圓的邊緣
    arrow_length = origin_point_x - arrow_start_x - circle_radius

    # 繪製箭頭
    ax.arrow(
        arrow_start_x,
        arrow_start_y,
        arrow_length,
        0,
        head_width=0.3,
        head_length=min(0.3, arrow_length / 3),
        fc="blue",
        ec="blue",
        width=0.05,
    )

    # 在箭頭上方添加x值標籤
    arrow_mid_x = arrow_start_x + arrow_length / 2
    x_display = (
        f"x={origin[0]:.2f}mm"
        if origin[0] > 1
        else f"x={int(float(origin[0])*100)}%產品長度"
    )
    ax.text(
        arrow_mid_x,
        arrow_start_y + 10,
        x_display,
        color="blue",
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="top",
        fontweight="bold",
    )

    # 繪製Y軸原點箭頭（從下方到原點，與原點在同一橫軸）
    arrow_start_x = origin_point_x  # 修改為與原點同橫軸
    arrow_start_y = height

    # 計算箭頭長度，使箭頭尖剛好碰到圓的邊緣
    arrow_length = height - origin_point_y - circle_radius

    # 繪製箭頭
    ax.arrow(
        arrow_start_x,
        arrow_start_y,
        0,
        -arrow_length,
        head_width=0.3,
        head_length=min(0.3, arrow_length / 3),
        fc="blue",
        ec="blue",
        width=0.05,
    )

    # 在箭頭左側添加y值標籤
    arrow_mid_y = arrow_start_y - arrow_length / 2
    y_display = (
        f"y={origin[1]:.2f}mm"
        if origin[1] > 1
        else f"y={int(float(origin[1])*100)}%產品寬度"
    )
    ax.text(
        arrow_start_x - 5,
        arrow_mid_y,
        y_display,
        color="blue",
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="center",
        fontweight="bold",
    )

    # 移除座標軸
    ax.axis("off")

    # 添加X軸和Y軸主線
    # X軸 - 從原點向右
    arrow_length = min(width, height) * 0.2
    ax.plot(
        [origin_point_x, origin_point_x + arrow_length],
        [origin_point_y, origin_point_y],
        color="white",
        linestyle="-",
        alpha=1,
        linewidth=2,
    )
    # Y軸 - 從原點向上
    ax.plot(
        [origin_point_x, origin_point_x],
        [origin_point_y, origin_point_y - arrow_length],
        color="white",
        linestyle="-",
        alpha=1,
        linewidth=2,
    )

    # 添加箭頭頭部
    # X軸箭頭
    ax.arrow(
        origin_point_x,
        origin_point_y,
        arrow_length,
        0,
        head_width=min(width, height) * 0.02,
        head_length=min(width, height) * 0.02,
        fc="white",
        ec="white",
        alpha=1,
        linewidth=2,
    )
    # Y軸箭頭
    ax.arrow(
        origin_point_x,
        origin_point_y,
        0,
        -arrow_length,
        head_width=min(width, height) * 0.02,
        head_length=min(width, height) * 0.02,
        fc="white",
        ec="white",
        alpha=1,
        linewidth=2,
    )

    # 添加坐標軸標籤
    ax.text(
        origin_point_x + arrow_length + min(width, height) * 0.03,
        origin_point_y,
        "X",
        fontsize=12,
        fontweight="bold",
        color="white",
    )
    ax.text(
        origin_point_x,
        origin_point_y - arrow_length - min(width, height) * 0.03,
        "Y",
        fontsize=12,
        fontweight="bold",
        color="white",
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, [origin_point_x, origin_point_y, origin_point_z]


def generate_origin_image_3d(origin, origin_types, array_data=None, precision=4):
    """
    生成原點位置三維示意圖

    Args:
        origin: 原點位置值，表示該軸的原點設置
        array_data: 工件形狀尺寸 (寬度, 高度)
        precision: 精度設置

    Returns:
        plotly.graph_objects.Figure: 生成的3D圖像對象
        list: [origin_point_x, origin_point_y, origin_point_z] 原點坐標
    """
    import plotly.graph_objects as go
    import numpy as np

    # 計算原點位置
    origin_x, origin_y, origin_z = origin
    origin_x_type, origin_y_type, origin_z_type = origin_types

    # 處理輸入參數
    if array_data is None:
        width = 1600
        height = 1000
        depth = 200  # 默認深度
        origin_point_x = origin_x * width if 0 < origin_x < 1 else 0.5 * width
        origin_point_y = origin_y * height if 0 < origin_y < 1 else 0.5 * height
        origin_point_z = origin_z * depth if 0 < origin_z < 1 else 0.5 * depth
    else:
        width = array_data.shape[1]
        height = array_data.shape[0]
        depth = array_data.shape[2] if array_data.ndim > 2 else 200

        # 設置原點在工件上的位置
        origin_point_x = (
            width * origin_x if 0 < origin_x < 1 else origin_x * 10 ** (precision - 3)
        )
        origin_point_y = (
            height * origin_y if 0 < origin_y < 1 else origin_y * 10 ** (precision - 3)
        )
        origin_point_z = (
            depth * origin_z if 0 < origin_z < 1 else origin_z * 10 ** (precision - 3)
        )

    # 建立3D圖形
    fig = go.Figure()

    # 添加工件的3D模型（立方體）
    fig.add_trace(
        go.Mesh3d(
            x=[0, width, width, 0, 0, width, width, 0],
            y=[0, 0, height, height, 0, 0, height, height],
            z=[0, 0, 0, 0, depth, depth, depth, depth],
            # 定義立方體的所有6個面（12個三角形）
            # 底面 (z=0)
            i=[
                0,
                0,
                # 頂面 (z=depth)
                4,
                4,
                # 前面 (y=0)
                0,
                0,
                # 後面 (y=height)
                2,
                2,
                # 左面 (x=0)
                0,
                0,
                # 右面 (x=width)
                1,
                1,
            ],
            j=[1, 2, 5, 6, 1, 4, 3, 6, 3, 4, 2, 5],
            k=[2, 3, 6, 7, 5, 5, 7, 7, 7, 7, 6, 6],
            opacity=0.2,
            color="lightblue",
            flatshading=True,
            name="工件",
        )
    )

    # 添加原點（紅色球體）
    fig.add_trace(
        go.Scatter3d(
            x=[origin_point_x],
            y=[origin_point_y],
            z=[origin_point_z],
            mode="markers",
            marker=dict(
                size=10,
                color="red",
            ),
            name="工件坐標系原點",
        )
    )

    # 添加原點信息文本
    x_display = (
        f"X距左側{origin[0]:.2f}mm"
        if origin_x_type == "mm"
        else f"X距左側{int(float(origin[0])*100)}%產品長度"
    )
    y_display = (
        f"Y距下側{origin[1]:.2f}mm"
        if origin_y_type == "mm"
        else f"Y距下側{int(float(origin[1])*100)}%產品寬度"
    )
    z_display = (
        f"Z距底部{origin[2]:.2f}mm"
        if origin_z_type == "mm"
        else f"Z距底部{int(float(origin[2])*100)}%產品厚度"
    )

    # X軸箭頭（從原點向右）
    arrow_length = min(width, height) * 0.2
    fig.add_trace(
        go.Scatter3d(
            x=[origin_point_x, origin_point_x + arrow_length],
            y=[origin_point_y, origin_point_y],
            z=[origin_point_z, origin_point_z],
            mode="lines",
            line=dict(color="red", width=5),
            name=x_display,
        )
    )

    # Y軸箭頭（從原點向前）
    fig.add_trace(
        go.Scatter3d(
            x=[origin_point_x, origin_point_x],
            y=[origin_point_y, origin_point_y + arrow_length],
            z=[origin_point_z, origin_point_z],
            mode="lines",
            line=dict(color="green", width=5),
            name=y_display,
        )
    )

    # Z軸箭頭（從原點向上）
    fig.add_trace(
        go.Scatter3d(
            x=[origin_point_x, origin_point_x],
            y=[origin_point_y, origin_point_y],
            z=[origin_point_z, origin_point_z + arrow_length],
            mode="lines",
            line=dict(color="blue", width=5),
            name=z_display,
        )
    )

    # 設置圖形佈局
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X",
                # 將刻度值除以10
                tickvals=list(range(0, int(width) + 1, 200)),
                ticktext=[str(val // 10) for val in range(0, int(width) + 1, 200)],
            ),
            yaxis=dict(
                title="Y",
                # 將刻度值除以10
                tickvals=list(range(0, int(height) + 1, 200)),
                ticktext=[str(val // 10) for val in range(0, int(height) + 1, 200)],
            ),
            zaxis=dict(
                title="Z",
                # 將刻度值除以10
                tickvals=list(range(0, int(depth) + 1, 50)),
                ticktext=[str(val // 10) for val in range(0, int(depth) + 1, 50)],
            ),
            aspectmode="data",  # 保持比例一致
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder="normal",
            font=dict(size=12),
        ),
    )

    # 設置初始視角
    camera = dict(
        eye=dict(x=0, y=-2, z=1),  # 從Y軸負方向看過去，Z軸稍微向上
        center=dict(x=0, y=0, z=0),  # 設置視角中心為工件中心
        up=dict(x=0, y=0, z=1),  # 確保Z軸是向上的
    )
    fig.update_layout(scene_camera=camera)

    return fig, [origin_point_x, origin_point_y, origin_point_z]


def render_simulation():
    """
    渲染CNC仿真頁面，顯示仿真結果
    """
    # 打印診斷信息
    print("render_simulation診斷:")
    print(f"session_state.clamping_name: {st.session_state.get('clamping_name', None)}")
    if "simulation_config" in st.session_state:
        print(
            f"simulation_config['clamping_name']: {st.session_state.simulation_config.get('clamping_name', None)}"
        )
        print(
            f"simulation_config['path']['dir_app']: {st.session_state.simulation_config['path'].get('dir_app', None)}"
        )

    # 設置matplotlib中文字體
    set_matplotlib_chinese_font()

    st.subheader("CNC切割仿真")

    st.markdown(
        "#### <span style='color:yellow'>第3步: 確定毛坯圖形</span>",
        unsafe_allow_html=True,
    )

    if "simulating" not in st.session_state:
        st.session_state["simulating"] = False

    if "last_check_time" not in st.session_state:
        st.session_state["last_check_time"] = time.time()

    if "simulation_started" not in st.session_state:
        st.session_state["simulation_started"] = False

    if "cmd" not in st.session_state:
        st.session_state["cmd"] = ""

    if "process_pid" not in st.session_state:
        st.session_state["process_pid"] = None

    # 創建兩列佈局
    col1, col2, col3 = st.columns([1, 1, 1])

    # 上傳夾位毛坯圖形文件
    with col1:

        st.markdown("##### 3.1. 輸入毛坯圖形")
        if "input_method" not in st.session_state:
            st.session_state["input_method"] = "file"
        if "stock_path" not in st.session_state:
            st.session_state["stock_path"] = None

        # 初始化 array_data，解決未定義問題
        array_data = None

        input_method = st.radio(
            "請選擇輸入方式",
            ["上傳夾位毛坯圖形文件", "人工測量毛坯近似圖形", "使用同機種仿真結果"],
            index=0,
            key="input_method_radio",
        )

        if "allow_simulation" not in st.session_state:
            st.session_state["allow_simulation"] = False

        # 根據選擇顯示不同的輸入界面
        if input_method == "上傳夾位毛坯圖形文件":
            st.session_state["input_method"] = "file"

            graphic_file = st.file_uploader(
                "請上傳夾位毛坯的STP/STL/ZST格式圖形文件",
                type=["stl", "zst", "stp"],
                key="graphic_file",
            )
            if graphic_file is not None and st.session_state.get("clamping_name"):
                # 保存上傳的文件
                st.session_state["stock_path"] = (
                    f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/{graphic_file.name}"
                )
                with open(st.session_state["stock_path"], "wb") as f:
                    f.write(graphic_file.getbuffer())

                # 如果是ZST文件，顯示一些信息
                if graphic_file.name.endswith(".zst"):

                    # 從ZST文件加載數據到numpy數組
                    array_data, _ = load_from_zst(st.session_state["stock_path"])
                    st.session_state["allow_simulation"] = True

                # 如果是STL文件，不提供預覽
                elif graphic_file.name.endswith(".stl") or graphic_file.name.endswith(
                    ".stp"
                ):
                    # 讓用戶輸入xyz的origin
                    st.success(
                        "已上傳STL/STP毛坯文件，在啟動任務後係統會將其轉換為像素矩陣，右側僅為示意圖..."
                    )
                    array_data = None
                    st.session_state["allow_simulation"] = True
                else:
                    st.error("請上傳正確的文件格式")
                    array_data = None
                    st.session_state["allow_simulation"] = False
        elif input_method == "人工測量毛坯近似圖形":
            st.session_state["input_method"] = "manual"

            st.write("請輸入毛坯形狀尺寸：")
            col11, col22 = st.columns([1, 1])
            with col11:
                manual_length = st.number_input(
                    "工件長度 X (mm)", min_value=0.0, value=173.0, step=1.0
                )
                manual_width = st.number_input(
                    "工件寬度 Y (mm)", min_value=0.0, value=85.0, step=1.0
                )
                manual_height = st.number_input(
                    "工件高度 Z (mm)", min_value=0.0, value=9.2, step=1.0
                )
                manual_thickness = st.number_input(
                    "工件底部厚度 (mm)", min_value=0.0, value=9.2, step=0.1
                )
            with col22:
                manual_thickness_left = st.number_input(
                    "工件左側厚度 (mm)", min_value=0.0, value=1.0, step=0.1
                )
                manual_thickness_right = st.number_input(
                    "工件右側厚度 (mm)", min_value=0.0, value=1.0, step=0.1
                )
                manual_thickness_top = st.number_input(
                    "工件上側厚度 (mm)", min_value=0.0, value=1.0, step=0.1
                )
                manual_thickness_bottom = st.number_input(
                    "工件下側厚度 (mm)", min_value=0.0, value=1.0, step=0.1
                )

            # 添加確認按鈕
            if st.button("確認毛坯尺寸並生成", key="confirm_manual_stock"):
                # 生成手動輸入的形狀數據
                array_data = generate_stock(
                    precision=st.session_state["precision"],
                    manual_length=manual_length,
                    manual_width=manual_width,
                    manual_height=manual_height,
                    manual_thickness=manual_thickness,
                    manual_thickness_left=manual_thickness_left,
                    manual_thickness_right=manual_thickness_right,
                    manual_thickness_top=manual_thickness_top,
                    manual_thickness_bottom=manual_thickness_bottom,
                )
                # 將生成的數據保存到session_state中
                st.session_state["manual_array_data"] = array_data
                st.session_state["allow_simulation"] = True
                st.success("毛坯數據已生成！")

            # 如果已經生成過數據，則使用保存的數據
            if "manual_array_data" in st.session_state:
                array_data = st.session_state["manual_array_data"]
                st.session_state["allow_simulation"] = True

            # 顯示手動生成的圖形預覽
            # st.write("圖像預覽")
            # fig, ax = plt.subplots()
            # ax.imshow(array_data[:, :, -1], cmap="viridis")
            # ax.axis("off")
            # st.pyplot(fig)

        else:  # 使用同機種仿真結果
            # 這裡首先要從夾位中提取出機種，例如Diamond-Cell-CNC5，中Diamond-Cell就是機種。遍歷所有夾位找到相同機種的夾位的仿真結果，即simulation_master/{夾位}/simulation/lastest中的*.zst
            array_data = None  # 初始化 array_data

            # 檢查 clamping_name 是否存在
            if not st.session_state.get("clamping_name"):
                st.error("請先設定夾位名稱")
                return

            product_type = st.session_state.clamping_name.rsplit("-", 1)[0]
            st.write(f"當前機種: {product_type}")

            # 找到所有simulation_master/{product_type}*/simulation/lastest/*.zst,並返回所有*.zst的路徑和對應的{product_type}*，即夾位
            zst_files = glob.glob(
                f"{st.session_state.simulation_config['path']['dir_app']}/{product_type}*/simulation/latest/*.zst"
            )
            options = [zst_files.split("/")[4] for zst_files in zst_files]
            # 加一个下拉菜单，选择夹位，找到对应的zst文件
            st.session_state["selected_zst"] = None
            selected_zst = st.selectbox("選擇仿真結果", options)
            zst_paths = glob.glob(
                f"{st.session_state.simulation_config['path']['dir_app']}/{selected_zst}/simulation/latest/*.zst"
            )
            if len(zst_paths) > 0:
                zst_path = zst_paths[0]
                st.session_state["selected_zst"] = selected_zst
                file_name = zst_path.split("/")[-1]
            else:
                st.error(
                    f"未找到當前機種: {product_type}的仿真結果，請檢查{st.session_state.simulation_config['path']['dir_app']}/{selected_zst}/simulation/latest/目錄下是否存在.zst文件"
                )
                st.stop()
            if st.button("確認毛坯文件"):
                if len(zst_paths) > 0:
                    # 复制到f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}"目录下同名
                    shutil.copy(
                        zst_path,
                        f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/{file_name}",
                    )
                    st.session_state["stock_path"] = (
                        f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/{file_name}"
                    )
                    st.session_state["allow_simulation"] = True
                    st.success(
                        f"毛坯文件已確認，已復製到{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}目錄下"
                    )
                    # display_stl(st.session_state["stock_path"])
                else:
                    st.error("未找到可用的仿真文件，請重新選擇")
            if st.session_state.get("selected_zst") is None:
                st.error("請選擇毛坯文件")

        # 這裡加一個radio 選項，是否選擇二值圖像進行仿真
        # use_binary_image = st.radio(
        #     "使用二值圖像進行仿真",
        #     ["是", "否"],
        #     index=0,
        #     key="use_binary_image_radio",
        # )
        # if use_binary_image == "是":
        #     st.info("使用二值圖像進行仿真，可以減少內存使用，加速仿真")
        #     st.session_state["use_binary_image"] = True
        # else:
        #     st.info("使用浮點圖像進行仿真，可以追蹤每行代碼的切削過程")
        #     st.session_state["use_binary_image"] = False

        st.session_state["precision"] = 4
        st.session_state["resolution"] = 1.0
        st.session_state["linear_deflection"] = 0.2
        st.session_state["angular_deflection"] = 0.2

        # col11, col22 = st.columns([1, 1])
        # if "precision" not in st.session_state:
        #     st.session_state["precision"] = 4
        # with col11:
        #     precision = st.number_input(
        #         "數值精度",
        #         min_value=3,
        #         max_value=5,
        #         value=st.session_state["precision"],
        #         step=1,
        #     )
        #     st.session_state["precision"] = precision
        # with col22:
        #     resolution = st.number_input(
        #         "解析度 (上传STL文件时使用)",
        #         min_value=0.1,
        #         max_value=10.0,
        #         value=1.0,
        #         step=0.1,
        #     )
        #     st.session_state["resolution"] = resolution
        # with col11:
        #     linear_deflection = st.number_input(
        #         "STP转STL線性偏差 (上传STL/STP文件时使用)",
        #         min_value=0.0,
        #         max_value=1.0,
        #         value=0.5,
        #     )
        #     st.session_state["linear_deflection"] = linear_deflection
        # with col22:
        #     angular_deflection = st.number_input(
        #         "STP转STL角度偏差 (上传STL/STP文件时使用)",
        #         min_value=0.0,
        #         max_value=1.0,
        #         value=0.2,
        #     )
        #     st.session_state["angular_deflection"] = angular_deflection

    with col2:
        st.markdown("##### 3.2. 設定裝夾方向")

    with col3:
        st.markdown("##### 3.3. 設定原點位置")

        # 使用較寬的單層佈局，為每個參數設置足夠空間
        param_cols = st.columns([1, 1, 1])

        # X原點設置
        with param_cols[0]:
            # 这里加一个radio，选择mm或者%
            origin_x_type = st.radio(
                "X原點位置類型",
                ["mm", "比例(0~1)"],
                index=0,
                horizontal=True,
                key="origin_x_type_radio",
            )
            origin_x = st.number_input(
                f"請以{origin_x_type}輸入，工件左側為零點",
                value=0.0 if origin_x_type == "mm" else 0.5,
                step=0.1,
                format="%.2f",
            )

        # Y原點設置
        with param_cols[1]:
            origin_y_type = st.radio(
                "Y原點位置類型",
                ["mm", "比例(0~1)"],
                index=0,
                horizontal=True,
                key="origin_y_type_radio",
            )
            origin_y = st.number_input(
                f"請以{origin_y_type}輸入，工件下側為零點",
                value=0.0 if origin_y_type == "mm" else 0.5,
                step=0.1,
                format="%.2f",
            )

        # Z原點設置
        with param_cols[2]:
            origin_z_type = st.radio(
                "Z原點位置類型",
                ["mm", "比例(0~1)"],
                index=0,
                horizontal=True,
                key="origin_z_type_radio",
            )
            origin_z = st.number_input(
                f"請以{origin_z_type}輸入，工件底部為零點",
                value=0.0,
                step=0.1,
            )

        origin = [origin_x, origin_y, origin_z]

        # 在最寬的列中顯示幫助圖示
        st.write("##### 工件及原點位置示意圖")
        fig, origin_pixel = generate_origin_image_3d(
            [origin_x, origin_y, origin_z],
            [origin_x_type, origin_y_type, origin_z_type],
            array_data,
            st.session_state["precision"],
        )

        st.plotly_chart(fig)

    # 添加靠左的兩個相鄰按鈕
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        if st.button(
            "返回CNC360 V1首頁", key="back_to_home_sim", use_container_width=True
        ):
            st.session_state.current_page = "landing"
            st.rerun()
    with btn_col2:
        if st.button("返回上一頁", key="back_to_prev_sim", use_container_width=True):
            # 清除simulation頁面創建的session_state變量
            simulation_vars = [
                "simulating",
                "last_check_time",
                "simulation_started",
                "input_method",
                "allow_simulation",
                "stock_path",
                "cmd",
            ]
            for var in simulation_vars:
                if var in st.session_state:
                    del st.session_state[var]

            # 清除驗證狀態，讓用戶需要重新驗證表格
            validation_vars = [
                "validation_completed",
                "validation_results",
            ]
            for var in validation_vars:
                if var in st.session_state:
                    del st.session_state[var]

            # 返回到create_simulation頁面
            st.session_state.current_page = "create_simulation"
            st.rerun()
    with btn_col3:
        if st.button(
            "開始仿真", key="start_simulation", use_container_width=True, type="primary"
        ):
            if st.session_state.get("allow_simulation", False) and st.session_state.get(
                "clamping_name"
            ):
                # 只有在未開始仿真時才顯示開始按鈕
                if not st.session_state.get("simulation_started", False):
                    if not st.session_state.get("allow_simulation", False):
                        st.error("請先上傳毛坯圖形")
                    elif array_data is not None:
                        stock_path = f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/stock.zst"
                        saved_stock_path = save_to_zst(
                            array_data,
                            stock_path,
                            origin=origin_pixel,
                            # binary=bool(st.session_state["use_binary_image"]),
                        )
                        st.session_state["stock_path"] = saved_stock_path

                    st.session_state["simulating"] = True

                    # 解析NC代码
                    st.session_state.simulation_config["path"][
                        "dir_machine_folder"
                    ] = "製工標準"
                    _ = run_code_parsing(
                        st.session_state.simulation_config, verbose=True
                    )

                    # 使用subprocess啟動後台進程
                    product_master_excel_path = f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/{st.session_state.simulation_config['path']['master_path']}"
                    parsed_nc_code_excel_path = f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/{st.session_state.simulation_config['path']['dir_parsed_line']}"
                    tools_excel_path = f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/{st.session_state.simulation_config['path']['tool_path']}"

                    # 檢查 stock_path 是否存在且不為空
                    if not st.session_state.get("stock_path"):
                        st.error("請先上傳毛坯圖形或選擇輸入方式")
                        return

                    stock_path = st.session_state["stock_path"]
                    precision = st.session_state["precision"]
                    linear_deflection = st.session_state["linear_deflection"]
                    angular_deflection = st.session_state["angular_deflection"]
                    resolution = st.session_state["resolution"]
                    r_slack = 2 * 10 ** (3 - precision)
                    z_slack = 2 * 10 ** (3 - precision)
                    numpy_out_path = f"{st.session_state.simulation_config['path']['dir_intermediate']}/{st.session_state['clamping_name']}/{st.session_state.simulation_config['path']['dir_simulation']}"
                    excel_out_path = f"{st.session_state.simulation_config['path']['dir_app']}/{st.session_state['clamping_name']}/{st.session_state.simulation_config['path']['dir_simulation']}"
                    os.makedirs(numpy_out_path, exist_ok=True)
                    os.makedirs(excel_out_path, exist_ok=True)
                    log_file = os.path.join(excel_out_path, "simulation_log.txt")
                    # cmd = f"nohup python -u -m cnc_genai.src.simulation.simulate_new --clamping_name {st.session_state['clamping_name']} --product_master {product_master_excel_path} --parsed_nc_code {parsed_nc_code_excel_path} --tools {tools_excel_path} --stock_path {stock_path} --precision {precision} --verbose --r_slack {r_slack} --z_slack {z_slack} --numpy_out_path {numpy_out_path} --excel_out_path {excel_out_path} > {log_file} 2>&1 &"
                    # cmd = (
                    #    f"nohup python -u -m cnc_genai.src.simulation.simulate_new "
                    #    f"--clamping_name {st.session_state['clamping_name']} "
                    #    f"--product_master {product_master_excel_path} "
                    #    f"--parsed_nc_code {parsed_nc_code_excel_path} "
                    #    f"--tools {tools_excel_path} "
                    #    f"--stock_path {stock_path} "
                    #    f"--precision {precision} --r_slack {r_slack} --z_slack {z_slack} "
                    #    f"--origin {' '.join([str(x) for x in matrix_origin_final])} "
                    #    f"--numpy_out_path {numpy_out_path} "
                    #    f"--excel_out_path {excel_out_path} "
                    #    # f"--verbose "
                    #    f"> {log_file} 2>&1 &"
                    # )
                    # st.session_state["cmd"] = cmd

                    # process = subprocess.Popen(
                    #    cmd,
                    #    shell=True,
                    #    text=True,
                    #    stdout=subprocess.PIPE,
                    #    stderr=subprocess.PIPE,
                    # )

                    # 基本命令參數
                    base_cmd = [
                        "python",
                        "-u",
                        "-m",
                        "cnc_genai.src.simulation.main",
                        "--clamping_name",
                        st.session_state["clamping_name"],
                        "--product_master",
                        product_master_excel_path,
                        "--parsed_nc_code",
                        parsed_nc_code_excel_path,
                        "--tools",
                        tools_excel_path,
                        "--stock_path",
                        stock_path,
                        "--precision",
                        str(precision),
                        "--linear_deflection",
                        str(linear_deflection),
                        "--angular_deflection",
                        str(angular_deflection),
                        "--resolution",
                        str(resolution),
                        "--r_slack",
                        str(r_slack),
                        "--z_slack",
                        str(z_slack),
                        "--origin",
                        *[str(x) for x in origin],
                        "--numpy_out_path",
                        numpy_out_path,
                        "--excel_out_path",
                        excel_out_path,
                        # "--binary",
                        # str(st.session_state["use_binary_image"]),
                    ]

                    # 檢查環境變數並加入 --use-gpu 參數
                    if os.getenv("SIM_USE_GPU"):
                        base_cmd.append("--use-gpu")

                    st.session_state["cmd"] = " ".join(base_cmd)

                    # 跨平台背景執行，處理 UTF-8 編碼
                    try:
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

                    # 記錄進程ID到JSON文件中
                    if st.session_state.get("process_pid"):
                        try:
                            process_info_file = os.path.join(
                                excel_out_path, f"process_info.json"
                            )
                            process_info = {
                                "process_id": st.session_state["process_pid"],
                                "username": st.session_state.username,
                                "precision": st.session_state["precision"],
                                "start_time": datetime.datetime.now().strftime(
                                    "%Y/%m/%d %H:%M:%S"
                                ),
                                "start_timestamp": datetime.datetime.now().timestamp(),
                                "finish_flag": False,
                                "finish_time": None,
                                "finish_timestamp": None,
                                "cmd": st.session_state["cmd"],
                            }
                            with open(process_info_file, "w", encoding="utf-8") as f:
                                json.dump(process_info, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            st.warning(f"無法記錄進程信息: {str(e)}")

                        # 顯示成功消息
                        st.success(
                            f"仿真任務已發佈，仿真任務耗時最多可達數小時，請返回主頁點擊查看進度，進程ID: {st.session_state['process_pid']}"
                        )

                else:
                    if st.button(
                        "停止仿真", key="stop_simulation", use_container_width=True
                    ):
                        st.session_state["simulating"] = False

            else:
                if not st.session_state.get("allow_simulation", False):
                    st.error("請先定義毛坯圖形，並設定代碼坐標原點")
                if not st.session_state.get("clamping_name"):
                    st.error("請先設定夾位名稱")

            if st.session_state.get("simulating", False):
                with st.expander("點擊展開後台任務詳情"):
                    cmd = st.session_state.get("cmd", "")
                    if cmd:
                        st.write(cmd)
                    else:
                        st.write("仿真命令尚未生成")

            # 確保進程完成
            # process.wait()


def generate_stock(precision=3, **args):
    """
    生成毛坯圖形
    """
    size = np.array(
        [args["manual_length"], args["manual_width"], args["manual_height"]]
    )
    pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

    thickness = int(args["manual_thickness"] * 10 ** (precision - 3))
    edge_top = int(args["manual_thickness_top"] * 10 ** (precision - 3))
    edge_bottom = int(args["manual_thickness_bottom"] * 10 ** (precision - 3))
    edge_left = int(args["manual_thickness_left"] * 10 ** (precision - 3))
    edge_right = int(args["manual_thickness_right"] * 10 ** (precision - 3))

    image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
    image[:] = MATERIAL_COLOR
    image[thickness:-thickness, thickness:-thickness, thickness:-thickness] = (
        EMPTY_COLOR
    )
    image[edge_top:-edge_bottom, edge_left:-edge_right, thickness:] = EMPTY_COLOR

    return image
