import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_image_annotation import detection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import os


def image_annotation(
    product_image,
    origin,
    key_suffix="",
    precision=4,
) -> dict:

    label_list = [
        "振刀紋",
    ]

    # 初始化状态
    bbox_data_key = f"bbox_data_{key_suffix}"
    if bbox_data_key not in st.session_state:
        st.session_state[bbox_data_key] = {}

    # 增加一个radio selection button，三个选项横着摆
    plane_options = ["XY平面", "XZ平面", "YZ平面"]

    # 獲取預設索引值
    if (
        hasattr(st.session_state, "selected_plane")
        and st.session_state.selected_plane in plane_options
    ):
        default_index = plane_options.index(st.session_state.selected_plane)
    else:
        default_index = 0

    selected_plane = st.radio(
        "選擇觀察平面",
        options=plane_options,
        index=default_index,
        horizontal=True,
        key=f"plane_select_{key_suffix}",
        help="選擇要進行標註的平面視圖",
    )
    st.session_state.selected_plane = selected_plane

    # 創建兩個下拉選擇框
    col1, col2 = st.columns(2)

    with col1:
        subprograms = ["所有子程式"] + st.session_state.df_product_master[
            "sub_program"
        ].tolist()
        selected_subprogram = st.selectbox(
            "選擇子程序", options=subprograms, key=f"subprogram_select_{key_suffix}"
        )

    with col2:
        step = float(10 ** (3 - precision))

        if selected_plane == "XY平面":
            # 计算最大值和最小值in mm
            max_value_pixel = float(product_image.shape[2] - origin[2])
            min_value_pixel = float(0.0 - origin[2])
            max_value_mm = float(max_value_pixel / 10 ** (precision - 3))
            min_value_mm = float(min_value_pixel / 10 ** (precision - 3))

            selected_z = st.number_input(
                f"選擇XY平面的Z軸坐標(mm)",
                value=min_value_mm,
                min_value=min_value_mm,
                max_value=max_value_mm,
                step=step,
                help="請選擇Z平面坐標(mm)",
                key=f"z_plane_select_all_{key_suffix}",
            )
            # 將選擇的Z值轉換為索引
            selected_z_pixel = round(selected_z * (10 ** (precision - 3)) + origin[2])
            selected_z_pixel = max(min(selected_z_pixel, product_image.shape[2] - 1), 0)

            # 直接使用product中的已缩放图像
            image = product_image[:, :, selected_z_pixel]
            selected_mm = selected_z
            coord_label = "Z"

        elif selected_plane == "XZ平面":
            # 计算最大值和最小值in mm
            max_value_pixel = float(product_image.shape[0] - origin[1])
            min_value_pixel = float(0.0 - origin[1])
            max_value_mm = float(max_value_pixel / 10 ** (precision - 3))
            min_value_mm = float(min_value_pixel / 10 ** (precision - 3))

            selected_y = st.number_input(
                f"選擇XZ平面的Y軸坐標(mm)",
                value=min_value_mm,
                min_value=min_value_mm,
                max_value=max_value_mm,
                step=step,
                help="請選擇Y平面坐標(mm)",
                key=f"y_plane_select_all_{key_suffix}",
            )
            # 將選擇的Y值轉換為索引
            selected_y_pixel = round(
                product_image.shape[0]
                - selected_y * (10 ** (precision - 3))
                - origin[1]
            )

            # 獲取XZ平面圖像
            selected_y_pixel = max(min(selected_y_pixel, product_image.shape[0] - 1), 0)
            image = product_image[selected_y_pixel, :, :]

            # 注意选择YZ平面时，需要旋转截图
            image = np.rot90(image, k=1)
            image = np.flip(image, axis=1)

            selected_mm = selected_y
            coord_label = "Y"
        else:  # YZ平面
            max_value_pixel = float(product_image.shape[1] - origin[0])
            min_value_pixel = float(0.0 - origin[0])
            max_value_mm = float(max_value_pixel / 10 ** (precision - 3))
            min_value_mm = float(min_value_pixel / 10 ** (precision - 3))

            selected_x = st.number_input(
                f"選擇YZ平面的X軸坐標(mm)",
                value=min_value_mm,
                min_value=min_value_mm,
                max_value=max_value_mm,
                step=step,
                help="請選擇X平面坐標(mm)",
                key=f"x_plane_select_all_{key_suffix}",
            )
            # 將選擇的X值轉換為索引\
            selected_x_pixel = round(selected_x * (10 ** (precision - 3)) + origin[0])

            selected_x_pixel = max(min(selected_x_pixel, product_image.shape[1] - 1), 0)

            # 獲取YZ平面圖像
            image = product_image[:, selected_x_pixel, :]

            # 注意选择XZ平面时，需要旋转截图
            image = np.rot90(image, k=1)

            selected_mm = selected_x
            coord_label = "X"

    # 缩放图片，如果image width > 600，则等比例缩放为600
    # if image.shape[1] > 600:
    scale = 600 / image.shape[1]
    image = numpy_nearest_resize(
        image, (int(image.shape[0] * scale), 600)
    )  # 等比例缩放

    # 如果image width < 600，则填充为600
    # if image.shape[1] < 600:
    #     image = np.pad(image, ((0, 0), (0, 600 - image.shape[1]), (0, 0)), mode="constant", constant_values=0)

    # 如果image height < 600，则填充为600
    target_height = 600
    current_height = image.shape[0]
    padding_top = 0
    padding_bottom = 0

    if current_height < target_height:
        print("[INFO] 圖像高度小於目標高度，進行填充")
        total_padding = target_height - current_height
        padding_top = total_padding // 2
        padding_bottom = total_padding - padding_top

        # 根據圖像維度設定填充參數
        if image.ndim == 2:
            # 2D 圖像 (height, width)
            pad_width = ((padding_top, padding_bottom), (0, 0))
        elif image.ndim == 3:
            # 3D 圖像 (height, width, channels)
            pad_width = ((padding_top, padding_bottom), (0, 0), (0, 0))
        else:
            raise ValueError(f"不支援的圖像維度: {image.ndim}")

        # 對稱填充圖像（上下填充）
        image = np.pad(image, pad_width, mode="constant", constant_values=0)
        print("[INFO] 填充后的图像尺寸", image.shape)

    # 將 numpy array 轉換為 PIL Image 並保存
    pil_image = Image.fromarray(image)
    os.makedirs("./cnc_genai/demo_ui/temp_images/", exist_ok=True)
    target_image_path = f"./cnc_genai/demo_ui/temp_images/{st.session_state.selected_clamping}-{selected_plane}-{selected_mm}.png"
    # if not os.path.exists(target_image_path):
    pil_image.save(target_image_path)

    # 使用平面和坐標作為key
    plane_coord_key = f"{coord_label}={selected_mm}"

    if selected_subprogram not in st.session_state[bbox_data_key]:
        st.session_state[bbox_data_key][selected_subprogram] = {
            "Z": {},
            "Y": {},
            "X": {},
        }

    if selected_mm in st.session_state[bbox_data_key][selected_subprogram][coord_label]:
        current_bboxes = st.session_state[bbox_data_key][selected_subprogram][
            coord_label
        ][selected_mm]["bboxes"]
    else:
        st.session_state[bbox_data_key][selected_subprogram][coord_label][
            selected_mm
        ] = {
            "bboxes": [],
            "labels": [],
        }
        current_bboxes = []

    # 將存儲的坐標轉換為填充後的坐標系統
    if current_bboxes is not None:
        if selected_plane == "XY平面":
            padded_bboxes = [
                [
                    bbox["x1"] * scale,  # x
                    bbox["y1"] * scale + padding_top,  # y 加上上邊填充
                    (bbox["x2"] - bbox["x1"]) * scale,  # width
                    (bbox["y2"] - bbox["y1"]) * scale,  # height
                ]
                for bbox in current_bboxes
            ]
        elif selected_plane == "XZ平面":
            # XZ平面：旋转后 横轴=X轴，纵轴=Z轴(翻转)
            # 原始3D坐标(x1,y,z1)到(x2,y,z2) -> 显示坐标(x1, Z_max-z1)到(x2, Z_max-z2)
            padded_bboxes = [
                [
                    bbox["x1"] * scale,  # 横轴 = X坐标
                    (product_image.shape[2] - 1 - bbox["z2"]) * scale
                    + padding_top,  # 纵轴 = 翻转的Z坐标
                    (bbox["x2"] - bbox["x1"]) * scale,  # width
                    (bbox["z2"] - bbox["z1"]) * scale,  # height
                ]
                for bbox in current_bboxes
            ]
        elif selected_plane == "YZ平面":
            # YZ平面：旋转+翻转后 横轴=翻转的Y轴，纵轴=翻转的Z轴
            # 原始3D坐标(x,y1,z1)到(x,y2,z2) -> 显示坐标(Y_max-y2, Z_max-z2)到(Y_max-y1, Z_max-z1)
            padded_bboxes = [
                [
                    (product_image.shape[0] - 1 - bbox["y2"])
                    * scale,  # 横轴 = 翻转的Y坐标
                    (product_image.shape[2] - 1 - bbox["z2"]) * scale
                    + padding_top,  # 纵轴 = 翻转的Z坐标
                    (bbox["y2"] - bbox["y1"]) * scale,  # width
                    (bbox["z2"] - bbox["z1"]) * scale,  # height
                ]
                for bbox in current_bboxes
            ]
        else:
            raise ValueError(f"Invalid selected_plane: {selected_plane}")
    else:
        padded_bboxes = []

    # 标注
    new_labels = detection(
        image_path=target_image_path,
        bboxes=padded_bboxes,
        labels=st.session_state[bbox_data_key][selected_subprogram][coord_label][
            selected_mm
        ]["labels"],
        label_list=label_list,
        # use_space=True,
        height=500,
        key=f"st_keys.{key_suffix}_{selected_subprogram}_{plane_coord_key}",
    )

    # 更新标注
    if new_labels is not None:
        if selected_plane == "XY平面":
            st.session_state[bbox_data_key][selected_subprogram][coord_label][
                selected_mm
            ]["bboxes"] = [
                {
                    "x1": float(
                        min(max(obj["bbox"][0] / scale, 0), product_image.shape[1] - 1)
                    ),
                    "y1": float(
                        min(
                            max((obj["bbox"][1] - padding_top) / scale, 0),
                            product_image.shape[0] - 1,
                        )
                    ),
                    "x2": float(
                        min(
                            max((obj["bbox"][0] + obj["bbox"][2]) / scale, 0),
                            product_image.shape[1] - 1,
                        )
                    ),
                    "y2": float(
                        min(
                            max(
                                (obj["bbox"][1] + obj["bbox"][3] - padding_top) / scale,
                                0,
                            ),
                            product_image.shape[0] - 1,
                        )
                    ),
                }
                for obj in new_labels
            ]
            for obj in st.session_state[bbox_data_key][selected_subprogram][
                coord_label
            ][selected_mm]["bboxes"]:
                assert obj["x1"] <= obj["x2"]
                assert obj["y1"] <= obj["y2"]
        elif selected_plane == "XZ平面":
            # 注意Z轴需要翻转
            st.session_state[bbox_data_key][selected_subprogram][coord_label][
                selected_mm
            ]["bboxes"] = [
                {
                    "x1": float(
                        min(max(obj["bbox"][0] / scale, 0), product_image.shape[1] - 1)
                    ),
                    "z1": product_image.shape[2]
                    - 1
                    - float(
                        min(
                            max(
                                (obj["bbox"][1] + obj["bbox"][3] - padding_top) / scale,
                                0,
                            ),
                            product_image.shape[2] - 1,
                        )
                    ),  # 減去上邊填充
                    "x2": float(
                        min(
                            max((obj["bbox"][0] + obj["bbox"][2]) / scale, 0),
                            product_image.shape[1] - 1,
                        )
                    ),
                    "z2": product_image.shape[2]
                    - 1
                    - float(
                        min(
                            max((obj["bbox"][1] - padding_top) / scale, 0),
                            product_image.shape[2] - 1,
                        )
                    ),  # 減去上邊填充
                }
                for obj in new_labels
            ]
            for obj in st.session_state[bbox_data_key][selected_subprogram][
                coord_label
            ][selected_mm]["bboxes"]:
                assert obj["x1"] <= obj["x2"]
                assert obj["z1"] <= obj["z2"]
        elif selected_plane == "YZ平面":
            # 注意YZ轴都需要翻转
            st.session_state[bbox_data_key][selected_subprogram][coord_label][
                selected_mm
            ]["bboxes"] = [
                {
                    "y1": product_image.shape[0]
                    - 1
                    - float(
                        min(
                            max((obj["bbox"][0] + obj["bbox"][2]) / scale, 0),
                            product_image.shape[0] - 1,
                        )
                    ),
                    "z1": product_image.shape[2]
                    - 1
                    - float(
                        min(
                            max(
                                (obj["bbox"][1] + obj["bbox"][3] - padding_top) / scale,
                                0,
                            ),
                            product_image.shape[2] - 1,
                        )
                    ),  # 減去上邊填充
                    "y2": product_image.shape[0]
                    - 1
                    - float(
                        min(
                            max((obj["bbox"][0]) / scale, 0), product_image.shape[0] - 1
                        )
                    ),
                    "z2": product_image.shape[2]
                    - 1
                    - float(
                        min(
                            max((obj["bbox"][1] - padding_top) / scale, 0),
                            product_image.shape[2] - 1,
                        )
                    ),  # 減去上邊填充
                }
                for obj in new_labels
            ]
            for obj in st.session_state[bbox_data_key][selected_subprogram][
                coord_label
            ][selected_mm]["bboxes"]:
                assert obj["y1"] <= obj["y2"]
                assert obj["z1"] <= obj["z2"]
        st.session_state[bbox_data_key][selected_subprogram][coord_label][selected_mm][
            "labels"
        ] = [v["label_id"] for v in new_labels]

    # 顯示當前標註的缺陷位置
    st.markdown("### 當前標註缺陷位置:")
    for sp, plane_data in st.session_state[bbox_data_key].items():
        for coord_label, coord_defect_data in plane_data.items():
            for coord_pixel, defect_data in coord_defect_data.items():
                if len(defect_data["bboxes"]) > 0:
                    with st.expander(
                        f"{sp} | {coord_label}={coord_pixel}mm平面 | 出现{label_list[defect_data['labels'][0]]}等{len(defect_data['labels'])}個缺陷"
                    ):
                        st.write(defect_data["bboxes"])

    # clean st.session_state[bbox_data_key] dict
    # 清理沒有bbox的空數據項
    for subprogram in list(st.session_state[bbox_data_key].keys()):
        for coord_type in list(st.session_state[bbox_data_key][subprogram].keys()):
            # 清理空的坐標值
            coords_to_remove = []
            for coord_value in st.session_state[bbox_data_key][subprogram][coord_type]:
                if (
                    len(
                        st.session_state[bbox_data_key][subprogram][coord_type][
                            coord_value
                        ]["bboxes"]
                    )
                    == 0
                    and len(
                        st.session_state[bbox_data_key][subprogram][coord_type][
                            coord_value
                        ]["labels"]
                    )
                    == 0
                ):
                    coords_to_remove.append(coord_value)

            # 刪除空的坐標值
            for coord_value in coords_to_remove:
                del st.session_state[bbox_data_key][subprogram][coord_type][coord_value]

        # 如果整個子程序都空了，也刪除
        if len(st.session_state[bbox_data_key][subprogram]) == 0:
            del st.session_state[bbox_data_key][subprogram]

    # # Create canvas with PIL image
    # canvas_result = st_canvas(
    #     fill_color="rgba(255, 255, 255, 0.5)",  # Disable fill
    #     stroke_width=stroke_width,
    #     stroke_color=f"rgb({stroke_color[0]},{stroke_color[1]},{stroke_color[2]})",  # Convert to RGB
    #     background_image=pil_image,  # Use PIL Image directly
    #     update_streamlit=True,
    #     height=product_image.shape[0],
    #     width=product_image.shape[1],
    #     drawing_mode="rect",  # Only allow rectangle drawing
    #     key=f"st_keys.{selected_z_pixel}-{key_suffix}",  # Unique key for each canvas
    #     initial_drawing=initial_drawing,  # 添加初始繪圖數據
    # )

    # 处理 BBox 输出
    # if canvas_result.json_data is not None:
    #     if canvas_result.json_data != []:
    #         objects = canvas_result.json_data["objects"]
    #         if len(objects):
    #             if selected_subprogram not in st.session_state[bbox_data_key]:
    #                 st.session_state[bbox_data_key][selected_subprogram] = {}
    #             st.session_state[bbox_data_key][selected_subprogram][selected_z] = [
    #                 {
    #                     "x1": obj["left"] / scale,
    #                     "y1": obj["top"] / scale,
    #                     "x2": (obj["left"] + obj["width"]) / scale,
    #                     "y2": (obj["top"] + obj["height"]) / scale,
    #                 }
    #                 for obj in objects
    #                 if obj["type"] == "rect"
    #             ]
    #             st.write(
    #                 "當前標註缺陷位置數量：\n",
    #                 "\n".join(
    #                     [
    #                         f"{sp} - {z}mm: {len(v[z])}"
    #                         for sp, v in st.session_state[bbox_data_key].items()
    #                         for z in v
    #                     ]
    #                 ),
    #             )

    return st.session_state[bbox_data_key]


def numpy_nearest_resize(image, new_shape):
    """
    純numpy實現的最近鄰插值縮放

    參數:
    image: numpy array (H, W)或(H, W, C)
    new_shape: 目標形狀元組 (height, width)

    返回:
    resized_image: 縮放後的numpy array
    """
    height, width = new_shape
    h_ratio = image.shape[0] / height
    w_ratio = image.shape[1] / width

    y_indices = (np.floor(np.arange(height) * h_ratio)).astype(int)
    x_indices = (np.floor(np.arange(width) * w_ratio)).astype(int)

    if image.ndim == 3:
        return image[y_indices[:, None], x_indices].astype(np.uint8)
    else:
        return image[y_indices[:, None], x_indices].astype(np.uint8)


def visualize_code_area(product_image, scale, bbox_coords, selected_code=None):
    """
    在图像上可视化特定代码段对应的区域，并绘制移动路径
    """
    # 坐标映射
    x_min = int(bbox_coords["x_coords"][0] * scale)
    x_max = int(bbox_coords["x_coords"][1] * scale)
    y_min = int(bbox_coords["y_coords"][0] * scale)
    y_max = int(bbox_coords["y_coords"][1] * scale)

    # 直接从session_state获取数据
    z_index = int(round(bbox_coords["z_coords"]))

    # #################
    # # 确保所有坐标都是整数
    # x_min = int(round(bbox_coords['x_coords'][0]))
    # x_max = int(round(bbox_coords['x_coords'][1]))
    # y_min = int(round(bbox_coords['y_coords'][0]))
    # y_max = int(round(bbox_coords['y_coords'][1]))

    # zst_path = '/Users/liyang/Library/CloudStorage/OneDrive-波士顿咨询公司/FXN/CNC/im_setpoint_genai/model/mass_production/data/cnc_data/simulation/5519_shape=3763_5486_125_3.zst'
    # product, aaa = load_from_zst(zst_path)
    # #################

    # 创建图形 - 增加figsize并设置较大的边距
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    # 確保字體文件路徑正確
    # my_font = fm.FontProperties(fname="/Users/liyang/Library/CloudStorage/OneDrive-波士顿咨询公司/FXN/CNC/im_setpoint_genai/model/agent/langchain/fonts/SimHei.ttf")
    my_font = fm.FontProperties(fname="cnc_v0/model/agent/langchain/fonts/SimHei.ttf")
    print(os.getcwd())

    # 获取Z平面图像，确保z_index在范围内
    z_index = max(0, min(z_index, product_image.shape[2] - 1))
    image = product_image[:, :, z_index]

    # 显示图像
    ax.imshow(image)

    # 去掉坐标轴的刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])

    # BBOX# BBOX# BBOX# BBOX# BBOX# BBOX
    # 計算寬度和高度
    width = x_max - x_min + 4
    height = y_max - y_min + 4

    # BBOX
    linewidth = 8
    xy, w, h = (x_min - 2, y_min - 2), width, height
    r = patches.Rectangle(xy, w, h, fc="none", ec="gold", lw=3)

    offsetbox = AuxTransformBox(ax.transData)
    offsetbox.add_artist(r)
    ab = AnnotationBbox(
        offsetbox,
        (xy[0] + w / 2.0, xy[1] + h / 2.0),
        boxcoords="data",
        pad=0.52,
        fontsize=linewidth,
        bboxprops=dict(facecolor="none", edgecolor="r", lw=linewidth),
    )
    ax.add_artist(ab)

    # 繪製移動路徑（如果提供了selected_code）
    if selected_code is not None and not selected_code.empty:
        # 按照行號排序，確保路徑順序正確
        if "row_id" in selected_code.columns:
            selected_code = selected_code.sort_values("row_id")

        # 篩選出包含X和Y坐標的行
        path_rows = selected_code.dropna(subset=["X_pixel", "Y_pixel"])

        if not path_rows.empty:
            # 獲取所有X和Y坐標
            x_coords = (path_rows["X_pixel"] * scale).values
            y_coords = (path_rows["Y_pixel"] * scale).values
            move_codes = (
                path_rows["move_code"].values
                if "move_code" in path_rows.columns
                else [""] * len(x_coords)
            )

            # 繪製路徑 - 修正邏輯：每一行的move_code決定從上一點到當前點的線段樣式
            for i in range(len(x_coords)):
                # 在每個點添加一個小圓點
                ax.scatter(
                    x_coords[i],
                    y_coords[i],
                    color="black",  # 所有點都用黑色標記
                    s=10,  # 點的大小
                    zorder=5,  # 確保點在線的上面
                )

                # 從第二個點開始繪製線段（連接到前一個點）
                if i > 0:
                    # 根據當前點的move_code選擇顏色和線型
                    if move_codes[i] == "G00":
                        color = "green"  # G00快速定位用綠色
                        linestyle = "--"  # 虛線
                        linewidth = 1
                    elif move_codes[i] in ["G01", "G02", "G03"]:
                        color = "red"  # G01/G02/G03切削進給用紅色
                        linestyle = "-"  # 實線
                        linewidth = 2
                    else:
                        color = "blue"  # 其他代碼用藍色
                        linestyle = "-."  # 點劃線
                        linewidth = 1

                    # 繪製從上一點到當前點的線段
                    ax.plot(
                        [x_coords[i - 1], x_coords[i]],
                        [y_coords[i - 1], y_coords[i]],
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=0.8,
                    )

            # 創建不帶標籤的圖例項
            legend_elements = [
                Line2D([0], [0], color="green", linestyle="--", lw=2),
                Line2D([0], [0], color="red", linestyle="-", lw=3),
                Line2D([0], [0], color="blue", linestyle="-.", lw=2),
            ]

            # 使用中文標籤創建圖例
            labels = ["G00 快速定位", "G01/G02/G03 切削進給", "其他移動"]
            ax.legend(legend_elements, labels, loc="upper right", prop=my_font)

    # 显示图像
    st.pyplot(fig)
