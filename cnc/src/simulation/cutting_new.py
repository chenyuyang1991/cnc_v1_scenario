import math
import numpy as np

# import cupynumeric as np
import cv2
from datetime import datetime
from viztracer import log_sparse

from cnc_genai.src.simulation.colors import CUTTING_MASK_COLOR
from cnc_genai.src.simulation.rotate_coordinates import (
    multiple_rotated_physical_to_pixel,
)
from cnc_genai.src.simulation.utils import physical_to_pixel
from cnc_genai.src.simulation.utils import get_smart_tracer


def _get_rotated_basis(
    img,
    origin,
    rotating_centers,
    rotating_angles,
    rotating_axes,
    precision,
    verbose=False,
):
    """基於工件坐標系原點origin獲取旋轉後的基向量"""
    # 定義標準基向量
    base_vectors = np.eye(3, dtype=np.float32)

    # 計算旋轉後的基向量
    rotated_basis = []
    for i in range(3):
        # 將基向量從物理坐標轉換到像素坐標
        physical_point, pixel_point = multiple_rotated_physical_to_pixel(
            img,
            point_physical=base_vectors[i],  # 直接使用標準基向量
            rotating_centers=rotating_centers,
            angles=rotating_angles,
            axes=rotating_axes,
            origin=origin,
            precision=precision,
        )
        vec = physical_point / np.linalg.norm(physical_point)
        vec[1] = -vec[1]  # Y軸反轉，将右手坐标系转换为左手坐标系
        rotated_basis.append(vec)
    rotated_basis[1] = -rotated_basis[1]

    # 正交性檢查與修正
    basis = np.array(rotated_basis)
    # 檢查正交性
    dot_xy = np.dot(basis[0], basis[1])
    dot_xz = np.dot(basis[0], basis[2])
    dot_yz = np.dot(basis[1], basis[2])

    if verbose:
        for i in range(3):
            print(
                f"旋轉後刀具坐標單位向量{'XYZ'[i]}: {rotated_basis[i]} | 旋轉後的基向量 in 工件坐標系"
            )
        print(f"rotated_basis: {rotated_basis}")
        print(f"正交性檢查 - XY: {dot_xy:.6f}, XZ: {dot_xz:.6f}, YZ: {dot_yz:.6f}")
        print("-" * 20)
    assert dot_xy < 1e-6 and dot_xz < 1e-6 and dot_yz < 1e-6, "刀具坐標系基向量不正交"

    return rotated_basis


@log_sparse(stack_depth=4)
def draw_G01_cv(img, row, start_point, end_point, tool_r, tool_h, **kwargs):
    """
    計算G01代碼劃過工件的像素點，並設置為CUTTING_MASK_COLOR。

    参数:
    img: numpy.ndarray
        工件img，三维图像数据，像素坐標。
    start_point: tuple
        直线的起始点坐标 (x1, y1, z1)，物理坐標(mm)。
    end_point: tuple
        直线的终止点坐标 (x2, y2, z2)，物理坐標(mm)。
    tool_r: float
        刀具的刀頭半徑，物理坐標(mm)。
    tool_h: float
        刀具的刀頭高度，物理坐標(mm)。

    返回:
    mask: numpy.ndarray
        工件img的局部，三维图像数据，被切割的区域局部，像素坐標。
    mask_range: tuple
        工件img的mask的坐标范围(min_0, max_0, min_1, max_1, min_2, max_2)，像素坐標。
    """
    assert len(img.shape) == 3, "mask should be 3D ([H,W,D])"
    logger = kwargs.get("logger", None)

    def _log(*args):
        message = " ".join(map(str, args))
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    # 取回物理坐標係旋轉參數
    rotating_angles = kwargs.get(
        "rotating_angles", []
    )  # 旋轉角度，使用标准物理坐标系定义，右手坐标系，正方向顺时针为正
    rotating_centers = kwargs.get("rotating_centers", [])  # 旋轉中心
    rotating_axes = kwargs.get("rotating_axes", [])  # 旋轉軸

    # 取回參數：是否加速螺旋下刀
    turbo_spiral = kwargs.get("turbo_spiral", False)

    # 獲取工件img的尺寸
    pixel_size = np.array([img.shape[1], img.shape[0], img.shape[2]]).astype(int)

    # 獲取物理坐標原點在工件img坐標系中的像素坐標
    origin = kwargs.get(
        "origin",
        np.array(
            [
                pixel_size[0] // 2,
                pixel_size[1] // 2,
                0,
            ]
        ).astype(int),
    )
    precision = kwargs.get("precision", 4)
    verbose = kwargs.get("verbose", False)

    if verbose:
        _log(f"start_point: {start_point} | 起點 in 物理坐標系")
        _log(f"end_point: {end_point} | 終點 in 物理坐標系")
        _log(f"origin: {origin} | 旋轉前物理坐標原點 in 工件坐標系")
        _log(f"rotating_centers: {rotating_centers} | 旋轉中心 in 物理坐標系")
        _log(f"angles: {rotating_angles} | 旋轉角度")
        _log(f"axes: {rotating_axes} | 旋轉軸")
        _log(f"pixel_size: {pixel_size} | 工件尺寸 in 工件坐標系")
        _log("-" * 20)

    # 將起點和終點的物理坐標係旋轉後的物理坐標轉化為工件img的坐標系中的像素坐標
    start_point_pixel = physical_to_pixel(start_point, origin, pixel_size, precision)
    end_point_pixel = physical_to_pixel(end_point, origin, pixel_size, precision)

    if len(rotating_angles) > 0:
        _, start_point_rotated_pixel = multiple_rotated_physical_to_pixel(
            img,
            start_point,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            origin,
            precision,
        )
        _, end_point_rotated_pixel = multiple_rotated_physical_to_pixel(
            img,
            end_point,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            origin,
            precision,
        )
    else:
        start_point_rotated_pixel = physical_to_pixel(
            start_point, origin, pixel_size, precision
        )
        end_point_rotated_pixel = physical_to_pixel(
            end_point, origin, pixel_size, precision
        )
    row["X_prev_pixel"], row["Y_prev_pixel"], row["Z_prev_pixel"] = (
        start_point_rotated_pixel
    )
    row["X_pixel"], row["Y_pixel"], row["Z_pixel"] = end_point_rotated_pixel
    if verbose:
        _log(
            f"start_point_pixel: {start_point} -> {start_point_pixel} | 起點的像素坐標 in 工件坐標系"
        )
        _log(
            f"end_point_pixel: {end_point} -> {end_point_pixel} | 終點的像素坐標 in 工件坐標系"
        )
        _log(
            f"start_point_rotated_pixel: {start_point} -> {start_point_rotated_pixel} | 起點旋轉後的像素坐標 in 工件坐標系"
        )
        _log(
            f"end_point_rotated_pixel: {end_point} -> {end_point_rotated_pixel} | 終點旋轉後的像素坐標 in 工件坐標系"
        )
        _log("-" * 20)

    if not turbo_spiral:
        if np.array_equal(start_point_rotated_pixel, end_point_rotated_pixel):
            _log(
                f"【起止點旋轉後的像素坐標相同，不仿真】start_pixel={start_point_rotated_pixel}, end_pixel={end_point_rotated_pixel}"
            )
            row["same_start_end_pixel"] = True
            return None, None, row, "identical_start_end_pixel"

    if turbo_spiral:
        # 檢查起止點在像素坐標系下是否相近（XYZ的差均小於0.1mm）
        pixel_diff = np.abs(start_point - end_point)
        if np.all(pixel_diff < (0.1 * 10 ** (precision - 3))):
            _log(
                f"【物理坐標相近（XYZ的差均小於0.1mm），不仿真】, start_point={start_point}, end_point={end_point}"
            )
            row["same_start_end_pixel"] = True
            return None, None, row, "start_end_coordinate_too_close"

    # 修正後的單位轉換
    tool_r_pixel = int(round(tool_r * 10 ** (precision - 3)))
    tool_h_pixel = int(round(tool_h * 10 ** (precision - 3)))

    # 確保 tool_r_pixel 和 tool_h_pixel 至少為 1，以滿足 OpenCV 函數的要求
    if tool_r_pixel <= 0:
        print(f"[ERROR] tool_r_pixel <= 0: {tool_r_pixel}")
        tool_r_pixel = 1
    if tool_h_pixel <= 0:
        print(f"[ERROR] tool_h_pixel <= 0: {tool_h_pixel}")
        tool_h_pixel = 1

    # 計算刀具路徑向量（像素坐標系）
    path_vector = end_point_rotated_pixel - start_point_rotated_pixel
    path_length = np.linalg.norm(path_vector)
    if path_length == 0:
        return None, None, row, "zero_length_toolpath"

    # 獲取旋轉後的坐標系基向量
    x_axis, y_axis, z_axis = _get_rotated_basis(
        img,
        origin,
        rotating_centers,
        rotating_angles,
        rotating_axes,
        precision,
        verbose,
    )

    # 構造旋轉矩陣 (從局部坐標到世界坐標)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)

    # 如果原刀具路徑向量在Z軸方向上，則生成圓柱體
    if (
        start_point_pixel[0] == end_point_pixel[0]
        and start_point_pixel[1] == end_point_pixel[1]
    ):

        with get_smart_tracer().log_event("Z.initial"):
            # DrawG01 case
            casename = "Zcircle"
            # 創建3D畫布
            mask_height = int(path_length) + tool_h_pixel
            canvas_mask = np.zeros(
                (2 * tool_r_pixel + 1, 2 * tool_r_pixel + 1, mask_height),
                dtype=np.uint8,
            )

            # 坐標轉換，定義刀具坐標係原點
            tool_origin = np.array([tool_r_pixel, tool_r_pixel, 0]).astype(int)
            # 刀具坐標系原點在工件img坐標系中的像素坐標
            base_offset = (
                start_point_rotated_pixel
                if start_point_rotated_pixel[2] < end_point_rotated_pixel[2]
                else end_point_rotated_pixel
            )

        # 生成圓柱體刀具路徑，直徑為刀具直徑，高度為刀具高度+行進距離
        # op1: 每一層都用opencv
        # for z in range(mask_height):
        #     canvas_mask[:, :, z] = cv2.circle(canvas_mask[:, :, z].astype(np.uint8), (tool_r_pixel, tool_r_pixel), tool_r_pixel, 1, -1)
        # op2: 僅最下一層用opencv
        with get_smart_tracer().log_event("Z.cv2.circle"):
            # canvas_mask[:, :, 0] = cv2.circle(
            #    canvas_mask[:, :, 0].astype(np.uint8),
            #    (tool_r_pixel, tool_r_pixel),
            #    tool_r_pixel,
            #    1,
            #    -1,
            # )

            temp_canvas = np.zeros(
                (2 * tool_r_pixel + 1, 2 * tool_r_pixel + 1),
                dtype=np.uint8,
            )

            cv2.circle(
                temp_canvas,
                (tool_r_pixel, tool_r_pixel),
                tool_r_pixel,
                1,
                -1,
            )
            # 將結果拷貝回原陣列
            canvas_mask[:, :, 0] = temp_canvas

        with get_smart_tracer().log_event("extend Z.cv2.circle"):
            # for z in range(1, mask_height):
            #    xy_indices = np.array(np.where(canvas_mask[:, :, 0] == 1)).T
            #    for x, y in xy_indices:
            #        canvas_mask[x, y, z] = 1

            # 使用 numpy boardcast 特性直接將第一層複製到所有其他層
            canvas_mask[:, :, 1:mask_height] = canvas_mask[:, :, 0:1]

    # 如果原刀具路徑向量在XY平面上，則生成的直線
    elif start_point_pixel[2] == end_point_pixel[2]:

        with get_smart_tracer().log_event("XY.initial"):
            # DrawG01 case
            casename = "XYline"

            # 計算未旋轉像素坐標系中的路徑向量和長度
            path_vector_pixel = end_point_pixel - start_point_pixel
            path_length = np.linalg.norm(path_vector_pixel[:2])  # XY平面長度

            # 創建3D畫布(包含XY最大範圍和Z軸刀具高度)
            canvas_size = 2 * (int(path_length) + tool_r_pixel) + 1
            canvas_mask = np.zeros(
                (canvas_size, canvas_size, tool_h_pixel), dtype=np.uint8
            )

            # 定義刀具坐標系原點(畫布中心)
            tool_origin = np.array([canvas_size // 2, canvas_size // 2, 0], dtype=int)

            # 刀具坐標系原點在工件坐標系中的位置(取旋轉後的起點位置)
            base_offset = start_point_rotated_pixel.copy()

            # 計算在畫布中的相對位置，注意Y方向需要反轉
            start_canvas = tool_origin[:2]
            end_canvas = np.array(
                [
                    tool_origin[0]
                    + (end_point_pixel - start_point_pixel)[0],  # X方向不變
                    tool_origin[1]
                    + (end_point_pixel - start_point_pixel)[1],  # Y方向反轉
                ]
            )

        with get_smart_tracer().log_event("XY.cv2.line"):
            # 在畫布底層繪製直線(考慮刀具半徑)
            # canvas_mask[:, :, 0] = cv2.line(
            #    canvas_mask[:, :, 0].astype(np.uint8),
            #    tuple(start_canvas.astype(int)),  # 使用畫布中心作為起點
            #    tuple(end_canvas.astype(int)),  # 計算終點相對位置
            #    1,
            #    thickness=2 * tool_r_pixel,  # 線寬為刀具直徑
            # )

            temp_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
            cv2.line(
                temp_canvas,
                tuple(start_canvas.astype(int)),  # 使用畫布中心作為起點
                tuple(end_canvas.astype(int)),  # 計算終點相對位置
                1,
                thickness=2 * tool_r_pixel,  # 線寬為刀具直徑
            )
            # 將結果拷貝回原陣列
            canvas_mask[:, :, 0] = temp_canvas

        with get_smart_tracer().log_event("extend XY.cv2.line"):
            # 沿Z軸延伸刀具高度
            # for z in range(1, tool_h_pixel):
            #    canvas_mask[:, :, z] = canvas_mask[:, :, 0]

            # 使用 numpy boardcast 特性沿Z軸延伸刀具高度
            canvas_mask[:, :, 1:tool_h_pixel] = canvas_mask[:, :, 0:1]

    # 螺旋下刀，如果原始位置Z不同且XY至少有一個不同，則為螺旋下刀，生成一系列圓柱體
    elif start_point_pixel[2] != end_point_pixel[2] and not (
        start_point_pixel[0] == end_point_pixel[0]
        and start_point_pixel[1] == end_point_pixel[1]
    ):
        with get_smart_tracer().log_event("XYZ.initial"):
            # DrawG01 case
            casename = "XYZcircle"
            # 計算Z軸變化量
            z_diff = end_point_pixel[2] - start_point_pixel[2]
            z_steps = abs(z_diff) + 1  # 包含起終點

            # 計算未旋轉像素坐標系中的路徑向量和長度
            path_vector_pixel = end_point_pixel - start_point_pixel
            path_length = np.linalg.norm(path_vector_pixel[:2])  # 只考慮XY平面的長度

            # 創建3D畫布(包含XY最大範圍和Z軸刀具高度)
            canvas_size = 2 * (int(path_length) + tool_r_pixel) + 1
            canvas_mask = np.zeros(
                (canvas_size, canvas_size, abs(z_diff) + tool_h_pixel), dtype=np.uint8
            )

            # 定義刀具坐標系原點(畫布中心)
            tool_origin = np.array([canvas_size // 2, canvas_size // 2, 0], dtype=int)

            # 刀具坐標系原點在工件坐標系中的位置(取旋轉後的起點位置)
            base_offset = np.array(
                [
                    start_point_rotated_pixel[0],
                    start_point_rotated_pixel[1],
                    min(start_point_rotated_pixel[2], end_point_rotated_pixel[2]),
                ],
                dtype=int,
            )

            # 找到Z更小的点作为base
            bottom_point_pixel = (
                start_point_pixel
                if start_point_pixel[2] < end_point_pixel[2]
                else end_point_pixel
            )
            top_point_pixel = (
                start_point_pixel
                if start_point_pixel[2] > end_point_pixel[2]
                else end_point_pixel
            )

        temp_canvas = np.zeros(
            (canvas_size, canvas_size),
            dtype=np.uint8,
        )

        # 生成螺旋下刀軌跡
        for z_step in range(z_steps):

            with get_smart_tracer().log_event("XYZ.cv2.initial"):
                # 通过线性插值计算z_step平面上的XY位置
                x_step = (
                    (top_point_pixel[0] - bottom_point_pixel[0])
                    * z_step
                    / (z_steps - 1)
                )
                y_step = (
                    (top_point_pixel[1] - bottom_point_pixel[1])
                    * z_step
                    / (z_steps - 1)
                )
                current_canvas = np.array(
                    [
                        tool_origin[0]
                        + bottom_point_pixel[0]
                        - start_point_pixel[0]
                        + x_step,
                        tool_origin[1]
                        + bottom_point_pixel[1]
                        - start_point_pixel[1]
                        + y_step,
                    ]
                )

                # reset temp_canvas
                temp_canvas.fill(0)

            # 在當前Z平面繪製圓形
            # print('tool_origin', tool_origin)
            # print('top_point_pixel', top_point_pixel)
            # print('bottom_point_pixel', bottom_point_pixel)
            # print('start_point_pixel', start_point_pixel)
            # print('x_step', x_step)
            # print('current_canvas', current_canvas)
            with get_smart_tracer().log_event("XYZ.cv2.circle"):
                # canvas_mask[:, :, z_step] = cv2.circle(
                #    canvas_mask[:, :, z_step].astype(np.uint8),
                #    tuple(current_canvas.astype(int)),
                #    tool_r_pixel,
                #    1,
                #    -1,
                # )
                current_center = tuple(current_canvas.astype(int))
                cv2.circle(
                    temp_canvas,
                    current_center,
                    tool_r_pixel,
                    1,
                    -1,
                )

                with get_smart_tracer().log_event("XYZ.cv2.circle -> find range of xy"):
                    # 找出 xy 的範圍, 直接從圓心和半徑計算範圍
                    x_min = max(current_center[0] - tool_r_pixel, 0)
                    x_max = min(
                        current_center[0] + tool_r_pixel + 1, temp_canvas.shape[1]
                    )
                    y_min = max(current_center[1] - tool_r_pixel, 0)
                    y_max = min(
                        current_center[1] + tool_r_pixel + 1, temp_canvas.shape[0]
                    )

            with get_smart_tracer().log_event("extend XYZ.cv2.circle"):
                # 沿Z軸延伸刀具高度
                # xy_indices = np.array(np.where(canvas_mask[:, :, z_step] == 1)).T
                # for x, y in xy_indices:
                #    canvas_mask[
                #        x,
                #        y,
                #        z_step + 1 : min(z_step + tool_h_pixel, canvas_mask.shape[2]),
                #    ] = 1

                with get_smart_tracer().log_event(
                    "extend XYZ.cv2.circle -> |= temp_canvas"
                ):
                    # 沿Z軸延伸刀具高度 (從 z_step 出發) 至毛胚頂部
                    min_of_z_step = canvas_mask.shape[2]  # 必須要切穿當前毛胚頂部

                    # 只在找到的 xy 範圍內執行位元運算
                    target_region = canvas_mask[
                        y_min:y_max, x_min:x_max, z_step:min_of_z_step
                    ]
                    source_region = temp_canvas[y_min:y_max, x_min:x_max, np.newaxis]
                    # 原地進行位元運算
                    np.bitwise_or(target_region, source_region, out=target_region)

    else:
        _log("weird G01 cutting path, not simulated")
        return None, None, row, "weird_toolpath"

    with get_smart_tracer().log_event("drawG01 mask_points"):
        # 獲取切割點坐標(刀具局部坐標系)
        mask_points = np.argwhere(canvas_mask > 0)  # yxz
        mask_points = mask_points[:, [1, 0, 2]]  # xyz
        mask_points = mask_points - tool_origin  # xyz

    # 如果沒有則為沒有切割軌跡
    if mask_points.size == 0:
        _log(f"无切割像素点")
        return None, None, row, "no_cutting"
    else:
        if verbose:
            _log(
                f"X.range: {min(mask_points[:, 0])} -> {max(mask_points[:, 0])} | 切割點 in 刀具坐標系"
            )
            _log(
                f"Y.range: {min(mask_points[:, 1])} -> {max(mask_points[:, 1])} | 切割點 in 刀具坐標系"
            )
            _log(
                f"Z.range: {min(mask_points[:, 2])} -> {max(mask_points[:, 2])} | 切割點 in 刀具坐標系"
            )

    with get_smart_tracer().log_event("drawG01 transform"):
        # 切割點坐標轉換到工件img坐標系
        # world_points = (
        #    cv2.transform(mask_points.reshape(-1, 1, 3), rotation_matrix).reshape(-1, 3)
        #    + base_offset
        # )
        world_points = (
            np.matmul(mask_points.reshape(-1, 1, 3), rotation_matrix).reshape(-1, 3)
            + base_offset
        )

    with get_smart_tracer().log_event("drawG01 round world_points"):
        # 轉換為整數像素坐標
        pixel_coords = np.round(world_points).astype(int)  # xyz

    if verbose:
        _log(
            f"X.range: {min(pixel_coords[:, 0])} -> {max(pixel_coords[:, 0])} | 切割點 in 工件坐標系"
        )
        _log(
            f"Y.range: {min(pixel_coords[:, 1])} -> {max(pixel_coords[:, 1])} | 切割點 in 工件坐標系"
        )
        _log(
            f"Z.range: {min(pixel_coords[:, 2])} -> {max(pixel_coords[:, 2])} | 切割點 in 工件坐標系"
        )
        _log(f"工件voxel坐標系: {img.shape}")

    with get_smart_tracer().log_event("drawG01 check valid dim"):
        # 檢查每個維度是否在有效範圍內
        x_valid = (pixel_coords[:, 0] >= 0) & (
            pixel_coords[:, 0] < img.shape[1]
        )  # X軸 (2540)
        y_valid = (pixel_coords[:, 1] >= 0) & (
            pixel_coords[:, 1] < img.shape[0]
        )  # Y軸 (1849)
        z_valid = (pixel_coords[:, 2] >= 0) & (
            pixel_coords[:, 2] < img.shape[2]
        )  # Z軸 (80)

        # 組合所有維度的有效性檢查
        valid_mask = x_valid & y_valid & z_valid

        # 篩選出有效的座標點
        valid_coords = pixel_coords[valid_mask]  # xyz

    if verbose:
        _log(f"總座標點數: {len(pixel_coords)}")
        _log(f"有效座標點數: {len(valid_coords)}")

    with get_smart_tracer().log_event("drawG01 operation other"):
        # 設置切割區域顏色
        img_copy = img.copy().astype(np.uint8)
        img_copy[valid_coords[:, 1], valid_coords[:, 0], valid_coords[:, 2]] = (
            CUTTING_MASK_COLOR  # img[yxz]
        )

        # 使用現有的valid_coords計算mask範圍
        if valid_coords.size == 0:
            _log(f"切割像素点：{canvas_mask.sum()} 均不在工件內")
            return None, None, row, "cutting_outside_workpiece"

        # 修正座標軸索引
        min_x, max_x = valid_coords[:, 0].min(), valid_coords[:, 0].max() + 1  # X軸
        min_y, max_y = valid_coords[:, 1].min(), valid_coords[:, 1].max() + 1  # Y軸
        min_z, max_z = valid_coords[:, 2].min(), valid_coords[:, 2].max() + 1  # Z軸

        # 安全邊界檢查
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        min_z = max(min_z, 0)
        max_x = min(max_x, img_copy.shape[1])
        max_y = min(max_y, img_copy.shape[0])
        max_z = min(max_z, img_copy.shape[2])

        mask = img_copy[min_y:max_y, min_x:max_x, min_z:max_z].copy()
        mask_range = (min_y, max_y, min_x, max_x, min_z, max_z)

    return mask, mask_range, row, casename


@log_sparse(stack_depth=4)
def draw_G02_cv(img, row, start_point, end_point, rel_center, tool_r, tool_h, **kwargs):
    """
    根据给定的起始点、终止点、圆心和阈值，在三维图像中绘制一条顺时针圆弧，并将距离圆弧距离小于阈值的像素设为CUTTING_MASK_COLOR
    """
    mask, mask_range, row, case = draw_curve_cv(
        img,
        row,
        start_point,
        end_point,
        rel_center,
        tool_r,
        tool_h,
        clockwise=True,
        **kwargs,
    )
    return mask, mask_range, row, case if case else "CURVEclockwise"


@log_sparse(stack_depth=4)
def draw_G03_cv(img, row, start_point, end_point, rel_center, tool_r, tool_h, **kwargs):
    """
    根据给定的起始点、终止点、圆心和阈值，在三维图像中绘制一条逆时针圆弧，并将距离圆弧距离小于阈值的像素设为CUTTING_MASK_COLOR
    """

    mask, mask_range, row, case = draw_curve_cv(
        img,
        row,
        start_point,
        end_point,
        rel_center,
        tool_r,
        tool_h,
        clockwise=False,
        **kwargs,
    )
    return mask, mask_range, row, case if case else "CURVENOTclockwise"


def draw_curve_cv(
    img,
    row,
    start_point,
    end_point,
    rel_center,
    tool_r,
    tool_h,
    clockwise=True,
    **kwargs,
):
    """
    根据给定的起始点、终止点、圆心和阈值，在三维图像中绘制一条圆弧，并将距离圆弧距离小于阈值的像素设为CUTTING_MASK_COLOR

    參數:
    img: numpy.ndarray
        工件img，三維圖像數據，像素坐標。
    start_point: tuple
        圓弧的起始點坐標 (x1, y1, z1)，物理坐標(mm)。
    end_point: tuple
        圓弧的終點坐標 (x2, y2, z2)，物理坐標(mm)。
    center: tuple
        圓弧的圓心坐標 (i, j)，物理坐標(mm)。
    tool_r: float
        刀具的刀頭半徑，物理坐標(mm)。
    tool_h: float
        刀具的刀頭高度，物理坐標(mm)。
    clockwise: bool
        是否順時針，預設為True。

    返回:
    mask: numpy.ndarray
        工件img的局部，三维图像数据，被切割的区域局部，像素坐標。
    mask_range: tuple
        工件img的mask的坐标范围(min_0, max_0, min_1, max_1, min_2, max_2)，像素坐標。
    """

    # 取回物理坐標係旋轉參數
    rotating_angles = kwargs.get("rotating_angles", [])  # 旋轉角度
    rotating_centers = kwargs.get("rotating_centers", [])  # 旋轉中心
    rotating_axes = kwargs.get("rotating_axes", [])  # 旋轉軸

    # 獲取工件img的尺寸
    pixel_size = np.array([img.shape[1], img.shape[0], img.shape[2]]).astype(int)

    # 獲取物理坐標原點在工件img坐標系中的像素坐標
    origin = kwargs.get(
        "origin",
        np.array([pixel_size[0] // 2, pixel_size[1] // 2, 0], dtype=int),
    )

    # 取回精度
    precision = kwargs.get("precision", 4)
    verbose = kwargs.get("verbose", False)

    if verbose:
        print(f"start_point: {start_point} | 起點 in 物理坐標系")
        print(f"end_point: {end_point} | 終點 in 物理坐標系")
        print(f"rel_center: {rel_center} | 圓心 in 物理坐標系")
        print(f"origin: {origin} | 旋轉前物理坐標原點 in 工件坐標系")
        print(f"rotating_centers: {rotating_centers} | 旋轉中心 in 物理坐標系")
        print(f"angles: {rotating_angles} | 旋轉角度")
        print(f"axes: {rotating_axes} | 旋轉軸")
        print(f"pixel_size: {pixel_size} | 工件尺寸 in 工件坐標系")
        print("-" * 20)

    # 將起點和終點的物理坐標係旋轉後的物理坐標轉化為工件img的坐標系中的像素坐標
    start_point_pixel = physical_to_pixel(start_point, origin, pixel_size, precision)
    end_point_pixel = physical_to_pixel(end_point, origin, pixel_size, precision)
    center = np.array(start_point) + np.array(rel_center)
    center_pixel = physical_to_pixel(center, origin, pixel_size, precision)

    # 旋轉坐標
    if len(rotating_angles) > 0:
        _, start_point_rotated_pixel = multiple_rotated_physical_to_pixel(
            img,
            start_point,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            origin,
            precision,
        )
        _, end_point_rotated_pixel = multiple_rotated_physical_to_pixel(
            img,
            end_point,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            origin,
            precision,
        )
        _, center_rotated_pixel = multiple_rotated_physical_to_pixel(
            img,
            center,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            origin,
            precision,
        )
    else:
        start_point_rotated_pixel = physical_to_pixel(
            start_point, origin, pixel_size, precision
        )
        end_point_rotated_pixel = physical_to_pixel(
            end_point, origin, pixel_size, precision
        )
        center_rotated_pixel = physical_to_pixel(center, origin, pixel_size, precision)

    if verbose:
        print(
            f"start_point_pixel: {start_point} -> {start_point_pixel} | 起點的像素坐標 in 工件坐標系"
        )
        print(
            f"end_point_pixel: {end_point} -> {end_point_pixel} | 終點的像素坐標 in 工件坐標系"
        )
        print(f"center_pixel: {center} -> {center_pixel} | 圓心像素坐標 in 工件坐標系")
        print(
            f"start_point_rotated_pixel: {start_point} -> {start_point_rotated_pixel} | 起點旋轉後的像素坐標 in 工件坐標系"
        )
        print(
            f"end_point_rotated_pixel: {end_point} -> {end_point_rotated_pixel} | 終點旋轉後的像素坐標 in 工件坐標系"
        )
        print(
            f"center_rotated_pixel: {center} -> {center_rotated_pixel} | 圓心旋轉後的像素坐標 in 工件坐標系"
        )
        print("-" * 20)

    # 修正後的單位轉換
    tool_r_pixel = int(round(tool_r * 10 ** (precision - 3)))
    tool_h_pixel = int(round(tool_h * 10 ** (precision - 3)))

    # 確保 tool_r_pixel 和 tool_h_pixel 至少為 1，以滿足 OpenCV 函數的要求
    if tool_r_pixel <= 0:
        print(f"[ERROR] tool_r_pixel <= 0: {tool_r_pixel}")
        tool_r_pixel = 1
    if tool_h_pixel <= 0:
        print(f"[ERROR] tool_h_pixel <= 0: {tool_h_pixel}")
        tool_h_pixel = 1

    # 計算刀具路徑向量（像素坐標系）
    path_vector = end_point_rotated_pixel - start_point_rotated_pixel
    path_length = np.linalg.norm(path_vector)
    if path_length == 0:
        return None, None, row, "zero_length_toolpath"

    # 獲取旋轉後的坐標系基向量
    x_axis, y_axis, z_axis = _get_rotated_basis(
        img,
        origin,
        rotating_centers,
        rotating_angles,
        rotating_axes,
        precision,
        verbose,
    )

    # 構造旋轉矩陣 (從局部坐標到世界坐標)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)

    # 修正畫布建立方式
    # 計算旋轉後的圓心到起點的距離
    dx = start_point_pixel[0] - center_pixel[0]
    dy = start_point_pixel[1] - center_pixel[1]
    radius_pixel = int(np.sqrt(dx**2 + dy**2))

    # 建立以圓心為中心的畫布
    canvas_size = 2 * (radius_pixel + tool_r_pixel) + 1
    canvas_mask = np.zeros((canvas_size, canvas_size, tool_h_pixel), dtype=np.uint8)

    # 定義刀具坐標系原點(畫布中心)
    tool_origin = np.array([canvas_size // 2, canvas_size // 2, 0], dtype=int)
    base_offset = center_pixel.copy()

    # 修正角度計算(使用旋轉後坐標)
    dx_start = start_point_pixel[0] - center_pixel[0]
    dy_start = start_point_pixel[1] - center_pixel[1]
    theta_start = math.degrees(math.atan2(dy_start, dx_start))

    dx_end = end_point_pixel[0] - center_pixel[0]
    dy_end = end_point_pixel[1] - center_pixel[1]
    theta_end = math.degrees(math.atan2(dy_end, dx_end))

    # 修正角度範圍判斷
    if clockwise:
        if theta_end < theta_start:
            theta_end += 360
    else:
        if theta_start < theta_end:
            theta_start += 360

    with get_smart_tracer().log_event("CURVE.cv2.ellipse"):
        # 繪製橢圓時使用畫布中心作為圓心
        canvas_mask[:, :, 0] = cv2.ellipse(
            canvas_mask[:, :, 0].astype(np.uint8),
            (canvas_size // 2, canvas_size // 2),  # 使用畫布中心
            (radius_pixel, radius_pixel),
            0,
            theta_start,
            theta_end,
            1,
            thickness=2 * tool_r_pixel,
        )

    with get_smart_tracer().log_event("extend CURVE.cv2.ellipse"):
        # xy_indices = np.array(np.where(canvas_mask[:, :, 0] == 1)).T
        # for x, y in xy_indices:
        #    canvas_mask[x, y, 1:tool_h_pixel] = 1

        # 使用 numpy boardcast 特性直接將第一層複製到所有其他層
        canvas_mask[:, :, 1:tool_h_pixel] = canvas_mask[:, :, 0:1]

    with get_smart_tracer().log_event("drawcurve mask_points"):
        ## 獲取切割點坐標(刀具局部坐標系)
        mask_points = np.argwhere(canvas_mask > 0)  # yxz
        mask_points = mask_points[:, [1, 0, 2]]  # xyz
        mask_points = mask_points - tool_origin  # xyz

    # 如果沒有則為沒有切割軌跡
    if mask_points.size == 0:
        print(f"无切割像素点")
        return None, None, row, "no_cutting"

    with get_smart_tracer().log_event("drawcurve transform"):
        # 切割點坐標轉換到工件img坐標系
        # world_points = (
        #    cv2.transform(mask_points.reshape(-1, 1, 3), rotation_matrix).reshape(-1, 3)
        #    + base_offset
        # )
        world_points = (
            np.matmul(mask_points.reshape(-1, 1, 3), rotation_matrix).reshape(-1, 3)
            + base_offset
        )

    with get_smart_tracer().log_event("drawcurve round world points"):
        # 轉換為整數像素坐標
        pixel_coords = np.round(world_points).astype(int)  # xyz

    if verbose:
        print(
            f"X : {min(pixel_coords[:, 0])} -> {max(pixel_coords[:, 0])} | 切割點 in 工件坐標系"
        )
        print(
            f"Y : {min(pixel_coords[:, 1])} -> {max(pixel_coords[:, 1])} | 切割點 in 工件坐標系"
        )
        print(
            f"Z : {min(pixel_coords[:, 2])} -> {max(pixel_coords[:, 2])} | 切割點 in 工件坐標系"
        )
        print(f"工件坐標系: {img.shape}")

    with get_smart_tracer().log_event("drawcurve check valid dim"):
        # 檢查每個維度是否在有效範圍內
        x_valid = (pixel_coords[:, 0] >= 0) & (
            pixel_coords[:, 0] < img.shape[1]
        )  # X軸 (2540)
        y_valid = (pixel_coords[:, 1] >= 0) & (
            pixel_coords[:, 1] < img.shape[0]
        )  # Y軸 (1849)
        z_valid = (pixel_coords[:, 2] >= 0) & (
            pixel_coords[:, 2] < img.shape[2]
        )  # Z軸 (80)

        # 組合所有維度的有效性檢查
        valid_mask = x_valid & y_valid & z_valid

        # 篩選出有效的座標點
        valid_coords = pixel_coords[valid_mask]  # xyz

    if verbose:
        print(f"總座標點數: {len(pixel_coords)}")
        print(f"有效座標點數: {len(valid_coords)}")

    with get_smart_tracer().log_event("drawcurve operation other"):
        img_copy = img.copy().astype(np.uint8)
        # 設置切割區域顏色
        img_copy[valid_coords[:, 1], valid_coords[:, 0], valid_coords[:, 2]] = (
            CUTTING_MASK_COLOR  # img[yxz]
        )

        # 使用現有的valid_coords計算mask範圍
        if valid_coords.size == 0:
            print(f"切割像素点：{canvas_mask.sum()} 均不在工件內")
            return None, None, row, "cutting_outside_workpiece"

        # 修正座標軸索引
        min_x, max_x = valid_coords[:, 0].min(), valid_coords[:, 0].max() + 1  # X軸
        min_y, max_y = valid_coords[:, 1].min(), valid_coords[:, 1].max() + 1  # Y軸
        min_z, max_z = valid_coords[:, 2].min(), valid_coords[:, 2].max() + 1  # Z軸

        # 安全邊界檢查
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        min_z = max(min_z, 0)
        max_x = min(max_x, img_copy.shape[1])
        max_y = min(max_y, img_copy.shape[0])
        max_z = min(max_z, img_copy.shape[2])

        mask = img_copy[min_y:max_y, min_x:max_x, min_z:max_z].copy()
        mask_range = (min_y, max_y, min_x, max_x, min_z, max_z)

    return mask, mask_range, row, ""
