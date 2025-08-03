import math
import numpy as np
import cv2
from datetime import datetime
from cnc_genai.src.simulation.colors import CUTTING_MASK_COLOR


# numpy 计算太慢，已经舍弃，请用_cv方法，使用C++编译的opencv库
def draw_G01(img, start_point, end_point, threshold):
    """
    根据给定的起始点、终止点和阈值，在三维图像中绘制一条直线，并将距离直线距离小于阈值的像素设为目标值。

    参数:
    img: numpy.ndarray
        三维图像数据。
    start_point: tuple
        直线的起始点坐标 (x1, y1, z1)。
    end_point: tuple
        直线的终止点坐标 (x2, y2, z2)。
    threshold: float
        距离直线交点的最大距离阈值。

    返回:
    numpy.ndarray
        图片需要更新区域的像素点
    """
    # 直线定义，经过两点 (x1, y1, z1) 和 (x2, y2, z2)
    x1, y1, z1 = start_point
    x2, y2, z2 = end_point

    if z1 == z2:
        # 直线与图像平面重合，计算直线在 z = z1 平面的方程
        a = x1 - x2
        b = y2 - y1
        c = x2 * y1 - x1 * y2

        # 创建 z = z1 平面的像素网格
        xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))

        # 计算每个像素点到直线的距离
        distance = np.abs(a * xx + b * yy + c) / np.sqrt(a**2 + b**2)

        # 设置距离小于 s 的像素点值为 0
        pixels = np.where(distance < threshold)  # 获取满足条件的XY坐标
        pixels = np.column_stack((pixels[0], pixels[1], np.full_like(pixels[0], z1)))

    elif (x1, y1) != (x2, y2):
        pass

    else:
        # 方向向量
        vx, vy, vz = x2 - x1, y2 - y1, z2 - z1

        # 生成每个像素的坐标网格
        xx, yy, zz = np.meshgrid(
            np.arange(img.shape[0]),
            np.arange(img.shape[1]),
            np.arange(img.shape[2]),
            indexing="ij",
        )

        # 计算 t（交点的参数）
        with np.errstate(divide="ignore", invalid="ignore"):  # 忽略除以零的警告
            t = (zz - z1) / vz
            t = np.nan_to_num(t, nan=0.0)  # 将无效 t（如 vz = 0 的情况）置为 0

        # 计算交点坐标 (x', y', z')
        x_prime = x1 + t * vx
        y_prime = y1 + t * vy

        # 计算点到交点的距离
        distances = np.sqrt((xx - x_prime) ** 2 + (yy - y_prime) ** 2)

        # 更新图像
        pixels = np.where(distances < threshold)

    return pixels


def draw_G02(img, start_point, end_point, center, threshold, num_points=100):
    """
    根据给定的起始点、终止点、圆心和阈值，在三维图像中绘制一条顺时针圆弧，并将距离圆弧距离小于阈值的像素设为目标值。

    参数:
    img: numpy.ndarray
        三维图像数据。
    start_point: tuple
        圆弧的起始点坐标 (x1, y1, z1)。
    end_point: tuple
        圆弧的终止点坐标 (x2, y2, z2)。
    center: tuple
        圆弧的圆心坐标 (i, j)。
    threshold: float
        距离圆弧交点的最大距离阈值。
    num_points: int
        生成圆弧上的点的数量，默认为 1000。

    返回:
    numpy.ndarray
        图片需要更新区域的像素点
    """
    x1, y1, z = start_point
    x2, y2, z = end_point
    i, j = center

    # 计算半径
    R = np.sqrt((x1 - i) ** 2 + (y1 - j) ** 2)

    # 计算起点和终点的角度
    theta_start = np.arctan2(y1 - j, x1 - i)
    theta_end = np.arctan2(y2 - j, x2 - i)

    # 顺时针角度处理
    if theta_end < theta_start:
        theta_end += 2 * np.pi

    # 生成圆弧上的点
    theta = np.linspace(theta_start, theta_end, num_points)
    arc_points = np.stack((i + R * np.cos(theta), j + R * np.sin(theta)), axis=-1)

    # 创建像素坐标网格
    xx, yy = np.meshgrid(
        np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij"
    )

    # 计算每个像素点到圆弧的最小距离
    distances = np.full_like(xx, np.inf, dtype=np.float32)
    for px, py in arc_points:
        distances = np.minimum(distances, np.sqrt((xx - px) ** 2 + (yy - py) ** 2))

    # 更新图像
    pixels = np.where(distances < threshold)
    # valid_pixels = check_pixel_range(img, pixels)

    return pixels


def draw_G03(img, start_point, end_point, center, threshold, num_points=100):
    """
    根据给定的起始点、终止点和圆心，在三维图像中绘制一条逆时针圆弧，并将距离圆弧距离小于阈值的像素设为目标值。

    参数:
    img: numpy.ndarray
        三维图像数据。
    start_point: tuple
        圆弧的起始点坐标 (x1, y1, z1)。
    end_point: tuple
        圆弧的终止点坐标 (x2, y2, z2)。
    center: tuple
        圆弧的圆心坐标 (i, j)。
    threshold: float
        距离圆弧交点的最大距离阈值。
    num_points: int
        生成圆弧上的点的数量，默认为 1000。

    返回:
    numpy.ndarray
        更新后的图像数据。
    """
    x1, y1, z = start_point
    x2, y2, z = end_point
    i, j = center

    # 计算半径
    R = np.sqrt((x1 - i) ** 2 + (y1 - j) ** 2)

    # 计算起点和终点的角度
    theta_start = np.arctan2(y1 - j, x1 - i)
    theta_end = np.arctan2(y2 - j, x2 - i)

    # 逆时针角度处理
    if theta_start < theta_end:
        theta_start += 2 * np.pi

    # 生成圆弧上的点
    theta = np.linspace(theta_start, theta_end, num_points)  # 逆时针角度
    arc_points = np.stack((i + R * np.cos(theta), j + R * np.sin(theta)), axis=-1)

    # 创建像素坐标网格
    xx, yy = np.meshgrid(
        np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij"
    )

    # 计算每个像素点到圆弧的最小距离
    distances = np.full_like(xx, np.inf, dtype=np.float32)
    for px, py in arc_points:
        distances = np.minimum(distances, np.sqrt((xx - px) ** 2 + (yy - py) ** 2))

    pixels = np.where(distances < threshold)

    # valid_pixels = check_pixel_range(img, pixels)

    return pixels


# 以下为使用C++编译的opencv库画图的方法
def draw_G01_cv(img, start_point, end_point, tool_r, tool_h, **kwargs):
    """
    根据给定的起始点、终止点和阈值，在三维图像中绘制一条直线，并将距离直线距离小于阈值的像素设为目标值。

    参数:
    img: numpy.ndarray
        三维图像数据。
    start_point: tuple
        直线的起始点坐标 (x1, y1, z1)。
    end_point: tuple
        直线的终止点坐标 (x2, y2, z2)。
    threshold: float
        距离直线交点的最大距离阈值。

    返回:
    numpy.ndarray
        图片需要更新区域的像素点
    """

    use_gpu = kwargs.get("use_gpu", False)
    if use_gpu:
        import cupy

    mask = np.zeros_like(img)

    # 直线定义，经过两点 (x1, y1, z1) 和 (x2, y2, z2)
    x1, y1, z1 = start_point
    x2, y2, z2 = end_point

    # if max(z1, z2) + tool_h < 0:
    #     print(f"| 直线切割，起点{(x1, y1, z1)}，终点{(x2, y2, z2)}，刀頭無有效切割")
    #     return None, None

    current_time = datetime.now()

    if z1 == z2:
        print(
            f"| 在XY平面进行G01直线切割，起点{(x1, y1)}，终点{(x2, y2)}，当前刀具直径{2 * tool_r}，刀頭高度{tool_h}，深度为{z1}"
        )
        mask[:, :, max(0, z1)] = cv2.line(
            mask[:, :, max(0, z1)].astype(np.uint8),
            (x1, y1),
            (x2, y2),
            CUTTING_MASK_COLOR,
            2 * tool_r,
        )
        xy_indices = np.array(np.where(mask[:, :, max(0, z1)] == CUTTING_MASK_COLOR)).T
        for x, y in xy_indices:
            mask[x, y, max(0, z1) + 1 : z1 + tool_h + 1] = CUTTING_MASK_COLOR
    else:
        if z1 < 0 and z2 < 0:
            print(
                f"| 斜线切割，起点{(x1, y1, z1)}，终点{(x2, y2, z2)}，但是z1 < 0 and z2 < 0，在 z = 0 平面画直线，当前刀具直径{2 * tool_r}"
            )

            if max(z1, z2) + tool_h >= 0:
                mask[:, :, 0] = cv2.line(
                    mask[:, :, 0].astype(np.uint8),
                    (x1, y1),
                    (x2, y2),
                    CUTTING_MASK_COLOR,
                    2 * tool_r,
                )
                xy_indices = np.array(np.where(mask[:, :, 0] == CUTTING_MASK_COLOR)).T
                for x, y in xy_indices:
                    mask[x, y, : max(z1, z2) + tool_h + 1] = CUTTING_MASK_COLOR
        elif (x1, y1) != (x2, y2):
            print(
                f"| 斜线切割，起点{(x1, y1, z1)}，终点{(x2, y2, z2)}，当前刀具直径{2 * tool_r}"
            )

            for each_z in range(max(0, min(z1, z2)), max(0, max(z1, z2)) + 1):
                x = int(x1 + (x2 - x1) * (each_z - z1) / (z2 - z1))
                y = int(y1 + (y2 - y1) * (each_z - z1) / (z2 - z1))

                mask[:, :, each_z] = cv2.circle(
                    mask[:, :, each_z].astype(np.uint8),
                    (x, y),
                    tool_r,
                    CUTTING_MASK_COLOR,
                    -1,
                )
                xy_indices = np.array(
                    np.where(mask[:, :, each_z] == CUTTING_MASK_COLOR)
                ).T
                for x, y in xy_indices:
                    mask[x, y, max(0, each_z) + 1 : each_z + tool_h + 1] = (
                        CUTTING_MASK_COLOR
                    )
        else:
            print(
                f"| 沿Z轴进行G01直线切割，起点{(x1, y1, z1)}，终点{(x2, y2, z2)}，当前刀具直径{2 * tool_r}"
            )

            mask[:, :, max(0, min(z1, z2))] = cv2.circle(
                mask[:, :, max(0, min(z1, z2))].astype(np.uint8),
                (x1, y1),
                tool_r,
                CUTTING_MASK_COLOR,
                -1,
            )
            xy_indices = np.array(
                np.where(mask[:, :, max(0, min(z1, z2))] == CUTTING_MASK_COLOR)
            ).T
            for x, y in xy_indices:
                mask[
                    x, y, max(0, min(z1, z2)) + 1 : max(0, max(z1, z2) + tool_h) + 1
                ] = CUTTING_MASK_COLOR

    if use_gpu:
        current_time = datetime.now()
        mask_cupy = cupy.asarray(mask)
        voxels_xyz = cupy.where(mask_cupy == CUTTING_MASK_COLOR)

        min_0, max_0 = voxels_xyz[0].min(), voxels_xyz[0].max() + 1
        min_1, max_1 = voxels_xyz[1].min(), voxels_xyz[1].max() + 1
        min_2, max_2 = voxels_xyz[2].min(), voxels_xyz[2].max() + 1

        min_0 = cupy.asnumpy(min_0)
        max_0 = cupy.asnumpy(max_0)
        min_1 = cupy.asnumpy(min_1)
        max_1 = cupy.asnumpy(max_1)
        min_2 = cupy.asnumpy(min_2)
        max_2 = cupy.asnumpy(max_2)
        print(f"draw_G01_cv: per-voxel coord cost {datetime.now() - current_time}")

        return mask_cupy[min_0:max_0, min_1:max_1, min_2:max_2], (
            min_0,
            max_0,
            min_1,
            max_1,
            min_2,
            max_2,
        )

    pixels = np.where(mask == CUTTING_MASK_COLOR)
    min_0, max_0 = pixels[0].min(), pixels[0].max() + 1
    min_1, max_1 = pixels[1].min(), pixels[1].max() + 1
    min_2, max_2 = pixels[2].min(), pixels[2].max() + 1

    return mask[min_0:max_0, min_1:max_1, min_2:max_2], (
        min_0,
        max_0,
        min_1,
        max_1,
        min_2,
        max_2,
    )


def draw_G02_cv(img, start_point, end_point, center, threshold, tool_h, **kwargs):
    """
    根据给定的起始点、终止点、圆心和阈值，在三维图像中绘制一条顺时针圆弧，并将距离圆弧距离小于阈值的像素设为目标值。

    参数:
    img: numpy.ndarray
        三维图像数据。
    start_point: tuple
        圆弧的起始点坐标 (x1, y1, z1)。
    end_point: tuple
        圆弧的终止点坐标 (x2, y2, z2)。
    center: tuple
        圆弧的圆心坐标 (i, j)。
    threshold: float
        距离圆弧交点的最大距离阈值。
    num_points: int
        生成圆弧上的点的数量，默认为 1000。

    返回:
    numpy.ndarray
        图片需要更新区域的像素点
    """

    use_gpu = kwargs.get("use_gpu", False)
    if use_gpu:
        import cupy

    assert start_point[2] == end_point[2]
    mask = np.zeros_like(img)
    x1, y1, z = start_point
    x2, y2, z = end_point
    i, j = center

    # 计算半径
    R = int(np.sqrt((x1 - i) ** 2 + (y1 - j) ** 2))

    # 计算起点和终点的角度
    theta_start = math.degrees(math.atan2(y1 - j, x1 - i))
    theta_end = math.degrees(math.atan2(y2 - j, x2 - i))

    if theta_end < theta_start:
        theta_end += 360
    theta_end = max(theta_end, theta_start + 1)

    if z + tool_h < 0:
        print(
            f"| 在XY平面进行G02顺时针圆弧切割，起点{(x1, y1, z)}，终点{(x2, y2, z)}，刀頭無有效切割"
        )
        return None, None

    print(
        f"| 在XY平面进行G02顺时针圆弧切割，起点{(x1, y1, z)}，角度{theta_start}，终点{(x2, y2, z)}，角度{theta_end}，圆心为{(i, j)}，半径为{R}，深度为{z}，当前刀具直径{2*threshold}"
    )
    mask[:, :, max(0, z)] = cv2.ellipse(
        mask[:, :, max(0, z)].astype(np.uint8),
        np.array([i, j]),
        (R, R),
        0,
        theta_start,
        theta_end,
        CUTTING_MASK_COLOR,
        thickness=2 * threshold,
    )
    xy_indices = np.array(np.where(mask[:, :, max(0, z)] == CUTTING_MASK_COLOR)).T
    for x, y in xy_indices:
        mask[x, y, max(0, z) + 1 : max(0, z) + tool_h + 1] = CUTTING_MASK_COLOR

    if use_gpu:
        # GPU accelerate
        current_time = datetime.now()
        mask_cupy = cupy.asarray(mask)
        voxels_xyz = cupy.where(mask_cupy == CUTTING_MASK_COLOR)
        if len(voxels_xyz[0]):
            min_0, max_0 = voxels_xyz[0].min(), voxels_xyz[0].max() + 1
            min_1, max_1 = voxels_xyz[1].min(), voxels_xyz[1].max() + 1
            min_2, max_2 = voxels_xyz[2].min(), voxels_xyz[2].max() + 1

            min_0 = cupy.asnumpy(min_0)
            max_0 = cupy.asnumpy(max_0)
            min_1 = cupy.asnumpy(min_1)
            max_1 = cupy.asnumpy(max_1)
            min_2 = cupy.asnumpy(min_2)
            max_2 = cupy.asnumpy(max_2)

            print(f"draw_G02_cv: per-voxel coord cost {datetime.now() - current_time}")
            return mask_cupy[min_0:max_0, min_1:max_1, min_2:max_2], (
                min_0,
                max_0,
                min_1,
                max_1,
                min_2,
                max_2,
            )
        else:
            print("| 當前精度下沒有切割到像素點")
            return None, None

    pixels = np.where(mask == CUTTING_MASK_COLOR)
    if len(pixels[0]):
        min_0, max_0 = pixels[0].min(), pixels[0].max() + 1
        min_1, max_1 = pixels[1].min(), pixels[1].max() + 1
        min_2, max_2 = pixels[2].min(), pixels[2].max() + 1
        return mask[min_0:max_0, min_1:max_1, min_2:max_2], (
            min_0,
            max_0,
            min_1,
            max_1,
            min_2,
            max_2,
        )
    else:
        print("| 當前精度下沒有切割到像素點")
        return None, None


def draw_G03_cv(img, start_point, end_point, center, threshold, tool_h, **kwargs):
    """
    根据给定的起始点、终止点、圆心和阈值，在三维图像中绘制一条顺时针圆弧，并将距离圆弧距离小于阈值的像素设为目标值。

    参数:
    img: numpy.ndarray
        三维图像数据。
    start_point: tuple
        圆弧的起始点坐标 (x1, y1, z1)。
    end_point: tuple
        圆弧的终止点坐标 (x2, y2, z2)。
    center: tuple
        圆弧的圆心坐标 (i, j)。
    threshold: float
        距离圆弧交点的最大距离阈值。
    num_points: int
        生成圆弧上的点的数量，默认为 1000。

    返回:
    numpy.ndarray
        图片需要更新区域的像素点
    """

    use_gpu = kwargs.get("use_gpu", False)
    if use_gpu:
        import cupy

    assert start_point[2] == end_point[2]
    mask = np.zeros_like(img)
    x1, y1, z = start_point
    x2, y2, z = end_point
    i, j = center

    # 计算半径
    R = int(np.sqrt((x1 - i) ** 2 + (y1 - j) ** 2))

    # 计算起点和终点的角度
    theta_start = math.degrees(math.atan2(y1 - j, x1 - i))
    theta_end = math.degrees(math.atan2(y2 - j, x2 - i))

    if theta_start < theta_end:
        theta_start += 360

    if z + tool_h < 0:
        print(
            f"| 在XY平面进行G03逆时针圆弧切割，起点{(x1, y1, z)}，终点{(x2, y2, z)}，刀頭無有效切割"
        )
        return None, None

    print(
        f"| 在XY平面进行G03逆时针圆弧切割，起点{(x1, y1, z)}，角度{theta_start}，终点{(x2, y2, z)}，角度{theta_end}，圆心为{(i, j)}，半径为{R}，深度为{z}，当前刀具直径{2*threshold}"
    )

    current_time = datetime.now()

    mask[:, :, max(0, z)] = cv2.ellipse(
        mask[:, :, max(0, z)].astype(np.uint8),
        np.array([i, j]),
        (R, R),
        0,
        theta_start,
        theta_end,
        CUTTING_MASK_COLOR,
        thickness=2 * threshold,
    )
    print(f"draw_G03_cv: ellipse() cost {datetime.now() - current_time}")
    current_time = datetime.now()
    xy_indices = np.array(np.where(mask[:, :, max(0, z)] == CUTTING_MASK_COLOR)).T
    for x, y in xy_indices:
        mask[x, y, max(0, z) + 1 : max(0, z) + tool_h + 1] = CUTTING_MASK_COLOR
    print(f"draw_G03_cv: masking along z cost {datetime.now() - current_time}")
    current_time = datetime.now()

    if use_gpu:
        # GPU accelerate
        mask_cupy = cupy.asarray(mask)
        voxels_xyz = cupy.where(mask_cupy == CUTTING_MASK_COLOR)
        if len(voxels_xyz[0]):
            min_0, max_0 = voxels_xyz[0].min(), voxels_xyz[0].max() + 1
            min_1, max_1 = voxels_xyz[1].min(), voxels_xyz[1].max() + 1
            min_2, max_2 = voxels_xyz[2].min(), voxels_xyz[2].max() + 1

            min_0 = cupy.asnumpy(min_0)
            max_0 = cupy.asnumpy(max_0)
            min_1 = cupy.asnumpy(min_1)
            max_1 = cupy.asnumpy(max_1)
            min_2 = cupy.asnumpy(min_2)
            max_2 = cupy.asnumpy(max_2)

            print(f"draw_G03_cv: per-voxel coord cost {datetime.now() - current_time}")
            return mask_cupy[min_0:max_0, min_1:max_1, min_2:max_2], (
                min_0,
                max_0,
                min_1,
                max_1,
                min_2,
                max_2,
            )
        else:
            print("| 當前精度下沒有切割到像素點")
            return None, None

    pixels = np.where(mask == CUTTING_MASK_COLOR)
    if len(pixels[0]):
        min_0, max_0 = pixels[0].min(), pixels[0].max() + 1
        min_1, max_1 = pixels[1].min(), pixels[1].max() + 1
        min_2, max_2 = pixels[2].min(), pixels[2].max() + 1
        return mask[min_0:max_0, min_1:max_1, min_2:max_2], (
            min_0,
            max_0,
            min_1,
            max_1,
            min_2,
            max_2,
        )
    else:
        print("| 當前精度下沒有切割到像素點")
        return None, None
