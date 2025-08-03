import numpy as np
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.ndimage import convolve, distance_transform_edt
from cnc_genai.src.simulation.colors import MATERIAL_COLOR


def get_contour_points_sparse(mask):
    """
    使用稀疏点方法获取3D mask的轮廓点。
    仅处理非零点，减少无意义的遍历。
    """
    # 获取所有前景点的坐标
    coords = np.array(np.where(mask)).T

    # 定义26个方向的相对位移 (26邻域)
    directions = np.array(
        [
            [dx, dy, dz]
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
    )

    # 用于存储轮廓点
    contour_points = []

    # 遍历所有前景点，仅检查其邻居
    for point in coords:
        neighbors = point + directions
        # 检查是否有邻居超出边界或属于背景
        for neighbor in neighbors:
            if (
                (neighbor < 0).any()
                or (neighbor >= mask.shape).any()
                or mask[tuple(neighbor)] == 0
            ):
                contour_points.append(point)
                break

    return np.array(contour_points)


def get_contour_points_block(mask):
    """
    分块处理获取3D mask的轮廓点，仅在边界附近操作。
    """
    # 获取所有前景点的坐标
    coords = np.array(np.where(mask)).T

    # 定义边界范围
    min_coords = coords.min(axis=0) - 1
    max_coords = coords.max(axis=0) + 1

    # 截取边界小块
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, np.array(mask.shape) - 1)
    sub_mask = mask[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]

    # 获取小块的轮廓点
    sub_contour = get_contour_points_sparse(sub_mask)

    # 恢复全局坐标
    contour = sub_contour + min_coords
    return contour


def get_mask_contour(mask):
    """
    直接提取3D mask的轮廓点，可能更慢。
    使用3D卷积计算每个点的邻居情况。
    """
    # 定义3D卷积核，检查26邻域
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0  # 去掉中心点

    # 计算邻域和
    neighbor_sum = convolve(mask.astype(int), kernel, mode="constant", cval=0)

    # 轮廓点：mask中为1，且邻居中至少有一个是背景
    contour = (mask == 1) & (neighbor_sum < 26)
    return np.array(np.where(contour)).T


def get_distance_between_contours(contour1, contour2):
    """
    计算两个掩码之间的距离。该函数通过比较两个掩码的轮廓，计算它们之间的欧氏距离。

    參數:
    mask1: numpy.ndarray
        第一个轮廓列表。
    mask2: numpy.ndarray
        第二个轮廓列表。

    返回值:
    float
        两个掩码之间的欧氏距离。
    """
    # 使用KD树计算最小距离
    tree = cKDTree(contour2)  # 构建mask2的KD树
    distances, _ = tree.query(contour1)  # 查询contour1到contour2的最近距离

    # 返回最小的距离
    return np.min(distances)


def get_distance_between_masks(image, mask, tile_range, tile):
    internal_current_time = datetime.now()
    mask_cutting = np.zeros_like(mask, dtype=np.uint8)
    mask_cutting[
        tile_range[0] : tile_range[1],
        tile_range[2] : tile_range[3],
        tile_range[4] : tile_range[5],
    ] = tile
    print(f"||| 生成mask cutting {datetime.now() - internal_current_time}")

    internal_current_time = datetime.now()
    mask_material = np.zeros_like(mask, dtype=np.uint8)
    mask_material[np.all(image == MATERIAL_COLOR, axis=-1)] = 1
    print(f"||| 生成mask material {datetime.now() - internal_current_time}")
    internal_current_time = datetime.now()

    contour_cutting = get_mask_contour(mask_cutting)
    contour_material = get_mask_contour(mask_material)
    print(f"||| 计算contour {datetime.now() - internal_current_time}")
    internal_current_time = datetime.now()

    dist = get_distance_between_contours(
        contour_cutting,
        contour_material,
    )
    print(f"||| 计算contour之间的距离 {datetime.now() - internal_current_time}")
    return dist
