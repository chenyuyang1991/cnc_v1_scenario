import numpy as np
import cv2
import math
from cnc_genai.src.simulation.utils import physical_to_pixel


def rotated_physical_to_pixel(
    image,
    point_physical,
    rotating_center,
    angle,
    axis="Z",
    origin=None,
    precision=4,
    verbose=False,
):
    """
    旋转物理坐标系，將旋转后的物理坐標轉換為工件坐标系的像素坐標，工件坐标系不变，物理坐标系旋转
    注意这里的角度都是以右手坐标系为基准，逆时针为正，顺时针为负。与用户定义的角度有差异。

    image: np.ndarray, 工件的3D numpy图像
    point_physical: np.ndarray, 输入的旋转后的物理坐标系的坐標 (mm)
    rotating_center: np.ndarray, 旋轉中心原物理坐标系的坐标（mm）
    angle: float, 旋转的角度，角度单位为度
    origin: np.ndarray, 旋轉前的物理坐标系原点在工件坐标系中的像素坐标
    axis: str, 旋轉軸，"X","Y","Z"
    precision: int, 精度，預設值為4

    return: pixel_coordinates: 工件坐标系中的像素坐標（左上遠為原點）
    """

    # 如果origin为None，则使用这个原点位置
    size = np.array([image.shape[1], image.shape[0], image.shape[2]]).astype(int)
    if origin is None:
        origin = np.array([size[0] // 2, size[1] // 2, 0]).astype(int)

    # 物理坐标系绕Z轴旋转
    if axis == "Z":
        # 從Z軸正方向(屏幕外)觀察：
        # - 逆時針旋轉 → 角度為正
        # - 順時針旋轉 → 角度為負
        # 將角度轉換為弧度
        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 修正後的旋轉矩陣(符合從Z軸正方向觀察的逆時針旋轉)
        rotation_matrix = np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )

        # 確保輸入為浮點數類型並轉換為numpy數組
        point_physical = np.array(point_physical, dtype=np.float64)
        rotating_center = np.array(rotating_center, dtype=np.float64)
        relative_point = point_physical - rotating_center

        # 應用旋轉矩陣
        rotated_relative = np.dot(rotation_matrix, relative_point)

        # 轉換回原始物理坐標系
        original_point = rotated_relative + rotating_center

    # 新增X軸旋轉處理
    elif axis == "X":
        # 從X軸正方向(左到右)觀察：
        # - 順時針旋轉 → 角度為正
        # - 逆時針旋轉 → 角度為負
        theta = np.radians(-angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 修正後的旋轉矩陣(符合從X軸正方向觀察的順時針旋轉)
        rotation_matrix = np.array(
            [[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]]
        )

        # 確保輸入為浮點數類型並轉換為numpy數組
        point_physical = np.array(point_physical, dtype=np.float64)
        rotating_center = np.array(rotating_center, dtype=np.float64)
        relative_point = point_physical - rotating_center

        # 應用旋轉矩陣
        rotated_relative = np.dot(rotation_matrix, relative_point)

        # 轉換回原始物理坐標系
        original_point = rotated_relative + rotating_center

    # 新增Y軸旋轉處理
    elif axis == "Y":
        # 從Y軸正方向(下到上)觀察：
        # - 順時針旋轉 → 角度為正
        # - 逆時針旋轉 → 角度為負
        theta = np.radians(-angle)  # 修正角度方向
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 修正後的旋轉矩陣(符合從Y軸正方向觀察的順時針旋轉)
        rotation_matrix = np.array(
            [[cos_theta, 0, -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]]
        )

        # 確保輸入為浮點數類型並轉換為numpy數組
        point_physical = np.array(point_physical, dtype=np.float64)
        rotating_center = np.array(rotating_center, dtype=np.float64)
        relative_point = point_physical - rotating_center

        # 應用旋轉矩陣
        rotated_relative = np.dot(rotation_matrix, relative_point)

        # 轉換回原始物理坐標系
        original_point = rotated_relative + rotating_center

    else:
        raise ValueError(f"Invalid rotation axis: {axis}")

    # 以上为物理坐标系，右手坐标
    # 轉換到像素坐標系 (左手坐标系，注意Y軸方向相反)
    pixel_x = original_point[0] * 10 ** (precision - 3) + origin[0]
    pixel_y = size[1] - (original_point[1] * 10 ** (precision - 3) + origin[1])
    pixel_z = original_point[2] * 10 ** (precision - 3) + origin[2]
    pixel_coordinates = np.array([pixel_x, pixel_y, pixel_z])

    if verbose:
        print(
            f"坐标{point_physical}在{rotating_center}点绕{axis}轴正方向順時針旋转{angle}后为{[float(i) for i in original_point]}，在原始工件坐标系中的像素坐标为{[int(i) for i in pixel_coordinates]}"
        )

    return original_point.astype(float), pixel_coordinates.astype(int)


def multiple_rotated_physical_to_pixel(
    image,
    point_physical,
    rotating_centers,
    angles,
    axes=[],
    origin=None,
    precision=4,
    verbose=True,
):
    """
    连续旋转物理坐标系，將旋转后的物理坐標轉換為工件坐标系的像素坐標，工件坐标系不变，物理坐标系旋转

    image: np.ndarray, 工件的3D numpy图像
    point_physical: np.ndarray, 输入的旋转后的物理坐标系的坐標 (mm)
    rotating_centers: List[np.ndarray], 列表，每次的旋轉中心原物理坐标系的坐标（mm）
    angles: List[float], 列表，每次旋转的角度，角度单位为度
    origin: np.ndarray, 旋轉前的物理坐标系原点在工件坐标系中的像素坐标
    axes: List[str], 列表，每次的旋轉軸，"X","Y","Z"
    precision: int, 精度，預設值為4
    verbose: bool, 是否打印詳細信息

    正负与方向:
        1. [0,1,0] --(绕0/0/0绕Z轴旋转90°)--> [-1,0,0]，沿坐标方向顺时针为正，俯视逆时针为正（工件沿坐标轴逆时针为正）
        2. [0,1,0] --(绕0/0/0绕X轴旋转90°)--> [0,0,1]，沿坐标方向顺时针为正，左往右看顺时针为正（工件沿坐标轴逆时针为正）
        3. [1,0,0] --(绕0/0/0绕Y轴旋转90°)--> [0,0,-1]，沿坐标方向顺时针为正，后往前看顺时针为正（工件沿坐标轴逆时针为正）

    return:
    original_point: 最終物理坐標系中的坐標 (mm)
    pixel_coordinates: 工件坐标系中的像素坐標（左上遠為原點）
    """
    # 初始化當前物理坐標
    current_point = np.array(point_physical, dtype=np.float64)
    size = np.array([image.shape[1], image.shape[0], image.shape[2]]).astype(int)

    # 設置原點
    if origin is None:
        origin = np.array([size[0] // 2, size[1] // 2, 0]).astype(int)

    if not len(angles):
        # 無旋轉時直接轉換
        return current_point.astype(float), physical_to_pixel(
            current_point, origin, size, precision=precision
        )

    # 循環調用單次旋轉函數
    for i, (rot_center, angle, axis) in enumerate(zip(rotating_centers, angles, axes)):
        # 調用單次旋轉函數
        current_point, _ = rotated_physical_to_pixel(
            image=image,
            point_physical=current_point,  # 使用上次旋轉後的坐標
            rotating_center=rot_center,
            angle=angle,
            axis=axis,
            origin=origin,  # 始終使用原始原點
            precision=precision,
        )

        if verbose:
            print(
                f"第{i+1}次旋转({axis}轴 {angle}度)后物理坐标: {np.round(current_point, 2)}"
            )

    # 最終轉換到像素坐標系
    pixel_x = current_point[0] * 10 ** (precision - 3) + origin[0]
    pixel_y = size[1] - (current_point[1] * 10 ** (precision - 3) + origin[1])
    pixel_z = current_point[2] * 10 ** (precision - 3) + origin[2]
    pixel_coordinates = np.array([pixel_x, pixel_y, pixel_z])

    return current_point.astype(float), pixel_coordinates.astype(int)
