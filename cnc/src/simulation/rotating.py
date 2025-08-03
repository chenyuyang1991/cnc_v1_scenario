import cv2
import numpy as np
#import cupynumeric as np
from tqdm import tqdm
import math


def rotate_X_axis(image, rotating_center, angle, origin=None, precision=4):
    """
    四轴旋转，绕X轴旋转。左視圖顺时针为負，左視圖逆时针为正

    參數:
      image: numpy.ndarray
          4D 圖形數據，形狀為 (rows, cols, depth, channels)
      rotating_center: tuple
          旋轉之前旋轉中心的物理座標 (x, y, z)
      angle: float
          左視圖順時針旋轉為負角度（單位：度）
      origin: tuple, optional
          旋轉前物理坐標原點對應的像素坐標 (x, y, z)
          若未指定，則預設使用：
             (pixel_size[0]//2, size[1] - pixel_size[1]//2, 0)
      precision: int
          精度，預設為4

    返回:
      rotated: 旋轉後的圖像
      new_shape: rotated的形狀
      new_rotating_center: 旋轉後，旋轉中心在旋轉後的圖像rotated坐標系下的像素坐標
      new_origin: 旋轉後，物理坐標原點在旋轉後的圖像rotated坐標系下的像素坐標
    """

    # 獲取圖像尺寸
    pixel_size = np.array([image.shape[1], image.shape[0], image.shape[2]]).astype(int)

    # 設定預設的origin值
    if origin is None:
        origin = (pixel_size[0] // 2, pixel_size[1] - pixel_size[1] // 2, 0)

    # 將旋轉中心的物理坐標轉換到像素坐標
    rotating_center_pixel = (
        int(rotating_center[0] * 10 ** (precision - 3) + origin[0]),
        int(pixel_size[1] - (rotating_center[1] * 10 ** (precision - 3) + origin[1])),
        int(rotating_center[2] * 10 ** (precision - 3) + origin[2]),
    )

    # 計算旋轉矩陣
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    # 計算四個角點相對於旋轉中心的位置
    corners_y = np.array(
        [
            -rotating_center_pixel[1],  # 左上角
            -rotating_center_pixel[1],  # 右上角
            pixel_size[1] - rotating_center_pixel[1],  # 左下角
            pixel_size[1] - rotating_center_pixel[1],  # 右下角
        ]
    )

    corners_z = np.array(
        [
            -rotating_center_pixel[2],  # 左上角
            pixel_size[2] - rotating_center_pixel[2],  # 右上角
            -rotating_center_pixel[2],  # 左下角
            pixel_size[2] - rotating_center_pixel[2],  # 右下角
        ]
    )

    # 計算所有角點旋轉後的位置
    y_rotated = corners_y * cos_theta - corners_z * sin_theta
    z_rotated = corners_y * sin_theta + corners_z * cos_theta

    # 計算旋轉後的最大延伸距離
    max_y_extension = max(abs(y_rotated.max()), abs(y_rotated.min()))
    max_z_extension = max(abs(z_rotated.max()), abs(z_rotated.min()))

    # 計算新尺寸，確保旋轉中心在圖像正中心
    new_h = int(2 * max_y_extension) + 20  # 增加更多緩衝區
    new_d = int(2 * max_z_extension) + 20

    # 確保新尺寸為偶數，便於中心對齊
    new_h = new_h + (new_h % 2)
    new_d = new_d + (new_d % 2)

    # 創建新的圖像陣列
    rotated_image = np.zeros(
        (new_h, pixel_size[0], new_d, image.shape[3]), dtype=image.dtype
    )

    # 計算旋轉中心在新圖像中的位置（正中心）
    center_y = new_h // 2
    center_z = new_d // 2

    # 對每個x座標進行處理
    for x in tqdm(range(pixel_size[0])):
        # 獲取yz平面
        yz_plane = image[:, x, :, :]

        # 創建座標網格
        y_coords, z_coords = np.meshgrid(
            np.arange(pixel_size[1]), np.arange(pixel_size[2]), indexing="ij"
        )

        # 平移到旋轉中心
        y_centered = y_coords - rotating_center_pixel[1]
        z_centered = z_coords - rotating_center_pixel[2]

        # 計算旋轉後的座標(以旋轉中心為原點)
        y_rotated = (y_centered * cos_theta - z_centered * sin_theta + center_y).astype(
            int
        )
        z_rotated = (y_centered * sin_theta + z_centered * cos_theta + center_z).astype(
            int
        )

        # 找出有效的座標
        valid = (
            (y_rotated >= 0)
            & (y_rotated < new_h)
            & (z_rotated >= 0)
            & (z_rotated < new_d)
        )

        # 將原始圖像的值映射到新的位置
        for c in range(image.shape[3]):
            rotated_image[y_rotated[valid], x, z_rotated[valid], c] = yz_plane[
                y_coords[valid], z_coords[valid], c
            ]

    # 計算新坐標系下的中心點
    new_shape = np.array(
        [rotated_image.shape[1], rotated_image.shape[0], rotated_image.shape[2]]
    ).astype(int)
    new_rotating_center = (rotating_center_pixel[0], center_y, center_z)
    new_origin = (
        origin[0] - rotating_center_pixel[0] + new_rotating_center[0],
        origin[1] - rotating_center_pixel[1] + new_rotating_center[1],
        origin[2] - rotating_center_pixel[2] + new_rotating_center[2],
    )

    print(
        f"| 在YZ平面进行X轴旋转，绕中心点{rotating_center}，旋转{angle}角度，新的图像尺寸为{new_shape}，旋轉中心在新圖像的像素坐标为{new_rotating_center}，物理坐标原點在新圖像的像素坐标为{new_origin}"
    )

    return rotated_image, new_shape, new_rotating_center, new_origin


def rotate_Z_axis(image, rotating_center, angle, origin=None, precision=4):
    """
    將圖形在物理坐標的 X Y 平面上逆時針旋轉指定角度。即0.5轴旋转

    參數:
      image: numpy.ndarray
          4D 圖形數據，形狀為 (rows, cols, depth, channels)
      rotating_center: tuple
          旋轉之前旋轉中心的物理座標 (x, y, z)
      angle: float
          順時針旋轉的角度（單位：度）
      origin: tuple, optional
          旋轉前物理坐標原點對應的像素坐標 (x, y, z)
          若未指定，則預設使用：
             (pixel_size[0]//2, size[1] - pixel_size[1]//2, 0)
          使得物理坐標 (0, 0, 0) 對應的像素座標為 (origin[0], size[1]-origin[1], 0)

    返回:
      rotated: 旋轉後的圖像。
      new_shape: rotated的形狀
      new_rotating_center: 旋轉後，旋轉中心在旋轉後的圖像rotated坐標系下的像素坐標
      new_origin: 旋轉後，物理坐標原點在旋轉後的圖像rotated坐標系下的像素坐標
    """
    pixel_size = np.array([image.shape[1], image.shape[0], image.shape[2]]).astype(int)

    # 設定預設的 center 值
    if origin is None:
        origin = (pixel_size[0] // 2, pixel_size[1] - pixel_size[1] // 2, 0)
    # 將旋轉中心的物理坐標 rotating_center 轉換到像素坐標
    # 注意：物理座標轉像素時需要乘上比例因子 10**(precision-3)
    rotating_center_pixel = (
        int(rotating_center[0] * 10 ** (precision - 3) + origin[0]),
        int(pixel_size[1] - (rotating_center[1] * 10 ** (precision - 3) + origin[1])),
    )
    print(f"您的旋轉中心物理坐標為{rotating_center}，像素坐標為{rotating_center_pixel}")

    # OpenCV 的 getRotationMatrix2D 預設 angle 為逆時針
    M = cv2.getRotationMatrix2D(center=rotating_center_pixel, angle=-angle, scale=1)
    height, width = image.shape[0], image.shape[1]
    # 根據旋轉矩陣計算旋轉後的影像尺寸
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(height * abs_sin + width * abs_cos)
    new_h = int(height * abs_cos + width * abs_sin)

    # 調整旋轉矩陣中的平移項，使旋轉後影像置中
    M[0, 2] += (new_w / 2) - rotating_center_pixel[0]
    M[1, 2] += (new_h / 2) - rotating_center_pixel[1]

    rotated = np.zeros(
        (new_h, new_w, image.shape[2], image.shape[3]), dtype=image.dtype
    )
    for z in tqdm(range(image.shape[2])):
        slice_ = image[:, :, z, :]  # 取出該 z 層 (shape: (height, width, channels))
        # 使用 cv2.warpAffine 進行 2D 旋轉，尺寸使用新的 (new_w, new_h)
        rotated_slice = cv2.warpAffine(
            slice_,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rotated[:, :, z, :] = rotated_slice

    # 更新圖片尺寸和物理坐標中心位置
    new_shape = np.array([rotated.shape[1], rotated.shape[0], rotated.shape[2]]).astype(
        int
    )
    new_rotating_center = np.array([new_shape[0] // 2, new_shape[1] // 2, 0]).astype(
        int
    )
    new_origin = np.array([new_shape[0] // 2, new_shape[1] // 2, 0]).astype(int)

    return rotated, new_shape, new_rotating_center, new_origin


def rotate_G68_cv(image, rotating_center, angle, origin=None, precision=4):
    """
    將圖形在物理坐標的 X Y 平面上順時針旋轉指定角度。

    參數:
      image: numpy.ndarray
          4D 圖形數據，形狀為 (rows, cols, depth, channels)
      rotating_center: tuple
          旋轉之前旋轉中心的物理座標 (x, y, z)
      angle: float
          逆時針旋轉的角度（單位：度），注意是逆时针
      origin: tuple, optional
          旋轉前物理坐標原點對應的像素坐標 (x, y, z)
          若未指定，則預設使用：
             (pixel_size[0]//2, size[1] - pixel_size[1]//2, 0)
          使得物理坐標 (0, 0, 0) 對應的像素座標為 (origin[0], size[1]-origin[1], 0)

    返回:
      rotated: 旋轉後的圖像。
      new_shape: rotated的形狀
      new_center: 旋轉後，物理坐標原點對應的刑訴坐標
    """

    # 注意G68逆时针为正，而rotate_Z_axis顺时针为正，所以这里取负值
    rotated, new_shape, new_rotating_center, new_origin = rotate_Z_axis(
        image, rotating_center, -angle, origin=origin, precision=precision
    )

    print(
        f"| 在XY平面进行G68旋转，绕中心点{rotating_center}，顺时针旋转{angle}角度，新的图像尺寸为{new_shape}，物理坐标中心的像素坐标为{new_origin}"
    )

    return rotated, new_shape, new_rotating_center, new_origin
