import numpy as np

# import cupynumeric as np
from datetime import datetime
from viztracer import log_sparse

from cnc_genai.src.simulation.utils import (
    get_smart_tracer,
)
from cnc_genai.src.simulation.colors import (
    CUTTING_MASK_COLOR,
    MATERIAL_COLOR,
    MATERIAL_MASK_COLOR,
)


@log_sparse(stack_depth=4)
def identify_is_valid(
    img, mask, mask_range, verbose=False, binary=False, z_slack=0, precision=4
):
    """
    判斷是否空切，即刀具經過的像素點有沒有材料。

    参数:
    img: 四维numpy数组，表示原始图像。
    mask: 三维numpy数组，表示刀具的遮罩，用于指示刀具经过的区域。
    mask_range: 列表或元组，包含六个整数，用于定义图像中需要检查的区域。
    verbose: 布尔值，用于控制是否打印详细信息。
    binary: 布尔值，是否为二进制图像模式。

    返回值:
    bool: 如果刀具经过的区域中有材料，则返回True，否则返回False。
    cnt: 整数，表示刀具经过的区域中命中的材料像素点数量。
    """
    if binary:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,1] for binary)"
        assert img.shape[3] == 1, "img should be 4D ([H,W,D,1] for binary)"
    else:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,3] for RGB)"
        assert img.shape[3] == 3, "img should be 4D ([H,W,D,3] for RGB)"
    assert len(mask.shape) == 3, "mask should be 3D ([H,W,D])"

    with get_smart_tracer().log_event("copy image by mask"):
        # 根据mask_range裁剪图像的指定区域
        img_copy = img[
            mask_range[0] : mask_range[1],  # y
            mask_range[2] : mask_range[3],  # x
            mask_range[4] : mask_range[5],  # z
        ].copy()

        # 确保裁剪后的图像区域与mask的尺寸匹配
        assert img_copy.shape[:3] == mask.shape

    z_slack_pixel = int(z_slack * 10 ** (precision - 3))

    with get_smart_tracer().log_event("counting hit"):
        # 根據圖像格式選擇正確的材料顏色和比較方式
        if binary:
            cnt = np.sum((img_copy[..., 0] * mask) == MATERIAL_MASK_COLOR)
        else:
            cnt = np.sum(
                np.all(img_copy * mask[..., np.newaxis] == MATERIAL_COLOR, axis=-1)
            )

    with get_smart_tracer().log_event("is_z_finishing"):
        if z_slack_pixel > 0:
            z_slice = slice(z_slack_pixel + 1)
            if binary:
                is_z_finishing = int(
                    np.sum(
                        (img_copy[:, :, z_slice, 0] * mask[:, :, z_slice])
                        == MATERIAL_MASK_COLOR
                    )
                    > 0
                )
            else:
                is_z_finishing = int(
                    np.sum(
                        np.all(
                            img_copy[:, :, z_slice]
                            * mask[..., np.newaxis][:, :, z_slice]
                            == MATERIAL_COLOR,
                            axis=-1,
                        )
                    )
                    > 0
                )
        else:
            is_z_finishing = None

    # 如果verbose为True，则打印详细信息
    if verbose:
        # print(f"---判斷空切 ({datetime.now() - now})---")
        if cnt == 0:
            print(f"| 【空切!!!】刀具經過{np.sum(mask)}個像素點，未命中材料像素點")
        else:
            print(f"| 刀具經過{np.sum(mask)}個像素點，命中材料{cnt}個像素點")

    # 返回是否命中材料以及命中的像素点数量
    return bool(cnt > 0), cnt, is_z_finishing


@log_sparse(stack_depth=4)
def calculate_ap(img, mask, mask_range, verbose=False, binary=False):
    """
    計算切深相關指標。

    參數:
    img (numpy.ndarray): 輸入的四維圖像數據 [H,W,D,C]，C=1(binary) 或 C=3(RGB)。
    mask (numpy.ndarray): 對應於圖像的三維遮罩數據 [H,W,D]。
    mask_range (tuple): 定義圖像中需要處理的範圍，格式為 (x_start, x_end, y_start, y_end, z_start, z_end)。
    verbose (bool, 選擇性): 是否打印詳細的計算過程和結果。預設為 False。
    binary (bool, 選擇性): 是否為二進制圖像模式。預設為 False。

    回傳:
    dict: 包含以下鍵值對的字典：
        - ap_max_over_xy (float): 在 xy 平面上的最大切深。
        - ap_sum_voxel (float): 所有切割體素數量。
        - path_area_xy (float): 在 xy 平面上的路徑面積。
        - hit_area_xy (float): 在 xy 平面上的擊中面積。
        - ap_avg_over_hit (float): 擊中材料面積的平均切深。
        - ap_avg_over_path (float): 路徑面積的平均切深。
        - hit_ratio_xy (float): 在 xy 平面上的擊中材料的面積比率。
    """
    # 添加圖像格式斷言
    if binary:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,1] for binary)"
        assert img.shape[3] == 1, "img should be 4D ([H,W,D,1] for binary)"
    else:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,3] for RGB)"
        assert img.shape[3] == 3, "img should be 4D ([H,W,D,3] for RGB)"
    assert len(mask.shape) == 3, "mask should be 3D ([H,W,D])"

    now = datetime.now()

    with get_smart_tracer().log_event("copy image by mask"):
        img_copy = img[
            mask_range[0] : mask_range[1],
            mask_range[2] : mask_range[3],
            mask_range[4] : mask_range[5],
        ].copy()
        assert img_copy.shape[:3] == mask.shape

    with get_smart_tracer().log_event("cal projection"):
        # 根據圖像格式選擇正確的材料顏色和比較方式
        if binary:
            # 4D 圖像 (H,W,D,1) - Binary 格式
            projection = np.sum(
                ((img_copy[..., 0] * mask) == MATERIAL_MASK_COLOR).astype(int),
                axis=-1,
            )
        else:
            # 4D 圖像 (H,W,D,3) - RGB 格式
            projection = np.sum(
                np.all(
                    img_copy * mask[..., np.newaxis] == MATERIAL_COLOR, axis=-1
                ).astype(int),
                axis=-1,
            )

    with get_smart_tracer().log_event("cal mask projection"):
        mask_projection = np.sum((mask == CUTTING_MASK_COLOR).astype(int), axis=-1)
        assert projection.shape == mask_projection.shape

    out_dict = {
        "ap_max_over_xy": np.max(projection),
        "ap_sum_voxel": np.sum(projection),
        "path_area_xy": np.sum(mask_projection > 0),
        "hit_area_xy": np.sum(projection > 0),
        "ap_avg_over_hit": np.sum(projection) / np.sum(projection > 0),
        "ap_avg_over_path": np.sum(projection) / np.sum(mask_projection > 0),
        "hit_ratio_xy": np.sum(projection > 0) / np.sum(mask_projection > 0),
    }

    if verbose:
        print(f"---判斷空切 ({datetime.now() - now})---")
        for k, v in out_dict.items():
            print(k, v)
    return out_dict


@log_sparse(stack_depth=4)
def calculate_ae(img, mask, mask_range, verbose=False, binary=False):
    """
    計算切寬相關指標。

    參數:
    img (numpy.ndarray): 輸入的四維圖像數據 [H,W,D,C]，C=1(binary) 或 C=3(RGB)。
    mask (numpy.ndarray): 對應於圖像的三維遮罩數據 [H,W,D]。
    mask_range (tuple): 定義圖像中需要處理的範圍，格式為 (x_start, x_end, y_start, y_end, z_start, z_end)。
    verbose (bool, 選擇性): 是否打印詳細的計算過程和結果。預設為 False。
    binary (bool, 選擇性): 是否為二進制圖像模式。預設為 False。

    回傳:
    dict: 包含以下鍵值對的字典：
        - ae_max_over_z (float): 在 z 軸上的最大切寬。
        - ae_sum_voxel (float): 所有切割體素數量。
        - path_area_z (float): 在 z 軸上的路徑長度。
        - hit_area_z (float): 在 z 軸上的擊中材料長度。
        - ae_avg_over_hit (float): 擊中材料長度的平均切寬。
        - ae_avg_over_path (float): 路徑長度的平均切寬。
        - hit_ratio_z (float): 在 z 軸上的擊中材料的長度比率。
    """
    # 添加圖像格式斷言
    if binary:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,1] for binary)"
        assert img.shape[3] == 1, "img should be 4D ([H,W,D,1] for binary)"
    else:
        assert len(img.shape) == 4, "img should be 4D ([H,W,D,3] for RGB)"
        assert img.shape[3] == 3, "img should be 4D ([H,W,D,3] for RGB)"
    assert len(mask.shape) == 3, "mask should be 3D ([H,W,D])"

    now = datetime.now()

    with get_smart_tracer().log_event("copy image by mask"):
        img_copy = img[
            mask_range[0] : mask_range[1],
            mask_range[2] : mask_range[3],
            mask_range[4] : mask_range[5],
        ].copy()
        assert img_copy.shape[:3] == mask.shape

    with get_smart_tracer().log_event("cal projection"):
        # 根據圖像格式選擇正確的材料顏色和比較方式
        if binary:
            # 4D 圖像 (H,W,D,1) - Binary 格式
            projection = np.sum(
                ((img_copy[..., 0] * mask) == MATERIAL_MASK_COLOR).astype(int),
                axis=(0, 1),
            )
        else:
            # 4D 圖像 (H,W,D,3) - RGB 格式
            projection = np.sum(
                np.all(
                    img_copy * mask[..., np.newaxis] == MATERIAL_COLOR, axis=-1
                ).astype(int),
                axis=(0, 1),
            )

    with get_smart_tracer().log_event("cal mask projection"):
        mask_projection = np.sum((mask == CUTTING_MASK_COLOR).astype(int), axis=(0, 1))
        assert projection.shape == mask_projection.shape

    out_dict = {
        "ae_max_over_z": np.max(projection),
        "ae_sum_voxel": np.sum(projection),
        "path_area_z": np.sum(mask_projection > 0),
        "hit_area_z": np.sum(projection > 0),
        "ae_avg_over_hit": np.sum(projection) / np.sum(projection > 0),
        "ae_avg_over_path": np.sum(projection) / np.sum(mask_projection > 0),
        "hit_ratio_z": np.sum(projection > 0) / np.sum(mask_projection > 0),
    }

    if verbose:
        print(f"---判斷空切 ({datetime.now() - now})---")
        for k, v in out_dict.items():
            print(k, v)
    return out_dict
