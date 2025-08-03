from typing import Optional
import logging
from datetime import datetime
import cupy
from cnc_genai.src.simulation.colors import (
    CUTTING_MASK_COLOR,
    MATERIAL_COLOR,
)


def calculate_metric_gpu(img, mask, mask_range, mask_slack, mask_range_slack, *, logger: Optional[logging.Logger] = None):
    """
    根據產品和切割路徑的voxel表示, 計算該路徑的多種負載指標; GPU加速版本

    Args:
        img: product voxels array of shape [X,Y,Z,3]
        mask: path voxels array of shape [MX, MY, MZ]; MX < X (Y,Z同理)
        mask_range: the edge of `mask` in `img`, e.g. [min_MX, max_MX, min_MY, max_MY]
        mask_slack: 同`mask`, 但路徑包括容忍誤差
        mask_range_slack: 同`mask_range`, the edge of `mask_range_slack`

    Return:
        dict of all metrics
    """

    def _log(*args):
        message = " ".join(map(str, args))
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    current_time = datetime.now()
    cu_material_color = cupy.array(MATERIAL_COLOR)

    # refer to `identifty_is_valid`
    img_view = img[
        mask_range[0] : mask_range[1],
        mask_range[2] : mask_range[3],
        mask_range[4] : mask_range[5],
    ]
    hit_voxels = cupy.all(
        img_view * cupy.expand_dims(mask, axis=-1) == cu_material_color, axis=-1
    )
    hit_count = cupy.asnumpy(cupy.sum(hit_voxels)).item()
    cutting_count = cupy.asnumpy(cupy.sum(mask)).item()

    if hit_count == 0:
        _log(f"GPU| 【空切!!!】刀具經過{cutting_count}個像素點，未命中材料像素點")
    else:
        _log(f"GPU| 刀具經過{cutting_count}個像素點，命中材料{hit_count}個像素點")

    metric1 = {
        "is_valid": bool(hit_count > 0),
        "hit_area": hit_count,
        "cutting_area:": cutting_count,
    }

    metric1_1 = {}
    if mask_slack is not None:
        _log("tile_slack is not None, do slack metric calculation")
        # refer to `identifty_is_valid` (but use slack mask)
        img_view = img[
            mask_range_slack[0] : mask_range_slack[1],
            mask_range_slack[2] : mask_range_slack[3],
            mask_range_slack[4] : mask_range_slack[5],
        ]
        hit_voxels_slack = cupy.all(
            img_view * cupy.expand_dims(mask_slack, axis=-1) == cu_material_color, axis=-1
        )
        hit_count_slack = cupy.asnumpy(cupy.sum(hit_voxels_slack)).item()
        cutting_count_slack = cupy.asnumpy(cupy.sum(mask_slack)).item()

        if hit_count_slack == 0:
            _log(f"| GPU:【空切!!!】刀具經過{cutting_count_slack}個像素點，未命中材料像素點")
        else:
            _log(f"| GPU: 刀具經過{cutting_count_slack}個像素點，命中材料{hit_count_slack}個像素點")

        metric1_1 = {
            "is_valid_slack": bool(hit_count_slack > 0),
            "hit_area_slack": hit_count_slack,
        }

    # refer to `calculate_ap`
    projection = cupy.sum(hit_voxels.astype(int), axis=-1)
    mask_projection = cupy.sum((mask == CUTTING_MASK_COLOR).astype(int), axis=-1)

    ap_max_over_xy = cupy.asnumpy(cupy.max(projection))
    ap_sum_voxel = cupy.asnumpy(cupy.sum(projection))
    path_area_xy = cupy.asnumpy(cupy.sum(mask_projection > 0))
    hit_area_xy = cupy.asnumpy(cupy.sum(projection > 0))
    metric2 = {
        "ap_max_over_xy": ap_max_over_xy,
        "ap_sum_voxel": ap_sum_voxel,
        "path_area_xy": path_area_xy,
        "hit_area_xy": hit_area_xy,
        "ap_avg_over_hit": ap_sum_voxel / hit_area_xy,
        "ap_avg_over_path": ap_sum_voxel / path_area_xy,
        "hit_ratio_xy": hit_area_xy / path_area_xy,
    }

    # debug
    for k, v in metric2.items():
        #logging.info(f"{k}={v}")
        _log(f"{k}={v}")


    # refer to `calculate_ae`
    projection = cupy.sum(hit_voxels.astype(int), axis=(0, 1))
    mask_projection = cupy.sum((mask == CUTTING_MASK_COLOR).astype(int), axis=(0, 1))

    ae_max_over_z = cupy.asnumpy(cupy.max(projection))
    ae_sum_voxel = cupy.asnumpy(cupy.sum(projection))
    path_area_z = cupy.asnumpy(cupy.sum(mask_projection > 0))
    hit_area_z = cupy.asnumpy(cupy.sum(projection > 0))
    metric3 = {
        "ae_max_over_z": ae_max_over_z,
        "ae_sum_voxel": ae_sum_voxel,
        "path_area_z": path_area_z,
        "hit_area_z": hit_area_z,
        "ae_avg_over_hit": ae_sum_voxel / hit_area_z,
        "ae_avg_over_path": ae_sum_voxel / path_area_z,
        "hit_ratio_z": hit_area_z / path_area_z,
    }
    for k, v in metric3.items():
        #logging.info(f"{k}={v}")
        _log(f"{k}={v}")

    _log(f"calculate_mteric_gpu cost {datetime.now() - current_time}")
    return {**metric1, **metric1_1, **metric2, **metric3}
