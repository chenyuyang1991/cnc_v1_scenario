import math
import os
import re
import logging
import threading
import signal
import json
import ast
import numpy as np

# import cupynumeric as np
import pandas as pd
from datetime import datetime
import warnings
import argparse
import glob
import shutil
import sys
from pathlib import Path
from viztracer import (
    log_sparse,
)

from cnc_genai.src.simulation.cutting_new import draw_G01_cv, draw_G02_cv, draw_G03_cv
from cnc_genai.src.simulation.calculate import (
    identify_is_valid,
    calculate_ap,
    calculate_ae,
)
from cnc_genai.src.simulation.utils import (
    display_recent_cutting,
    update_image,
    convert_stl_to_numpy,
    convert_stp_to_numpy,
    get_smart_tracer,
    SimulationDataAppender,
)
from cnc_genai.src.simulation.colors import (
    get_step_color,
    CUTTING_COLOR,
    MATERIAL_COLOR,
    EMPTY_COLOR,
)
from cnc_genai.src.simulation.utils import load_from_zst, save_to_zst


def configure_logging(log_file="simulation.log"):
    logging.basicConfig(
        level=logging.DEBUG,  # 設定最低日誌等級
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 將日誌輸出到檔案
            logging.StreamHandler(),  # 同時輸出到終端機（可選）
        ],
    )


keyboard_interrupt_evnt = threading.Event()


class IndexedQueue:
    def __init__(self, maxsize, expect_last):
        self.queue = {}
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.next_index = 0
        self.maxsize = maxsize
        # self.expect_total = expect_total
        self.expect_last = expect_last

    def last_id(self):
        return self.expect_last

    @log_sparse(stack_depth=2)
    def put(self, obj, index):
        """
        When index == None, it means the last one (END obj).
        """
        with self.lock:
            while len(self.queue) >= self.maxsize or (
                (self.maxsize - len(self.queue)) <= 1 and index != self.next_index
            ):
                if keyboard_interrupt_evnt.is_set():
                    raise RuntimeError("keyboard interrupt detected")

                print(
                    f"thread-{threading.get_native_id()}: IndexedQueue put wait", index
                )
                self.condition.wait(timeout=0.5)  # Block until space is available
            print(f"thread-{threading.get_native_id()}: IndexedQueue put", index)
            self.queue[index] = obj
            self.condition.notify_all()  # Notify waiting threads

    # def put_end(self):
    #    with self.lock:
    #        # forbidding END to occupy the last slot, so that unfinished work items have space to enqueue.
    #        while len(self.queue) >= self.maxsize or ((self.maxsize - len(self.queue)) <= 1):
    #            if keyboard_interrupt_evnt.is_set():
    #                raise RuntimeError("keyboard interrupt detected")

    #        print(f"put_end at index={self.expect_total}")
    #        self.queue[self.expect_total] = None
    #        self.condition.notify_all()  # Notify waiting threads

    @log_sparse(stack_depth=2)
    def get(self):
        """
        Return obj and its index.
        """
        with self.lock:
            while self.next_index not in self.queue:
                if keyboard_interrupt_evnt.is_set():
                    raise RuntimeError("keyboard interrupt detected")

                print(
                    f"thread-{threading.get_native_id()}: IndexedQueue get wait",
                    self.next_index,
                )
                self.condition.wait(
                    timeout=0.5
                )  # Block until the expected index is available
            print(
                f"thread-{threading.get_native_id()}: IndexedQueue get", self.next_index
            )
            obj = self.queue.pop(self.next_index)
            ret_index = self.next_index
            self.next_index += 1
            self.condition.notify_all()  # Notify waiting threads
            return obj, ret_index


class ThreadSafeIterator:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the iterator with a pandas DataFrame.
        :param dataframe: The source DataFrame.
        """
        self.dataframe = dataframe
        self.index = 0
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        return self

    @log_sparse(stack_depth=2)
    def __next__(self):
        """
        Retrieve the next row from the DataFrame in a thread-safe manner.
        :return: A copy of the next row as a DataFrame.
        """
        print(f"thread-{threading.get_native_id()}: wait for iterator")
        with self.lock:
            if self.index >= len(self.dataframe):
                raise StopIteration
            row = self.dataframe.iloc[self.index].copy()
            self.index += 1
        print(f"thread-{threading.get_native_id()}: return from iterator")
        return row, self.index  # return row and "number of  rows have been output"


def setup_logger(name, log_file, level=logging.NOTSET):
    """Setup a logger with a specific name and log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent propagation to the root logger
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()  # Clear existing handlers
    logger.addHandler(file_handler)
    return logger


def run(
    df: pd.DataFrame,
    sub_program: str,
    tools_df: pd.DataFrame,
    image: np.ndarray,  # 毛坯
    origin: np.ndarray,  # 坐标系原点
    precision: int = 4,
    verbose: bool = False,
    r_slack: float = 0,
    z_slack: float = 0,
    excel_out_path: str = None,
    use_gpu: bool = False,
    early_stop: int = None,
    log_path: str = None,
    turbo_spiral: bool = False,  # 快速模式，相同坐標不仿真，改為相近坐標不仿真（XYZ diff 均小於0.1mm）用以避免大量的螺旋下刀
    binary: bool = False,  # 是否使用二進制圖像格式
) -> np.ndarray:

    print("start subprogram:", sub_program)
    assert (
        len(image.shape) == 4
    ), "image should be 4D ([H,W,D,1] for binary, [H,W,D,3] for RGB)"
    if binary:
        assert image.shape[3] == 1, "Binary mode is not supported for RGB stock files"
        print("Running in BINARY mode - 使用二進制圖像格式，顏色功能將被簡化")
    else:
        assert image.shape[3] == 3, "RGB mode is not supported for Binary stock files"
        print("Running in RGB mode - 使用RGB圖像格式，顏色功能將被保留")
    if use_gpu:
        import cupy
        from cnc_genai.src.simulation.calculate_gpu import calculate_metric_gpu

        cupy_mempool = cupy.get_default_memory_pool()

    # 篩選出子程序代碼行
    df["sub_program"] = df["sub_program"].astype(str).str.zfill(4)
    df = df[df["sub_program"] == sub_program]

    # 假設初始化刀頭位置
    df["X"] = df["X"].fillna(0)
    df["Y"] = df["Y"].fillna(0)
    df["Z"] = df["Z"].fillna(99999)  # 无限高的初始抬刀距离

    df["X_prev"] = df["X"].shift(1).fillna(0)
    df["Y_prev"] = df["Y"].shift(1).fillna(0)
    df["Z_prev"] = df["Z"].shift(1).fillna(99999)

    # os.makedirs(f"../cnc_intermediate/parsed_code/", exist_ok=True)
    # df.to_excel(f"../cnc_intermediate/parsed_code/{sub_program}.xlsx", index=False)

    print(f"--子程序有{df.shape[0]}行代碼")

    # 檢查篩選後的 DataFrame 是否為空
    if df.shape[0] == 0:
        print(f"[WARNING] 篩選後的子程序 '{sub_program}' 沒有任何代碼行！")
        print("可能的原因：")
        print("1. sub_program 參數與 DataFrame 中的值不匹配")
        print("2. 原始 DataFrame 就是空的")
        print("3. 數據質量問題")
        # 返回原始圖像，因為沒有任何操作需要執行
        return image

    os.makedirs(excel_out_path, exist_ok=True)

    # 創建與原圖同樣大小的掩碼，加速運算
    # pixel_size = np.array([image.shape[1], image.shape[0], image.shape[2]]).astype(int)
    # mask = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2]), np.uint8)

    pixel_size = image.shape[:3]
    # mask = np.zeros(tuple(pixel_size), np.uint8)

    def stage_1_thread(
        df_iterator: ThreadSafeIterator,
        tools_df: pd.DataFrame,  # TODO: use closure?
        precision: int,  # TODO: use closure
        r_slack: float,  # TODO: use closure
        z_slack: float,  # TODO: use closure
        cutting_mask_queue: IndexedQueue,
        # mask: np.ndarray,       # TODO: use closure
        stock_size: tuple,  # TODO: use closure
        origin: np.ndarray,
        use_gpu: bool,
        turbo_spiral: bool,
        verbose: bool,
        binary: bool,  # 是否使用二進制圖像格式
        *,
        logger=None,
    ):

        def _log(*args):
            message = " ".join(map(str, args))
            if logger is not None:
                logger.info(message)
            else:
                print(message)

        if use_gpu:
            _log("WARNING: Cutting Phase has no GPU impl for now. Fall back to CPU.")

        step = 0

        # Initialize an empty mask
        # TODO: allocate later to reduce footprint, depends on toolpath length
        print("[INFO] building mask")
        mask = np.zeros(tuple(stock_size), dtype=np.uint8)
        print("[INFO] mask built")

        _log(f"Stage 1 thread-{threading.get_native_id()} launched...")
        # for idx, row in df.iterrows():
        count = 0  # 初始化 count，防止迴圈未執行時的 UnboundLocalError
        for row, count in df_iterator:

            idx = row["row_id"]

            if stop_event.is_set():  # Check if stop_event is set
                _log(f"Stage 1 thread-{threading.get_native_id()} stopping...")
                break

            _log("-" * 20)
            _log(f"{row['sub_program']}: 第{row['row_id']}行, {row['src']}")
            _log(
                f"代碼行進度 - {((int(row['row_id']) + 1) / (df.shape[0] + 1)) * 100:.2f}%"
            )

            tracer_casename = "DefualtUnKnown"

            # First Stage: prepare data and draw mask
            with get_smart_tracer().log_event("each nc code initial"):

                # 獲取當前刀具信息
                tool = row["T"]

                # 1. 如果尚未換刀，則不仿真
                if pd.isna(tool):
                    cutting_mask_queue.put(
                        {
                            "cut": None,
                            "row": row,
                            "status": "no_tool_number",
                            "tracer_casename": tracer_casename,
                        },
                        idx,
                    )
                    continue

                # 如果已經換好刀，則獲取刀具信息
                tool_info = tools_df.loc[tools_df["刀號"] == tool, "規格型號"]
                if len(tool_info):
                    tool_d = float(
                        tools_df.loc[tools_df["刀號"] == tool, "刀頭直徑"].values[0]
                    )
                    # TO REMOVE for test
                    # tool_d = max(
                    #     2, math.ceil(tool_d / 2) * 2
                    # )  # 確保刀具直徑至少為2mm，ceiling(tool_d/2)*2
                    tool_h = float(
                        tools_df.loc[tools_df["刀號"] == tool, "刀頭高度"].values[0]
                    )
                    # tool_d = round(tool_d * 10 ** (precision - 3))
                    # tool_h = round(tool_h * 10 ** (precision - 3))
                    tool_d_slack = tool_d + 2 * r_slack
                else:
                    # 2.匹配不到刀具表
                    print(
                        f"[ERROR] tool_d or tool_h is 0: {tool} {tools_df['刀號']} mismatched"
                    )
                    cutting_mask_queue.put(
                        {
                            "cut": None,
                            "row": row,
                            "status": "tool_number_not_matched",
                            "tracer_casename": tracer_casename,
                        },
                        idx,
                    )
                    continue

                row["tool_diameter"] = round(tool_d * 10 ** (precision - 3))
                row["tool_height"] = round(tool_h * 10 ** (precision - 3))
                row["tool_diameter_mm"] = tool_d
                row["tool_height_mm"] = tool_h
                row["tool_r_slack_mm"] = r_slack
                row["tool_z_slack_mm"] = z_slack
                # row["sub_program_step"] = step
                # row["global_step"] = idx

                step += 1

                # 3. 如果刀具Z位于起始高度，则不仿真
                if row["Z"] == 99999:
                    cutting_mask_queue.put(
                        {
                            "cut": None,
                            "row": row,
                            "status": "tool_at_initial_height",
                            "tracer_casename": tracer_casename,
                        },
                        idx,
                    )
                    continue

                # 4. 如果半径大于半米，则不仿真
                elif (
                    row["move_code"] in ["G02", "G03"]
                    and max(abs(row["I"]), abs(row["J"])) > 500
                ):
                    cutting_mask_queue.put(
                        {
                            "cut": None,
                            "row": row,
                            "status": "radius_too_large",
                            "tracer_casename": tracer_casename,
                        },
                        idx,
                    )
                    continue

                else:

                    # 從物理坐標轉換為像素坐標
                    start_point = np.array(
                        [row["X_prev"], row["Y_prev"], row["Z_prev"]]
                    )
                    end_point = np.array([row["X"], row["Y"], row["Z"]])
                    _log(start_point, "->", end_point)

                    # 轉換坐標
                    # 制工实际操作O8555，先转四轴(X)再转0.5轴(Z')，Z'轴为旋转后的Z轴，即工件垂直方向，所以两步旋转等价于先转Z轴再转X轴
                    # 因此对应的，刀具的逆操作应该先转X，再转Z
                    reverse_rotation_order = ["Y", "X", "Z"]

                    # 旧版：只考虑一套YXZ的旋转
                    # rotating_centers = np.array(
                    #     [
                    #         np.array(
                    #             [float(k) for k in row[f"rotate_{x}_center"].split("/")]
                    #         )
                    #         for x in reverse_rotation_order
                    #         if pd.notna(row[f"rotate_{x}_angle"])
                    #     ]
                    # )
                    # rotating_angles = np.array(
                    #     [
                    #         (
                    #             -float(row[f"rotate_{x}_angle"])
                    #             if x == "X"
                    #             else float(row[f"rotate_{x}_angle"])
                    #         )
                    #         for x in reverse_rotation_order
                    #         if pd.notna(row[f"rotate_{x}_angle"])
                    #     ]
                    # )
                    # rotating_axes = np.array(
                    #     [
                    #         x
                    #         for x in reverse_rotation_order
                    #         if row[f"rotate_{x}_axis"] is True
                    #     ]
                    # )

                    # 新版：考虑多套YXZ的旋转，IJK的旋转顺序是否为YXZ暂不确定，先开发
                    # IJK的旋转方向哪边为正，需要clarify

                    # 檢查 rotation 欄位是否為有效的 JSON 資料
                    try:
                        rotation_str = str(row["rotation"])
                        if pd.isna(row["rotation"]) or rotation_str.lower() in [
                            "nan",
                            "none",
                            "",
                        ]:
                            # 如果 rotation 欄位是空的或無效的，使用空列表
                            rotation_actions = []
                        else:
                            rotation_actions = ast.literal_eval(rotation_str)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(
                            f"[ERROR] 無法解析 rotation 欄位 '{row['rotation']}'，使用空列表。錯誤: {e}"
                        )
                        rotation_actions = []

                    rotating_centers = []
                    rotating_angles = []
                    rotating_axes = []

                    # 此处都已修正为绕XYZ轴方向逆时针旋转为正 ，如果实际定义不符则需要修改
                    # 即假想刀具绕XYZ轴顺时针旋转为正，这与multiple_rotated_physical_to_pixel保持一致
                    # 请支持没有XYZ任意key的情况，即没有旋转，无需append
                    print(["INFO"], rotation_actions)
                    for action in reversed(rotation_actions):
                        rotating_centers += [
                            np.array([float(k) for k in action[action_axis]["center"]])
                            for action_axis in reverse_rotation_order
                            if action_axis in action.keys()
                        ]
                        rotating_angles += [
                            float(action[action_axis]["angle"])
                            for action_axis in reverse_rotation_order
                            if action_axis in action.keys()
                        ]
                        rotating_axes += [
                            action_axis
                            for action_axis in reverse_rotation_order
                            if action_axis in action.keys()
                        ]

                    # 修改z位置以考慮z_slack
                    if z_slack > 0:
                        start_point_z_slack = start_point - np.array([0, 0, z_slack])
                        end_point_z_slack = end_point - np.array([0, 0, z_slack])
                    else:
                        start_point_z_slack = start_point
                        end_point_z_slack = end_point

            # 开始切割
            if row["move_code"] in ["G01", "G02", "G03", "G81", "G82", "G83", "G84"]:

                # 5. 如果起止點物理坐標相同，則不仿真
                if np.array_equal(start_point, end_point):
                    _log("【起止點物理坐標相同，不仿真】")
                    row["same_start_end_pixel"] = True
                    cutting_mask_queue.put(
                        {
                            "cut": None,
                            "row": row,
                            "status": "identical_start_end_physical",
                            "tracer_casename": tracer_casename,
                        },
                        idx,
                    )
                    continue

                else:
                    if len(re.findall(r"(G0[1-3])", row["src"])) > 1:
                        raise ValueError("G01|G02|G03 appeared in same line")
                    else:
                        current_time = datetime.now()
                        casename = "OK"

                        if row["move_code"] == "G01":
                            tile, tile_range, row, casename = draw_G01_cv(
                                mask,
                                row,
                                start_point,
                                end_point,
                                tool_d / 2,
                                tool_h,
                                rotating_angles=rotating_angles,
                                rotating_centers=rotating_centers,
                                rotating_axes=rotating_axes,
                                origin=origin,
                                precision=precision,
                                use_gpu=use_gpu,
                                verbose=verbose,
                                turbo_spiral=turbo_spiral,
                                binary=binary,  # 傳遞binary參數
                                logger=logger,
                            )

                            if r_slack + z_slack > 0:
                                tile_slack, tile_range_slack, _, _ = draw_G01_cv(
                                    mask,
                                    row,
                                    start_point_z_slack,
                                    end_point_z_slack,
                                    tool_d_slack / 2,
                                    tool_h + z_slack,
                                    rotating_angles=rotating_angles,
                                    rotating_centers=rotating_centers,
                                    rotating_axes=rotating_axes,
                                    origin=origin,
                                    precision=precision,
                                    use_gpu=False,
                                    verbose=verbose,
                                    turbo_spiral=turbo_spiral,
                                    binary=binary,  # 傳遞binary參數
                                    logger=logger,
                                )
                            else:
                                tile_slack, tile_range_slack = tile, tile_range
                        elif row["move_code"] == "G02":
                            rel_circle_center = np.array(
                                [row.get("I", 0), row.get("J", 0), 0]
                            ).astype(np.float64)
                            tile, tile_range, row, casename = draw_G02_cv(
                                mask,
                                row,
                                start_point,
                                end_point,
                                rel_circle_center,
                                tool_d / 2,
                                tool_h,
                                rotating_angles=rotating_angles,
                                rotating_centers=rotating_centers,
                                rotating_axes=rotating_axes,
                                origin=origin,
                                precision=precision,
                                use_gpu=use_gpu,
                                verbose=verbose,
                                binary=binary,  # 傳遞binary參數
                            )
                            if r_slack + z_slack > 0:
                                tile_slack, tile_range_slack, _, _ = draw_G02_cv(
                                    mask,
                                    row,
                                    start_point_z_slack,
                                    end_point_z_slack,
                                    rel_circle_center,
                                    tool_d_slack / 2,
                                    tool_h + z_slack,
                                    rotating_angles=rotating_angles,
                                    rotating_centers=rotating_centers,
                                    rotating_axes=rotating_axes,
                                    origin=origin,
                                    precision=precision,
                                    use_gpu=use_gpu,
                                    verbose=verbose,
                                    binary=binary,  # 傳遞binary參數
                                )
                            else:
                                tile_slack, tile_range_slack = tile, tile_range
                        elif row["move_code"] == "G03":
                            rel_circle_center = np.array(
                                [row.get("I", 0), row.get("J", 0), 0]
                            ).astype(np.float64)
                            tile, tile_range, row, casename = draw_G03_cv(
                                mask,
                                row,
                                start_point,
                                end_point,
                                rel_circle_center,
                                tool_d / 2,
                                tool_h,
                                rotating_angles=rotating_angles,
                                rotating_centers=rotating_centers,
                                rotating_axes=rotating_axes,
                                origin=origin,
                                precision=precision,
                                use_gpu=use_gpu,
                                verbose=verbose,
                                binary=binary,  # 傳遞binary參數
                            )
                            if r_slack + z_slack > 0:
                                tile_slack, tile_range_slack, _, _ = draw_G03_cv(
                                    mask,
                                    row,
                                    start_point_z_slack,
                                    end_point_z_slack,
                                    rel_circle_center,
                                    tool_d_slack / 2,
                                    tool_h + z_slack,
                                    rotating_angles=rotating_angles,
                                    rotating_centers=rotating_centers,
                                    rotating_axes=rotating_axes,
                                    origin=origin,
                                    precision=precision,
                                    use_gpu=use_gpu,
                                    verbose=verbose,
                                    binary=binary,  # 傳遞binary參數
                                )
                            else:
                                tile_slack, tile_range_slack = tile, tile_range
                        elif row["move_code"] in ["G81", "G82", "G83", "G84"]:

                            _log("【G81|G82|G83|G84钻孔代碼，不仿真】")
                            cutting_mask_queue.put(
                                {
                                    "cut": None,
                                    "row": row,
                                    "status": "G81|G82|G83|G84",
                                    "tracer_casename": tracer_casename,
                                },
                                idx,
                            )
                            continue

                            # # G81指令的座標存儲在標準的X,Y,Z列中
                            # deepest_point = np.array(
                            #     [row["X"], row["Y"], row[f"{row['move_code']}_Z"]]
                            # ).astype(np.float64)
                            # tile, tile_range, row, casename = draw_G01_cv(
                            #     mask,
                            #     row,
                            #     start_point,
                            #     deepest_point,
                            #     tool_d / 2,
                            #     tool_h,
                            #     rotating_angles=rotating_angles,
                            #     rotating_centers=rotating_centers,
                            #     rotating_axes=rotating_axes,
                            #     origin=origin,
                            #     precision=precision,
                            #     use_gpu=use_gpu,
                            #     verbose=verbose,
                            # )
                            # if r_slack + z_slack > 0:
                            #     deepest_point_z_slack = deepest_point + np.array(
                            #         [0, 0, z_slack]
                            #     )
                            #     tile_slack, tile_range_slack, _, _ = draw_G01_cv(
                            #         mask,
                            #         row,
                            #         start_point_z_slack,
                            #         deepest_point_z_slack,
                            #         tool_d_slack / 2,
                            #         tool_h + z_slack,
                            #         rotating_angles=rotating_angles,
                            #         rotating_centers=rotating_centers,
                            #         rotating_axes=rotating_axes,
                            #         origin=origin,
                            #         precision=precision,
                            #         use_gpu=use_gpu,
                            #         verbose=verbose,
                            #     )
                            # else:
                            #     tile_slack, tile_range_slack = tile, tile_range
                        else:
                            _log(f"不支援的指令: {row['move_code']}")
                            raise ValueError(f"不支援的指令: {row['move_code']}")

                        # for trace draw casename
                        if casename:
                            tracer_casename = casename

                        if verbose:
                            _log(f"|| 生成切割mask用時 {datetime.now() - current_time}")

                        if tile is None:
                            cutting_mask_queue.put(
                                {
                                    "cut": None,
                                    "row": row,
                                    "status": casename,
                                    "tracer_casename": tracer_casename,
                                },
                                idx,
                            )
                        else:
                            _log("tile.shape=", tile.shape)
                            cutting_mask_queue.put(
                                {
                                    "cut": (
                                        tile,
                                        tile_range,
                                        tile_slack,
                                        tile_range_slack,
                                    ),
                                    "row": row,
                                    "status": casename,
                                    "tracer_casename": tracer_casename,
                                },
                                idx,
                            )
            elif row["move_code"] in ["G68", "G68.2"]:
                # 旋转的角度计算已经在code_parsing中完成
                _log("【G68非切割代碼，不仿真】")
                cutting_mask_queue.put(
                    {
                        "cut": None,
                        "row": row,
                        "status": "G68_rotate",
                        "tracer_casename": tracer_casename,
                    },
                    idx,
                )
                continue
            else:
                _log("【非切割代碼，不仿真】")
                cutting_mask_queue.put(
                    {
                        "cut": None,
                        "row": row,
                        "status": "not_cutting_code",
                        "tracer_casename": tracer_casename,
                    },
                    idx,
                )
                continue

            _log(f"line {row['row_id']} completed!!")

        _log(f"No More Data, len(dataframe)={len(df_iterator)}, count={count}")
        # if len(df_iterator) <= count:
        #    _log(f"take last one, required to send End-of-Code")
        #    cutting_mask_queue.put_end()

    def stage_2_thread(
        cutting_mask_queue: IndexedQueue,
        image: np.ndarray,
        z_slack: float,
        use_gpu: bool,
        verbose: bool,
        ret_value: list,
        excel_out_path: str,
        sub_program: str,
        binary: bool,  # 是否使用二進制圖像格式
        # time_profile: list
        *,
        logger=None,
    ):
        assert (
            len(image.shape) == 4
        ), "image should be 4D ([H,W,D,1] for binary, [H,W,D,3] for RGB)"
        if binary:
            assert (
                image.shape[3] == 1
            ), "Binary mode is not supported for RGB stock files"
        else:
            assert (
                image.shape[3] == 3
            ), "RGB mode is not supported for Binary stock files"

        def _log(*args):
            message = " ".join(map(str, args))
            if logger is not None:
                logger.info(message)
            else:
                print(message)

        _log("debug: sum of init image=", np.sum(image))
        if use_gpu:
            image = cupy.asarray(image)

        # initial Excel Writer
        excel_appender = SimulationDataAppender(excel_out_path, sub_program)

        while not stop_event.is_set():  # Check if stop_event is set
            # Stage 2: calculate metric and update simultion image
            one_line_sim, idx_row = cutting_mask_queue.get()
            _log("get index:", idx_row)

            out_row = one_line_sim["row"]
            out_row["status"] = one_line_sim["status"]
            _log(
                f"---- SUBPROGRAM: {out_row['sub_program']}: LINE {out_row['row_id']} -----"
            )
            if one_line_sim["cut"] is None:
                _log(f"EMPTY tile")
            else:
                tile, tile_range, tile_slack, tile_range_slack = one_line_sim["cut"]
                _log(
                    f"tile.shape={tile.shape}, tile_range={tile_range}, tile_slack.shape={tile_slack.shape if tile_slack is not None else None}, tile_range_slack={tile_range_slack if tile_range_slack is not None else None}"
                )

                # 在binary模式下，簡化顏色處理
                if binary:
                    STEP_COLOR = [0]  # Binary模式使用簡單標記
                else:
                    STEP_COLOR = get_step_color(out_row["row_id"])

                current_time = datetime.now()
                if tile is None:
                    _log(f"无切割像素点")

                # phase 2 computation using CPU
                elif not use_gpu:
                    # 判斷是否空切
                    out_row["cutting_area"] = np.sum(tile)
                    out_row["is_valid"], out_row["hit_area"], _ = identify_is_valid(
                        image, tile, tile_range, verbose=verbose, binary=binary
                    )

                    if out_row["hit_area"] == 0:
                        _log(
                            f"CPU | 【空切!!!】刀具經過{out_row['cutting_area']}個像素點，未命中材料像素點"
                        )
                    else:
                        _log(
                            f"CPU | 刀具經過{out_row['cutting_area']}個像素點，命中材料{out_row['hit_area']}個像素點"
                        )

                    if tile_slack is not None:
                        (
                            out_row["is_valid_slack"],
                            out_row["hit_area_slack"],
                            out_row["hit_z_slack"],
                        ) = identify_is_valid(
                            image,
                            tile_slack,
                            tile_range_slack,
                            verbose=verbose,
                            binary=binary,
                            z_slack=z_slack,
                        )

                    # 計算切深
                    ap_res = calculate_ap(
                        image, tile, tile_range, verbose=False, binary=binary
                    )
                    for k, v in ap_res.items():
                        out_row[k] = v

                    # 計算切寬
                    ae_res = calculate_ae(
                        image, tile, tile_range, verbose=False, binary=binary
                    )
                    for k, v in ae_res.items():
                        out_row[k] = v

                    metric_time = datetime.now()
                    image, deepest_layer = update_image(
                        image, tile, tile_range, STEP_COLOR, binary=binary
                    )
                    _log(f"|| 計算切寬切深指標用時 {metric_time - current_time}")
                    _log(f"|| 真實切割用時 {datetime.now() - metric_time}")

                # phase 2 computation using GPU
                elif use_gpu:
                    _log(f"tile={type(tile)}, tile_slack={type(tile_slack)}")
                    # if isinstance(tile, np.ndarray):
                    #    _log(f"tile.shape={tile.shape}")
                    # if isinstance(tile_slack, np.ndarray):
                    #    _log(f"tile.shape={tile_slack.shape}")

                    cu_tile = cupy.asarray(tile)

                    cu_tile_slack = (
                        cupy.asarray(tile_slack) if tile_slack is not None else None
                    )

                    all_metrics = calculate_metric_gpu(
                        image,
                        cu_tile,
                        tile_range,
                        cu_tile_slack,
                        tile_range_slack,
                        logger=logger,
                    )
                    _log(
                        "cupy_memory_pool used:",
                        cupy_mempool.used_bytes() / 1024**2,
                        "MB",
                    )
                    _log(
                        "cupy_memory_pool total bytes:",
                        cupy_mempool.total_bytes() / 1024**2,
                        "MB",
                    )
                    for k, v in all_metrics.items():
                        out_row[k] = v
                    metric_time = datetime.now()
                    image, deepest_layer = update_image(
                        image,
                        cu_tile,
                        tile_range,
                        STEP_COLOR,
                        use_gpu=use_gpu,
                        binary=binary,
                    )
                    _log("after update: total=", cupy.sum(image))
                    _log(f"|| 計算切寬切深指標用時 {metric_time - current_time}")
                    _log(f"|| 真實切割用時 {datetime.now() - metric_time}")

                # 計算切割區域到材料區域的最小距離
                # if row['is_valid']:
                #     row['distance_to_material'] = 0
                # else:
                #     row['distance_to_material'] = get_distance_between_masks(
                #         image, mask, tile_range, tile
                #     )
                # print(f'|| 計算到材料的距離 {datetime.now() - current_time}')
                # current_time = datetime.now()

                # 显示
                # if verbose:
                #    if use_gpu and False:
                #        image_cpu = cupy.asnumpy(image)
                #        display_recent_cutting(image_cpu, deepest_layer, row)
                #    else:
                #        display_recent_cutting(image, deepest_layer, out_row)

                # TODO: append `out_row` to excel,

                # 記錄仿真時間
                # out_row["simulation_time_used"] = 0 # deprecated
                # out_row["debug_color"] = "/".join([str(int(x)) for x in STEP_COLOR])

            # 更新特征表
            excel_appender.append_row(out_row)

            # time_profile.append({"idx": idx, "latency": latency.total_seconds()})
            # pd.DataFrame.from_dict(df_output).to_excel(
            #    f"{excel_out_path}/{sub_program}.xlsx",
            #    index=False,
            # )
            # save_to_zst(image, f"../cnc_intermediate/simulation/simulation_output_{timestamp}/by_steps/{row['O']}_{row['row_id']}_{row['src']}.zst")

            # meet the last line -> this subprogram complete
            if idx_row == cutting_mask_queue.last_id():
                break

        # end of subprogram simulation
        result = image if not use_gpu else cupy.asnumpy(image)
        _log("debug: sum of result image=", np.sum(result))
        ret_value.append(result)

        # flush final rows
        excel_appender.close()

        #
        tracer = get_smart_tracer()
        tracer_filename = os.path.join(
            "./viztracer_logs/",
            f"{sub_program}.json",
        )
        tracer.save(output_file=tracer_filename)

    try:
        # initial VizTracer
        tracer = get_smart_tracer()
        tracer.start()

        # Create a local stop_event
        stop_event = threading.Event()

        def signal_handler(sig, frame):
            print("Ctrl-C detected! Stopping gracefully...")
            stop_event.set()
            keyboard_interrupt_evnt.set()

        signal.signal(signal.SIGINT, signal_handler)

        stg_1_threads = []
        df_iterator = ThreadSafeIterator(df)
        print(
            f"[WORKSIZE] len(df)={len(df_iterator)}, first_row_id={df['row_id'].min()}, last_row_id={df['row_id'].max()}"
        )

        sim_queue_size = os.getenv("SIM_QUEUE_SIZE")
        if sim_queue_size and sim_queue_size.isdigit():
            sim_queue_size = int(sim_queue_size)
        else:
            sim_queue_size = 10
        cutting_mask_queue = IndexedQueue(
            maxsize=sim_queue_size, expect_last=df["row_id"].max()
        )
        print(f"[QUEUESIZE] = {sim_queue_size}")

        sim_thread_num = os.getenv("SIM_THREAD_NUM")
        if sim_thread_num and sim_thread_num.isdigit():
            THREAD_NUM = int(sim_thread_num)
        else:
            THREAD_NUM = os.cpu_count() // 2

        # 根据工件尺寸计算内存，设置THREAD_NUM
        # todo

        print(f"create {THREAD_NUM} threads for stage 1: cutting")
        for i in range(THREAD_NUM):
            logger = setup_logger(
                f"CuttingLogger-{i}",
                excel_out_path + f"/stage_cutting_thread-{i}.log",
                logging.INFO,
            )
            stg_1_threads.append(
                threading.Thread(
                    target=stage_1_thread,
                    args=(
                        df_iterator,
                        tools_df,
                        precision,
                        r_slack,
                        z_slack,
                        cutting_mask_queue,
                        pixel_size,
                        origin,
                        use_gpu,
                        turbo_spiral,
                        verbose,
                        binary,  # 傳遞binary參數
                    ),
                    kwargs={"logger": logger},
                )
            )

        logger2 = setup_logger(
            "MetricLogger", excel_out_path + "/stage_metric.log", logging.INFO
        )
        return_image = []
        thread_2 = threading.Thread(
            target=stage_2_thread,
            args=(
                cutting_mask_queue,
                image,
                z_slack,
                use_gpu,
                verbose,
                return_image,
                excel_out_path,
                sub_program,
                binary,  # 傳遞binary參數
            ),
            kwargs={"logger": logger2},
        )

        for t in stg_1_threads:
            t.start()
        thread_2.start()

        for t in stg_1_threads:
            t.join()
        thread_2.join()
        print("debug: image return from thread:", np.sum(return_image[0]))

    finally:
        tracer.stop()

    return return_image[0]


def cli_run():
    sys.setswitchinterval(0.0005)
    warnings.filterwarnings("ignore")

    # 設置命令行參數
    parser = argparse.ArgumentParser(description="CNC 仿真工具")
    parser.add_argument(
        "--clamping_name", type=str, default="X2867-CNC2", help="仿真夾位名稱"
    )
    parser.add_argument(
        "--product_master",
        type=str,
        default="../app/mac1/simulation_master/X2867-CNC2/product_master.xlsx",
        help="產品主檔路徑",
    )
    parser.add_argument(
        "--parsed_nc_code",
        type=str,
        default="../app/mac1/simulation_master/X2867-CNC2/nc_code/parsed/parsed_command.xlsx",
        help="指令提取檔案路徑",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default="../app/mac1/simulation_master/X2867-CNC2/tools.xlsx",
        help="刀具檔案路徑",
    )
    parser.add_argument(
        "--stock_path",
        type=str,
        default="../app/mac1/simulation_master/X2867-CNC2/5519_shape=3763_5486_125_3_origin=0_0_0.zst",
        help="夾位毛坯路徑",
    )
    parser.add_argument("--precision", type=int, default=4, help="仿真精度")
    parser.add_argument("--resolution", type=float, default=1.0, help="解析度")
    parser.add_argument(
        "--linear_deflection", type=float, default=0.5, help="STP转STL線性偏差"
    )
    parser.add_argument(
        "--angular_deflection", type=float, default=0.2, help="STP转STL角度偏差"
    )
    parser.add_argument("--r_slack", type=float, default=0.1, help="半徑鬆弛值")
    parser.add_argument("--z_slack", type=float, default=0.1, help="Z軸鬆弛值")
    parser.add_argument(
        "--origin",
        type=float,  # 改為浮點數類型
        nargs=3,  # 接受3個數值參數
        default=[0.5, 0.5, 0],
        help="坐標原點位置 (範圍0~1認為是比例值，其他範圍認為是絕對坐標)",
    )
    parser.add_argument(
        "--numpy_out_path",
        type=str,
        default="../cnc_intermediate/simulation/mac1/X2867-CNC2/timestamp",
    )
    parser.add_argument(
        "--excel_out_path",
        type=str,
        default="../app/mac1/simulation/X2867-CNC2/simulation/latest",
    )
    parser.add_argument("--use-gpu", action="store_true", help="是否使用GPU")
    parser.add_argument(
        "--binary",
        type=bool,
        default=False,
        help="是否使用二進制圖像，二進制圖像主要針對大型工件，減少內存使用，加速仿真",
    )
    parser.add_argument("--check_point", type=str, default=None, help="斷點")
    parser.add_argument("--verbose", action="store_true", help="是否顯示詳細資訊")

    args = parser.parse_args()

    # 定義輸出路徑
    os.makedirs(args.excel_out_path, exist_ok=True)
    os.makedirs(args.numpy_out_path, exist_ok=True)

    # 開始仿真
    funcs = pd.read_excel(args.product_master)
    if "sub_program" in funcs.columns:
        pass  # 第一行作為 header
    else:
        funcs = pd.read_excel(args.product_master, header=1)  # 第二行作為 header

    funcs = funcs.drop_duplicates(["sub_program"], keep="first")
    funcs["sub_program"] = funcs["sub_program"].astype(int).astype(str).str.zfill(4)
    funcs["sub_program_last"] = funcs["sub_program"].shift(1)

    df = pd.read_excel(args.parsed_nc_code)
    df = df.drop_duplicates(["row_id", "src", "sub_program"], keep="last").reset_index()

    tools_df = pd.read_excel(args.tools)
    if "刀號" in tools_df.columns:
        pass  # 第一行作為 header
    else:
        tools_df = pd.read_excel(args.tools, header=1)  # 第二行作為 header
    tools_df = tools_df.drop_duplicates(["刀號", "規格型號"])
    precision = args.precision
    origin = np.array(args.origin)
    origin_initialized = False  # initalize in pixel range, later


    # 如果断点，找到断点在funcs的index
    if args.check_point is not None and args.check_point == "all_simulated":
        check_point_index = -1
    elif args.check_point is not None:
        check_point_index = funcs[funcs["sub_program"] == args.check_point].index[0]
    else:
        check_point_index = None

    # 開始仿真
    for idx, row in funcs.iterrows():

        # 跳過所有仔程序, 直接執行合併 all_simulated
        if check_point_index is not None and check_point_index == -1:
            break

        # 跳过断点之前的子程序
        if check_point_index is not None:
            if idx < check_point_index:
                continue

        print(f"子程序進度 - {row['sub_program_key']} ==> {(idx+1)}/{funcs.shape[0]+1}")

        if pd.isna(row["sub_program_last"]):

            if args.stock_path.endswith(".zst"):
                image, _ = load_from_zst(args.stock_path)  # 使用傳入的origin
            elif args.stock_path.endswith(".stl"):
                print(
                    f"成功讀取毛坯STL文件{args.stock_path}，係統會將其轉換為像素矩陣..."
                )
                start_time = datetime.now()
                image = convert_stl_to_numpy(
                    args.stock_path,
                    resolution=args.resolution,
                    precision=args.precision,
                    binary=args.binary,  # 使用參數控制二進制或RGB格式
                )
                print(f"已成功將其轉換為像素矩陣，耗時{datetime.now()-start_time}")

            elif args.stock_path.endswith(".stp") or args.stock_path.endswith(".step"):
                print(
                    f"成功讀取毛坯STP文件{args.stock_path}，係統會將其轉換為像素矩陣..."
                )
                start_time = datetime.now()
                image = convert_stp_to_numpy(
                    args.stock_path,
                    resolution=args.resolution,
                    precision=args.precision,
                    binary=args.binary,  # 添加binary參數
                    linear_deflection=args.linear_deflection,
                    angular_deflection=args.angular_deflection,
                )
                print(f"已成功將其轉換為像素矩陣，耗時{datetime.now()-start_time}")
            else:
                pixel_size = np.array([0, 0, 0]).astype(int)
                raise ValueError(f"毛坯僅支持STL/ZST/STP文件：{args.stock_path}")

        else:
            print(f'Loading from {row["sub_program_last"]}')
            # 使用 glob 尋找符合條件的.zst文件
            zst_files = glob.glob(
                f'{args.numpy_out_path}/{row["sub_program_last"]}_*.zst'
            )
            if not zst_files:
                raise FileNotFoundError(
                    f"No .zst file found starting with {row['sub_program_last']} in {args.numpy_out_path}"
                )

            # 取最新的一個文件
            latest_zst = sorted(zst_files, key=os.path.getmtime)[-1]
            image, _ = load_from_zst(latest_zst)

        # 处理image的维度，得到pixel_size
        # 如果image只有三维，请帮我延伸一维到[H,W,D,1]
        if len(image.shape) == 3:
            image = image[..., np.newaxis]
            assert (
                len(image.shape) == 4
            ), "ZST stock file should be 4D, but got {image.shape}"
        # 根據加載的圖像格式調整像素尺寸計算
        if args.binary:
            assert (
                image.shape[3] == 1
            ), "Binary mode is not supported for ZST stock files"
            pixel_size = np.array(image.shape[:3]).astype(int)
            print(f"成功讀取毛坯ZST文件{args.stock_path}（二進制格式）...")
        else:
            assert image.shape[3] == 3, "RGB mode is not supported for ZST stock files"
            pixel_size = np.array(
                [image.shape[1], image.shape[0], image.shape[2]]
            ).astype(int)
            print(f"成功讀取毛坯ZST文件{args.stock_path}（RGB格式）...")

        # 轉化比例格式的origin
        if not origin_initialized:
            for i in range(3):
                if origin[i] < 1 and origin[i] > 0:
                    origin[i] = int(origin[i] * pixel_size[i])
                else:
                    origin[i] = origin[i] * 10 ** (
                        args.precision - 3
                    )  # UI直接返回的是用户输入
            origin_initialized = True

        # 轉化圖像數據類型
        if image.dtype != np.bool_ and image.dtype != np.uint8:
            image = image.astype(np.uint8)

        print(f"----毛坯尺寸: {pixel_size}")
        print(f"----坐標原點位置: {origin}")

        # 如果毛坯是stl或stp，則將其保存為zst
        if pd.isna(row["sub_program_last"]) and (
            args.stock_path.endswith(".stl")
            or args.stock_path.endswith(".stp")
            or args.stock_path.endswith(".step")
        ):
            # 決定輸出文件名
            if args.stock_path.endswith(".stl"):
                zst_output_path = args.stock_path.replace(".stl", ".zst")
            elif args.stock_path.endswith(".stp"):
                zst_output_path = args.stock_path.replace(".stp", ".zst")
            elif args.stock_path.endswith(".step"):
                zst_output_path = args.stock_path.replace(".step", ".zst")
            else:
                raise ValueError(f"Unsupported file type: {args.stock_path}")

            save_to_zst(
                image,
                zst_output_path,
                origin=origin,
            )

        print(f"sum(image)={np.sum(image)}")
        # verbose=False 不渲染图片能节约20%时间
        out_image = run(
            df,
            row["sub_program"],
            tools_df,
            image,
            origin,
            precision=precision,
            verbose=args.verbose,
            r_slack=args.r_slack,
            z_slack=args.z_slack,
            excel_out_path=args.excel_out_path,
            use_gpu=args.use_gpu,
            binary=args.binary,  # 傳遞binary參數
        )

        # np.save(
        #     f'{args.numpy_out_path}/{row["sub_program"]}.npy',
        #     out_image,
        # )
        save_to_zst(
            out_image,
            f'{args.numpy_out_path}/{row["sub_program"]}.zst',
            origin=origin,
        )

    # 將所有仿真結果合併保存
    print("merge all subpgrogram simulation")
    
    # 使用 SimulationDataAppender 的靜態方法來合併所有仿真結果
    merge_stats = SimulationDataAppender.merge_all_simulations(
        product_master_path=args.product_master,
        excel_out_path=args.excel_out_path
    )

    # 寻找最后一个子程式的zst，并copy到simulation/lastest中
    # 获取最后一个子程式编号
    last_sub_program = funcs.iloc[-1]["sub_program"]
    print(f"尋找最後一個子程式 {last_sub_program} 的 zst 檔案")

    # 使用 glob 查找最后一个子程式的 zst 文件
    last_zst_files = glob.glob(f"{args.numpy_out_path}/{last_sub_program}_*.zst")

    if last_zst_files:
        # 创建目标目录
        os.makedirs(args.excel_out_path, exist_ok=True)

        # 复制文件到目标目录
        latest_zst_file = last_zst_files[0]
        target_path = os.path.join(
            args.excel_out_path, os.path.basename(latest_zst_file)
        )
        shutil.copy2(latest_zst_file, target_path)
        print(
            f"成功將最後一個子程式 {last_sub_program} 的 zst 檔案複製到 {target_path}"
        )
    else:
        print(f"警告：未找到最後一個子程式 {last_sub_program} 的 zst 檔案")

    # 在excel_out_path的process_info.json中，將finish_flag設為True，finish_time設為當前時間，finish_timestamp設為當前時間戳
    process_info_file = os.path.join(args.excel_out_path, "process_info.json")
    with open(process_info_file, "r", encoding="utf-8") as f:
        process_info = json.load(f)
    process_info["finish_flag"] = True
    process_info["finish_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    process_info["finish_timestamp"] = datetime.now().timestamp()
    with open(process_info_file, "w", encoding="utf-8") as f:
        json.dump(process_info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    cli_run()
