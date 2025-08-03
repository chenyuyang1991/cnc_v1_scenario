import math
import os
import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from cnc_genai.src.simulation.cutting import draw_G01_cv, draw_G02_cv, draw_G03_cv
from cnc_genai.src.simulation.rotating import rotate_G68_cv
from cnc_genai.src.simulation.rotate_coordinates import (
    rotated_physical_to_pixel,
    multiple_rotated_physical_to_pixel,
)
from cnc_genai.src.simulation.calculate import (
    identify_is_valid,
    calculate_ap,
    calculate_ae,
)
from cnc_genai.src.simulation.utils import (
    physical_to_pixel,
    display_recent_cutting,
    update_image,
)
from cnc_genai.src.simulation.utils import (
    save_to_zst,
    load_from_zst,
)
from cnc_genai.src.simulation.colors import (
    get_step_color,
    CUTTING_COLOR,
    MATERIAL_COLOR,
    EMPTY_COLOR,
)


def configure_logging(log_file="simulation.log"):
    logging.basicConfig(
        level=logging.DEBUG,  # 設定最低日誌等級
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 將日誌輸出到檔案
            logging.StreamHandler(),  # 同時輸出到終端機（可選）
        ],
    )


def run(
    df: pd.DataFrame,
    sub_program: str,
    tools_df: pd.DataFrame,
    image: np.ndarray,  # 毛坯
    origin: np.ndarray,  # 坐标系原点
    precision: int = 4,
    verbose: bool = False,
    r_slack: int = 0,
    z_slack: int = 0,
    timestamp: str = None,
    use_gpu: bool = False,
    early_stop: int = None,
    log_path: str = None,
) -> np.ndarray:

    if use_gpu:
        import cupy
        from cnc_genai.src.simulation.calculate_gpu import calculate_metric_gpu

    # 篩選出子程序代碼行
    if timestamp is None:
        timestamp = datetime.today().strftime("%y%m%d_%H%M%S")
    df["sub_program"] = df["sub_program"].astype(str)
    df = df[df["sub_program"] == sub_program]

    # 假設初始化刀頭位置
    df["X"] = df["X"].fillna(0)
    df["Y"] = df["Y"].fillna(0)
    df["Z"] = df["Z"].fillna(99999)  # 无限高的初始抬刀距离

    df["X_prev"] = df["X"].shift(1).fillna(0)
    df["Y_prev"] = df["Y"].shift(1).fillna(0)
    df["Z_prev"] = df["Z"].shift(1).fillna(99999)

    os.makedirs(f"../cnc_intermediate/parsed_code/", exist_ok=True)
    df.to_excel(f"../cnc_intermediate/parsed_code/{sub_program}.xlsx", index=False)

    print(f"--子程序有{df.shape[0]}行代碼")

    df_output = []

    # 創建與原圖同樣大小的掩碼，加速運算
    pixel_size = np.array([image.shape[1], image.shape[0], image.shape[2]]).astype(int)
    mask = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2]), np.uint8)

    # transfer voxels to GPU memory
    if use_gpu:
        image = cupy.asarray(image)

    # pixel_size = np.array([image.shape[0], image.shape[1], image.shape[2]]).astype(int)
    # mask = np.zeros((pixel_size[0], pixel_size[1], pixel_size[2]), np.uint8)
    step = 0
    step_time = datetime.now()
    inst_count = 0
    time_profile = []
    for idx, row in df.iterrows():

        step_time = datetime.now()

        # early stop (debug)
        if early_stop and inst_count > early_stop:
            image = cupy.asnumpy(image) if use_gpu else image
            break

        STEP_COLOR = get_step_color(idx)

        logging.info(f"line {idx}: {row['src']}")
        print(
            f"{row['sub_program']}: 第{row['row_id']}行, {row['src']}, [{(datetime.now() - step_time).seconds}]"
        )

        # 獲取當前刀具信息
        tool = row["T"]
        tool_info = tools_df.loc[tools_df["刀號"] == tool, "規格型號"]

        if len(tool_info):

            tool_d = float(tools_df.loc[tools_df["刀號"] == tool, "刀頭直徑"].values[0])
            tool_h = float(tools_df.loc[tools_df["刀號"] == tool, "刀頭高度"].values[0])
            tool_d = round(tool_d * 10 ** (precision - 3))
            tool_h = round(tool_h * 10 ** (precision - 3))
            tool_d_slack = round(tool_d + 2 * r_slack)
        else:
            tool_d = 0
            tool_h = 0
            tool_d_slack = 0

        row["tool_diameter"] = tool_d
        row["tool_height"] = tool_h
        row["tool_r_slack"] = r_slack * 10 ** (3 - precision)
        row["tool_z_slack"] = z_slack * 10 ** (3 - precision)

        print(f"!!!!!!!!!當前刀具: {row['T']}, 刀具直徑: {tool_d}, 刀具高度: {tool_h}")

        if row["Z"] == 99999:
            continue

        # 從物理坐標轉換為像素坐標
        start_point = np.array([row["X_prev"], row["Y_prev"], row["Z_prev"]])
        end_point = np.array([row["X"], row["Y"], row["Z"]])

        # 轉換坐標
        rotating_centers = np.array(
            [
                np.array([float(k) for k in row[f"rotate_{x}_center"].split("/")])
                for x in ["Z", "X", "Y"]
                if pd.notna(row[f"rotate_{x}_angle"])
            ]
        )
        rotating_angles = np.array(
            [
                float(row[f"rotate_{x}_angle"])
                for x in ["Z", "X", "Y"]
                if pd.notna(row[f"rotate_{x}_angle"])
            ]
        )
        rotating_axes = np.array(
            [x for x in ["Z", "X", "Y"] if row[f"rotate_{x}_axis"] is True]
        )

        # print(f'旋轉中心: {rotating_centers}')
        # print(f'旋轉角度: {rotating_angles}')
        # print(f'旋轉軸: {rotating_axes}')

        if len(rotating_centers) == 1:
            # 只有一個軸旋轉
            start_point = rotated_physical_to_pixel(
                image,
                start_point,
                rotating_centers[0],
                rotating_angles[0],
                axis=rotating_axes[0],
                origin=origin,
                precision=precision,
            )
            end_point = rotated_physical_to_pixel(
                image,
                end_point,
                rotating_centers[0],
                rotating_angles[0],
                axis=rotating_axes[0],
                origin=origin,
                precision=precision,
            )
        elif len(rotating_centers) > 1:
            # 至少兩個軸旋轉
            start_point = multiple_rotated_physical_to_pixel(
                image,
                start_point,
                rotating_centers,
                rotating_angles,
                axes=rotating_axes,
                origin=origin,
                precision=precision,
            )
            end_point = multiple_rotated_physical_to_pixel(
                image,
                end_point,
                rotating_centers,
                rotating_angles,
                axes=rotating_axes,
                origin=origin,
                precision=precision,
            )
        else:
            # 沒有旋轉
            start_point = physical_to_pixel(
                start_point, origin=origin, size=pixel_size, precision=precision
            )
            end_point = physical_to_pixel(
                end_point, origin=origin, size=pixel_size, precision=precision
            )

        # 修改z位置以考慮z_slack
        if z_slack > 0:
            start_point_z_slack = start_point - np.array([0, 0, z_slack])
            end_point_z_slack = end_point - np.array([0, 0, z_slack])
        else:
            start_point_z_slack = start_point
            end_point_z_slack = end_point

        row["X_prev_pixel"], row["Y_prev_pixel"], row["Z_prev_pixel"] = start_point
        row["X_pixel"], row["Y_pixel"], row["Z_pixel"] = end_point

        # G02|G03圆弧
        circle_center, _ = _prepare_G02_G03_points(
            row,
            image,
            pixel_size,
            origin,
            precision,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            z_slack,
        )

        # G81钻孔
        G81_point, G81_point_z_slack = _prepare_G81_points(
            row,
            image,
            pixel_size,
            origin,
            precision,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            z_slack,
        )

        # G83钻孔
        G83_point, G83_point_z_slack = _prepare_G83_points(
            row,
            image,
            pixel_size,
            origin,
            precision,
            rotating_centers,
            rotating_angles,
            rotating_axes,
            z_slack,
        )

        if row["move_code"] in ["G02", "G03"]:

            # 記錄從像素計算的，圓弧半徑、起點角度、終點角度、經過弧度
            row["radius_pixel"] = int(
                np.sqrt(
                    (start_point[0] - circle_center[0]) ** 2
                    + (start_point[1] - circle_center[1]) ** 2
                )
            )
            row["angle_start_pixel"] = math.degrees(
                math.atan2(
                    start_point[1] - circle_center[1], start_point[0] - circle_center[0]
                )
            )
            row["angle_end_pixel"] = math.degrees(
                math.atan2(
                    end_point[1] - circle_center[1], end_point[0] - circle_center[0]
                )
            )
            row["arc_angle_pixel"] = abs(
                row["angle_start_pixel"] - row["angle_end_pixel"]
            )
            row["circle_center_x_pixel"] = circle_center[0]
            row["circle_center_y_pixel"] = circle_center[1]

        if row["move_code"] in ["G01", "G02", "G03", "G81"]:
            if not np.array_equal(start_point, end_point):

                if len(re.findall(r"(G0[1-3])", row["src"])) > 1:
                    raise ValueError("G01|G02|G03 appeared in same line")
                else:
                    inst_count += 1
                    current_time = datetime.now()
                    if row["move_code"] == "G01":
                        tile, tile_range = draw_G01_cv(
                            mask,
                            start_point,
                            end_point,
                            tool_d // 2,
                            tool_h,
                            use_gpu=use_gpu,
                        )
                        if r_slack + z_slack > 0:
                            tile_slack, tile_range_slack = draw_G01_cv(
                                mask,
                                start_point_z_slack,
                                end_point_z_slack,
                                tool_d_slack // 2,
                                tool_h + z_slack,
                                use_gpu=use_gpu,
                            )
                        else:
                            tile_slack, tile_range_slack = tile, tile_range
                    elif row["move_code"] == "G02":
                        tile, tile_range = draw_G02_cv(
                            mask,
                            start_point,
                            end_point,
                            circle_center,
                            tool_d // 2,
                            tool_h,
                            use_gpu=use_gpu,
                        )
                        if r_slack + z_slack > 0:
                            tile_slack, tile_range_slack = draw_G02_cv(
                                mask,
                                start_point_z_slack,
                                end_point_z_slack,
                                circle_center,
                                tool_d_slack // 2,
                                tool_h + z_slack,
                                use_gpu=use_gpu,
                            )
                        else:
                            tile_slack, tile_range_slack = tile, tile_range
                    elif row["move_code"] == "G03":
                        tile, tile_range = draw_G03_cv(
                            mask,
                            start_point,
                            end_point,
                            circle_center,
                            tool_d // 2,
                            tool_h,
                            use_gpu=use_gpu,
                        )
                        if r_slack + z_slack > 0:
                            tile_slack, tile_range_slack = draw_G03_cv(
                                mask,
                                start_point_z_slack,
                                end_point_z_slack,
                                circle_center,
                                tool_d_slack // 2,
                                tool_h + z_slack,
                                use_gpu=use_gpu,
                            )
                        else:
                            tile_slack, tile_range_slack = tile, tile_range
                    elif row["move_code"] == "G81":
                        tile, tile_range = draw_G01_cv(
                            mask,
                            start_point,
                            G81_point,
                            tool_d // 2,
                            tool_h,
                            use_gpu=use_gpu,
                        )
                        if r_slack + z_slack > 0:
                            tile_slack, tile_range_slack = draw_G01_cv(
                                mask,
                                start_point_z_slack,
                                G81_point_z_slack,
                                tool_d_slack // 2,
                                tool_h + z_slack,
                                use_gpu=use_gpu,
                            )
                        else:
                            tile_slack, tile_range_slack = tile, tile_range
                    elif row["move_code"] == "G83":
                        tile, tile_range = draw_G01_cv(
                            mask,
                            start_point,
                            G83_point,
                            tool_d // 2,
                            tool_h,
                            use_gpu=use_gpu,
                        )
                        if r_slack + z_slack > 0:
                            tile_slack, tile_range_slack = draw_G01_cv(
                                mask,
                                start_point_z_slack,
                                G83_point_z_slack,
                                tool_d_slack // 2,
                                tool_h + z_slack,
                                use_gpu=use_gpu,
                            )
                        else:
                            tile_slack, tile_range_slack = tile, tile_range
                    else:
                        print(f"不支援的指令: {row['move_code']}")
                        raise ValueError(f"不支援的指令: {row['move_code']}")

                    if verbose:
                        print(f"|| 生成切割mask用時 {datetime.now() - current_time}")
                    current_time = datetime.now()

                    if tile is not None:

                        if not use_gpu:
                            # 判斷是否空切
                            row["cutting_area"] = np.sum(tile)
                            row["is_valid"], row["hit_area"], _ = identify_is_valid(
                                image, tile, tile_range, True
                            )
                            (
                                row["is_valid_slack"],
                                row["hit_area_slack"],
                                row["hit_z_slack"],
                            ) = identify_is_valid(
                                image,
                                tile_slack,
                                tile_range_slack,
                                True,
                                z_slack=z_slack,
                            )

                            # 計算切深
                            ap_res = calculate_ap(image, tile, tile_range, verbose)
                            for k, v in ap_res.items():
                                row[k] = v

                            # 計算切寬
                            ae_res = calculate_ae(image, tile, tile_range, verbose)
                            for k, v in ae_res.items():
                                row[k] = v

                        if use_gpu:
                            all_metrics = calculate_metric_gpu(
                                image, tile, tile_range, tile_slack, tile_range_slack
                            )
                            for k, v in all_metrics.items():
                                row[k] = v

                        print(
                            f"|| 計算切寬切深指標用時 {datetime.now() - current_time}"
                        )
                        current_time = datetime.now()

                        # 計算切割區域到材料區域的最小距離
                        # if row['is_valid']:
                        #     row['distance_to_material'] = 0
                        # else:
                        #     row['distance_to_material'] = get_distance_between_masks(
                        #         image, mask, tile_range, tile
                        #     )
                        # print(f'|| 計算到材料的距離 {datetime.now() - current_time}')
                        # current_time = datetime.now()

                        if use_gpu:
                            image, deepest_layer = update_image(
                                image,
                                tile,
                                tile_range,
                                STEP_COLOR,
                                CUTTING_COLOR,
                                use_gpu=use_gpu,
                            )
                        else:
                            # 進行切割
                            image, deepest_layer = update_image(
                                image, tile, tile_range, STEP_COLOR, CUTTING_COLOR
                            )
                        print(f"|| 真實切割用時 {datetime.now() - current_time}")

                        # 显示
                        if verbose:
                            if use_gpu:
                                image_cpu = cupy.asnumpy(image)
                                display_recent_cutting(image_cpu, deepest_layer, row)
                            else:
                                display_recent_cutting(image, deepest_layer, row)
                    else:
                        pass

        # G68旋轉XY平面
        if row["move_code"] == "G68":
            row["rotate_Z_center"] = np.array(row["G68_X"], row["G68_Y"], 0)
            # 注意G68逆时针为正，而rotate_Z_axis以工件顺时针为正，所以这里取负值
            row["rotate_Z_angle"] = row["rotate_Z_angle"] - row["G68_angle"]

        latency = datetime.now() - step_time

        # 記錄仿真時間
        row["simulation_time_used"] = latency.seconds
        row["global_step"] = idx
        row["sub_program_step"] = step
        row["debug_color"] = "/".join([str(int(x)) for x in STEP_COLOR])

        # 更新特征表
        step += 1
        df_output.append(row)

        time_profile.append({"idx": idx, "latency": latency.total_seconds()})
        os.makedirs(
            f"../cnc_intermediate/simulation/simulation_output_{timestamp}/by_steps",
            exist_ok=True,
        )
        pd.DataFrame.from_dict(df_output).to_excel(
            f"../cnc_intermediate/simulation/simulation_output_{timestamp}/res_{sub_program}_rslack={r_slack}_zslack={z_slack}_precision={precision}.xlsx",
            index=False,
        )
        # save_to_zst(image, f"../cnc_intermediate/simulation/simulation_output_{timestamp}/by_steps/{row['O']}_{row['row_id']}_{row['src']}.zst")

    if log_path:
        time_profile_df = pd.DataFrame(time_profile)
        time_profile_df.to_csv(str(log_path), index=False)

    return image


def _prepare_G02_G03_points(
    row,
    image,
    pixel_size,
    origin,
    precision,
    rotating_centers,
    rotating_angles,
    rotating_axes,
    z_slack,
):
    if row["move_code"] in ["G02", "G03"]:
        circle_center = np.array(
            [row["X_prev"] + row["I"], row["Y_prev"] + row["J"], row["Z_prev"]]
        )
        if len(rotating_centers) == 1:
            circle_center = rotated_physical_to_pixel(
                image,
                circle_center,
                rotating_centers[0],
                rotating_angles[0],
                axis=rotating_axes[0],
                origin=origin,
                precision=precision,
            )
        elif len(rotating_centers) > 1:
            circle_center = multiple_rotated_physical_to_pixel(
                image,
                circle_center,
                rotating_centers,
                rotating_angles,
                axes=rotating_axes,
                origin=origin,
                precision=precision,
            )
        else:
            circle_center = physical_to_pixel(
                circle_center, origin=origin, size=pixel_size, precision=precision
            )
        if z_slack > 0:
            circle_center_z_slack = circle_center - np.array([0, 0, z_slack])
        else:
            circle_center_z_slack = circle_center
        return circle_center, circle_center_z_slack
    else:
        return None, None


def _prepare_G81_points(
    row,
    image,
    pixel_size,
    origin,
    precision,
    rotating_centers,
    rotating_angles,
    rotating_axes,
    z_slack,
):
    if row["move_code"] == "G81":
        # G81指令的座標存儲在標準的X,Y,Z列中
        G81_point = np.array([row["X"], row["Y"], row["Z"]])
        if len(rotating_centers) == 1:
            G81_point = rotated_physical_to_pixel(
                image,
                G81_point,
                rotating_centers[0],
                rotating_angles[0],
                axis=rotating_axes[0],
                origin=origin,
                precision=precision,
            )
        elif len(rotating_centers) > 1:
            G81_point = multiple_rotated_physical_to_pixel(
                image,
                G81_point,
                rotating_centers,
                rotating_angles,
                axes=rotating_axes,
                origin=origin,
                precision=precision,
            )
        else:
            G81_point = physical_to_pixel(
                G81_point, origin=origin, size=pixel_size, precision=precision
            )
        if z_slack > 0:
            G81_point_z_slack = G81_point - np.array([0, 0, z_slack])
        else:
            G81_point_z_slack = G81_point
        return G81_point, G81_point_z_slack
    else:
        return None, None


def _prepare_G83_points(
    row,
    image,
    pixel_size,
    origin,
    precision,
    rotating_centers,
    rotating_angles,
    rotating_axes,
    z_slack,
):
    if row["move_code"] == "G83":
        # G83指令的座標存儲在標準的X,Y,Z列中
        G83_point = np.array([row["X"], row["Y"], row["Z"]])
        if len(rotating_centers) == 1:
            G83_point = rotated_physical_to_pixel(
                image,
                G83_point,
                rotating_centers[0],
                rotating_angles[0],
                axis=rotating_axes[0],
                origin=origin,
                precision=precision,
            )
        elif len(rotating_centers) > 1:
            G83_point = multiple_rotated_physical_to_pixel(
                image,
                G83_point,
                rotating_centers,
                rotating_angles,
                axes=rotating_axes,
                origin=origin,
                precision=precision,
            )
        else:
            G83_point = physical_to_pixel(
                G83_point, origin=origin, size=pixel_size, precision=precision
            )
        if z_slack > 0:
            G83_point_z_slack = G83_point - np.array([0, 0, z_slack])
        else:
            G83_point_z_slack = G83_point
        return G83_point, G83_point_z_slack
    else:
        return None, None


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    funcs = pd.read_excel("../app/simulation_master/x2867-cnc2/product_master.xlsx")
    funcs["sub_program"] = funcs["sub_program"].astype(int).astype(str)
    funcs["sub_program_last"] = funcs["sub_program"].shift(1)

    df = pd.read_excel("./cnc_genai/parsed_code/command_extract.xlsx")
    df = df.drop_duplicates(["row_id", "src", "sub_program"], keep="last").reset_index()

    tools_df = pd.read_excel("./cnc_genai/data/X2867刀具.xlsx").drop_duplicates(
        ["刀號", "規格型號"]
    )
    precision = 4
    timestamp = datetime.today().strftime("%y%m%d_%H%M%S")
    os.makedirs(
        f"../cnc_intermediate/simulation/simulation_npys_{timestamp}", exist_ok=True
    )

    for idx, row in funcs.iterrows():

        print(row["sub_program_key"])

        if pd.isna(row["sub_program_last"]):
            THICKNESS = 2.3
            EDGE_W = 8.2
            EDGE_H = 10.2
            SIZE_X = 548.63  # mm
            SIZE_Y = 376.32
            SIZE_Z = 12.46

            # 注意xy
            size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
            pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

            thickness = int(THICKNESS * 10 ** (precision - 3))
            edge_w = int(EDGE_W * 10 ** (precision - 3))
            edge_h = int(EDGE_H * 10 ** (precision - 3))

            image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
            image[:] = MATERIAL_COLOR
            image[thickness:-thickness, thickness:-thickness, thickness:-thickness] = (
                EMPTY_COLOR
            )
            image[edge_w:-edge_w, edge_h:-edge_h, thickness:] = EMPTY_COLOR

        else:
            print(f'Loading from {row["sub_program_last"]}')
            image = np.load(
                f'../cnc_intermediate/simulation/simulation_npys_{timestamp}/{row["sub_program_last"]}.npy'
            )
            pixel_size = np.array(
                [image.shape[1], image.shape[0], image.shape[2]]
            ).astype(int)

        origin = np.array(
            [
                pixel_size[0] // 2,
                pixel_size[1] // 2,
                0,
            ]
        ).astype(
            int
        )  # G54
        print(f"----毛坯尺寸: {pixel_size}")
        print(f"----坐標原點位置: {origin}")

        # verbose=False 不渲染图片能节约20%时间
        out_image = run(
            df,
            row["sub_program"],
            tools_df,
            image,
            origin,
            precision=4,
            verbose=False,
            r_slack=1,
            z_slack=1,
            timestamp=timestamp,
        )
        np.save(
            f'../cnc_intermediate/simulation/simulation_npys_{timestamp}/{row["sub_program"]}.npy',
            out_image,
        )
