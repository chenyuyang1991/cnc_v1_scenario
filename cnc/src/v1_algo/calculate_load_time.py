import math
import os
import pandas as pd
import numpy as np
from cnc_genai.src.utils.utils import load_config_v1
from cnc_genai.src.code_parsing.code_parsing import run_code_parsing


def _calculate_path_length(line):
    if line["move_code"] in ["G00", "G01"]:
        return _calculate_straight_length(line)
    elif line["move_code"] in ["G02", "G03"]:
        return _calculate_arc_length(line)
    else:
        return None


def _calculate_straight_length(line):
    """
    计算两点之间的距离
    """
    pos_prev = np.array([line["X_prev"], line["Y_prev"], line["Z_prev"]])
    pos = np.array([line["X"], line["Y"], line["Z"]])
    distance = np.linalg.norm(pos - pos_prev)
    return distance


def _calculate_arc_length(line):
    """
    计算圆弧的长度，暂时只考虑 K 不存在的情况，即 XY 平面的圆弧
    """
    # 定义点的坐标
    pos_prev = np.array([line["X_prev"], line["Y_prev"]])
    pos = np.array([line["X"], line["Y"]])
    center = np.array([line["X_prev"] + line["I"], line["Y_prev"] + line["J"]])
    radius = np.linalg.norm(pos_prev - center)

    # 计算点的角度
    theta_prev = np.arctan2(
        pos_prev[1] - center[1], pos_prev[0] - center[0]
    )  # A点的角度
    theta = np.arctan2(pos[1] - center[1], pos[0] - center[0])  # B点的角度

    # 顺时针方向夹角
    delta_theta_cw = (theta_prev - theta) % (2 * np.pi)

    # 逆时针方向夹角
    delta_theta_ccw = (theta - theta_prev) % (2 * np.pi)

    # 计算弧长
    if line["move_code"] == "G02":
        return radius * delta_theta_cw
    elif line["move_code"] == "G03":
        return radius * delta_theta_ccw
    else:
        return None


def _calculate_turning_angle(line):
    angle_degrees = None
    if line["move_code"] in ["G01", "GOO"]:
        if line["move_code_prev"] and (line["move_code_prev"] in ["G01", "GOO"]):
            # 定义三个点的坐标
            pos_prev_prev = np.array(
                [line["X_prev_prev"], line["Y_prev_prev"], line["Z_prev_prev"]]
            )
            pos_prev = np.array([line["X_prev"], line["Y_prev"], line["Z_prev"]])
            pos = np.array([line["X"], line["Y"], line["Z"]])

            # 计算向量
            dir_prev = pos_prev - pos_prev_prev
            dir_curr = pos - pos_prev

            # 计算点积和模长
            dot_product = np.dot(dir_prev, dir_curr)
            magnitude_prev = np.linalg.norm(dir_prev)
            magnitude_curr = np.linalg.norm(dir_curr)

            if magnitude_prev != 0 and magnitude_curr != 0:
                # 计算夹角（弧度）
                cos_theta = dot_product / (magnitude_prev * magnitude_curr)
                angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                # 如果需要角度制：
                angle_degrees = np.degrees(angle_radians)

            else:
                angle_degrees = 0

    return angle_degrees


def _calculate_time(
    line,
    rapid_speed=20000.0,  # mm/min
    rapid_acceleration=50000000,  # mm/min^2
    linear_acceleration=30000000,  # mm/min^2
    arc_acceleration=30000000,  # mm/min^2
    tool_change_time=5.0,  # s
    spindle_acceleration_assump=2000.0,  # rpm/s，assupmtion是不是太小？
    angle_slowdown_diff=0.001,  # s
):
    # 以秒输出
    output_dict = {
        "acc_time": None,
        "acc_dist": None,
        "const_time": None,
        "const_dist": None,
        "acc/dec": None,
        "time_physical": None,
        "time_tool_change": None,
        "time_spindle_acc": None,
        "time_turning": None,
    }
    # 由於不知道刀具初始位置，暫時不計算這一部分時間（不影響提升，會計入理論time_physical和實際的差）
    if line["Z_prev"] != 99999 and line["Z"] != 99999:

        # 考慮G04
        if line["move_code"] == "G04" and line["G04_time"]:
            output_dict["time_physical"] = line["G04_time"]

        # 考慮G00
        if line["move_code"] == "G00" and line["F"]:
            end_F = rapid_speed

            # 从非 rapid 切换到 rapid，需要从 feed_rate 加速到 rapid_speed，這一段發生在G00中
            if line["move_code_prev"] in ["G01", "G02", "G03"]:
                start_F = line["F_prev"]
                if start_F < end_F:
                    # 加速阶段
                    output_dict["acc/dec"] = "accelerate"
                    output_dict["acc_time"] = (
                        end_F - start_F
                    ) / rapid_acceleration  # 加速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) + (
                        0.5 * rapid_acceleration * (output_dict["acc_time"] ** 2)
                    )
                else:
                    # 减速阶段
                    output_dict["acc/dec"] = "decelerate"
                    output_dict["acc_time"] = (
                        start_F - end_F
                    ) / rapid_acceleration  # 减速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) - (
                        0.5 * rapid_acceleration * (output_dict["acc_time"] ** 2)
                    )

                # 如果加速距离超过总距离，调整为单纯加速或减速的情况
                if output_dict["acc_dist"] >= line["path_length"]:
                    if start_F > end_F:
                        # 减速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(
                                start_F**2
                                - 2 * linear_acceleration * line["path_length"]
                            )
                            - start_F
                        ) / (
                            -linear_acceleration
                        )  # 减速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s
                    else:
                        # 加速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(
                                start_F**2
                                + 2 * linear_acceleration * line["path_length"]
                            )
                            - start_F
                        ) / linear_acceleration  # 加速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s
                else:
                    # 计算匀速阶段时间
                    output_dict["const_dist"] = (
                        line["path_length"] - output_dict["acc_dist"]
                    )
                    output_dict["const_time"] = (
                        output_dict["const_dist"] / end_F
                    )  # 匀速阶段时间 (秒)
                    output_dict["time_physical"] = (
                        output_dict["acc_time"] + output_dict["const_time"]
                    ) * 60.0  # 单位s
            # 保持rapid不變（暫未考慮為了將來非rapid的減速）
            else:
                output_dict["time_physical"] = (
                    line["path_length"] / rapid_speed * 60.0
                )  # 单位s

        # 考慮G01
        elif line["move_code"] == "G01" and line["F"]:
            end_F = line["F"]
            # 根据上一行的速度计算加速或减速时间
            if line["move_code_prev"] == "G00":
                # 从 rapid 切换到 linear，需要从 rapid_speed 减速到目标 feed_rate，我們假設這一段減速發生在上一段G00的末尾，而非這段G01中
                # start_F = rapid_speed
                start_F = line["F"]
            else:
                # 从 linear 切换到 linear，根据前一行的速度判断是加速还是减速
                start_F = line["F_prev"]

            if start_F == end_F:
                output_dict["time_physical"] = (
                    line["path_length"] / start_F * 60.0
                )  # 单位s

            else:
                if start_F < end_F:
                    # 加速阶段
                    output_dict["acc/dec"] = "accelerate"
                    output_dict["acc_time"] = (
                        end_F - start_F
                    ) / linear_acceleration  # 加速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) + (
                        0.5 * linear_acceleration * (output_dict["acc_time"] ** 2)
                    )
                else:
                    # 减速阶段
                    output_dict["acc/dec"] = "decelerate"
                    output_dict["acc_time"] = (
                        start_F - end_F
                    ) / linear_acceleration  # 减速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) - (
                        0.5 * linear_acceleration * (output_dict["acc_time"] ** 2)
                    )

                # 如果加速距离超过总距离，调整为单纯加速或减速的情况
                if output_dict["acc_dist"] > line["path_length"]:
                    if start_F > end_F:
                        # 减速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(
                                start_F**2
                                - 2 * linear_acceleration * line["path_length"]
                            )
                            - start_F
                        ) / (
                            -linear_acceleration
                        )  # 减速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s

                    else:
                        # 加速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(
                                start_F**2
                                + 2 * linear_acceleration * line["path_length"]
                            )
                            - start_F
                        ) / linear_acceleration  # 加速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s
                else:
                    # 计算匀速阶段时间
                    output_dict["const_dist"] = (
                        line["path_length"] - output_dict["acc_dist"]
                    )
                    output_dict["const_time"] = (
                        output_dict["const_dist"] / end_F
                    )  # 匀速阶段时间 (秒)
                    output_dict["time_physical"] = (
                        output_dict["acc_time"] + output_dict["const_time"]
                    ) * 60.0  # 单位s

        # 考慮G02/G03
        elif line["move_code"] in ["G02", "G03"] and line["F"]:
            end_F = line["F"]
            # 根据上一行的速度计算加速或减速时间
            if line["move_code_prev"] == "G00":
                # 从 rapid 切换到 linear，需要从 rapid_speed 减速到目标 feed_rate，我們假設這一段減速發生在上一段G00的末尾，而非這段G02/G03中
                # start_F = rapid_speed
                start_F = line["F"]
            else:
                # 从 linear 切换到 linear，根据前一行的速度判断是加速还是减速
                start_F = line["F_prev"]

            if start_F == end_F:
                output_dict["time_physical"] = (
                    line["path_length"] / start_F * 60.0
                )  # 单位s

            else:

                if start_F < end_F:
                    # 加速阶段
                    output_dict["acc/dec"] = "accelerate"
                    output_dict["acc_time"] = (
                        end_F - start_F
                    ) / arc_acceleration  # 加速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) + (
                        0.5 * arc_acceleration * (output_dict["acc_time"] ** 2)
                    )
                else:
                    # 减速阶段
                    output_dict["acc/dec"] = "decelerate"
                    output_dict["acc_time"] = (
                        start_F - end_F
                    ) / arc_acceleration  # 减速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) - (
                        0.5 * arc_acceleration * (output_dict["acc_time"] ** 2)
                    )

                # 如果加速距离超过总距离，调整为单纯加速或减速的情况
                if output_dict["acc_dist"] > line["path_length"]:
                    if start_F > end_F:
                        # 减速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(
                                start_F**2 - 2 * arc_acceleration * line["path_length"]
                            )
                            - start_F
                        ) / (
                            -arc_acceleration
                        )  # 减速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s
                    else:
                        # 加速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(
                                start_F**2 + 2 * arc_acceleration * line["path_length"]
                            )
                            - start_F
                        ) / arc_acceleration  # 加速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s
                else:
                    # 计算匀速阶段时间
                    output_dict["const_dist"] = (
                        line["path_length"] - output_dict["acc_dist"]
                    )
                    output_dict["const_time"] = (
                        output_dict["const_dist"] / end_F
                    )  # 匀速阶段时间 (秒)
                    output_dict["time_physical"] = (
                        output_dict["acc_time"] + output_dict["const_time"]
                    ) * 60.0  # 单位s

        # 考慮G81
        elif line["move_code"] in ["G81"] and line["F"]:
            # 第一段G01
            end_F = line["F"]
            pos_prev = np.array([line["X_prev"], line["Y_prev"], line["Z_prev"]])
            # G81指令的座標存儲在標準的X,Y,Z列中，G81_Z是起始Z位置
            pos = np.array([line["X"], line["Y"], line["Z"]])
            distance = np.linalg.norm(pos - pos_prev)
            # 根据上一行的速度计算加速或减速时间
            if line["move_code_prev"] == "G00":
                # 从 rapid 切换到 linear，需要从 rapid_speed 减速到目标 feed_rate，我們假設這一段減速發生在上一段G00的末尾，而非這段G01中
                # start_F = rapid_speed
                start_F = line["F"]
            else:
                # 从 linear 切换到 linear，根据前一行的速度判断是加速还是减速
                start_F = line["F_prev"]

            if start_F == end_F:
                output_dict["time_physical"] = distance / start_F * 60.0  # 单位s

            else:
                if start_F < end_F:
                    # 加速阶段
                    output_dict["acc/dec"] = "accelerate"
                    output_dict["acc_time"] = (
                        end_F - start_F
                    ) / linear_acceleration  # 加速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) + (
                        0.5 * linear_acceleration * (output_dict["acc_time"] ** 2)
                    )
                else:
                    # 减速阶段
                    output_dict["acc/dec"] = "decelerate"
                    output_dict["acc_time"] = (
                        start_F - end_F
                    ) / linear_acceleration  # 减速时间 (秒)
                    output_dict["acc_dist"] = (start_F * output_dict["acc_time"]) - (
                        0.5 * linear_acceleration * (output_dict["acc_time"] ** 2)
                    )

                # 如果加速距离超过总距离，调整为单纯加速或减速的情况
                if output_dict["acc_dist"] > distance:
                    if start_F > end_F:
                        # 减速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(start_F**2 - 2 * linear_acceleration * distance)
                            - start_F
                        ) / (
                            -linear_acceleration
                        )  # 减速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s

                    else:
                        # 加速阶段重新计算时间（通过距离计算，考虑初速度）
                        output_dict["acc_time"] = (
                            math.sqrt(start_F**2 + 2 * linear_acceleration * distance)
                            - start_F
                        ) / linear_acceleration  # 加速时间 (秒)
                        output_dict["time_physical"] = (
                            output_dict["acc_time"] * 60.0
                        )  # 单位s
                else:
                    # 计算匀速阶段时间
                    output_dict["const_dist"] = distance - output_dict["acc_dist"]
                    output_dict["const_time"] = (
                        output_dict["const_dist"] / end_F
                    )  # 匀速阶段时间 (秒)
                    output_dict["time_physical"] = (
                        output_dict["acc_time"] + output_dict["const_time"]
                    ) * 60.0  # 单位s
            # 第二段G00 - 從鑽孔底部返回到安全高度
            # G81_Z是起始Z位置（R平面），Z是鑽孔底部位置
            pos_prev = np.array([line["X"], line["Y"], line["Z"]])
            pos = np.array([line["X"], line["Y"], line["G81_Z"]])
            distance = np.linalg.norm(pos - pos_prev)
            if line["move_code"] == "G00" and line["F"]:
                end_F = rapid_speed

                # 从非 rapid 切换到 rapid，需要从 feed_rate 加速到 rapid_speed，這一段發生在G00中
                if line["move_code_prev"] in ["G01", "G02", "G03"]:
                    start_F = line["F_prev"]
                    if start_F < end_F:
                        # 加速阶段
                        # output_dict["acc/dec"] = "accelerate"
                        acc_time = (
                            end_F - start_F
                        ) / rapid_acceleration  # 加速时间 (秒)
                        acc_dist = (start_F * acc_time) + (
                            0.5 * rapid_acceleration * (acc_time**2)
                        )
                    else:
                        # 减速阶段
                        # output_dict["acc/dec"] = "decelerate"
                        acc_time = (
                            start_F - end_F
                        ) / rapid_acceleration  # 减速时间 (秒)
                        acc_dist = (start_F * acc_time) - (
                            0.5 * rapid_acceleration * (acc_time**2)
                        )

                    # 如果加速距离超过总距离，调整为单纯加速或减速的情况
                    if acc_dist >= distance:
                        if start_F > end_F:
                            # 减速阶段重新计算时间（通过距离计算，考虑初速度）
                            acc_time = (
                                math.sqrt(
                                    start_F**2 - 2 * linear_acceleration * distance
                                )
                                - start_F
                            ) / (
                                -linear_acceleration
                            )  # 减速时间 (秒)
                            output_dict["time_physical"] += acc_time * 60.0  # 单位s
                        else:
                            # 加速阶段重新计算时间（通过距离计算，考虑初速度）
                            acc_time = (
                                math.sqrt(
                                    start_F**2 + 2 * linear_acceleration * distance
                                )
                                - start_F
                            ) / linear_acceleration  # 加速时间 (秒)
                            output_dict["time_physical"] += acc_time * 60.0  # 单位s
                    else:
                        # 计算匀速阶段时间
                        const_dist = distance - output_dict["acc_dist"]
                        const_time = const_dist / end_F  # 匀速阶段时间 (秒)
                        output_dict["time_physical"] += (
                            acc_time + const_time
                        ) * 60.0  # 单位s
                # 保持rapid不變（暫未考慮為了將來非rapid的減速）
                else:
                    output_dict["time_physical"] += (
                        distance / rapid_speed * 60.0
                    )  # 单位s

        # 考慮G82
        elif line["move_code"] in ["G82"] and line["F"]:
            # todo 计算G82的理论时间
            pass

        # 考慮G83
        elif line["move_code"] in ["G83"] and line["F"]:
            # todo 计算G83的理论时间
            pass

        else:
            pass

    # 增加换刀时间，如果当前行是换刀指令
    if "M06" in line["src"]:
        output_dict["time_tool_change"] = tool_change_time
        if output_dict["time_physical"]:
            output_dict["time_physical"] += tool_change_time
        else:
            output_dict["time_physical"] = tool_change_time

    # 增加主轴加速/减速时间，如果当前行是主轴指令
    if any(code in line["src"] for code in ["M03", "M04", "M05"]):
        # 假设一个简单的加速度（单位：RPM/秒）
        if line["S_prev"] and line["S"]:
            speed_difference = abs(line["S_prev"] - line["S"])
            # 计算加速/减速所需时间
            output_dict["time_spindle_acc"] = (
                speed_difference / spindle_acceleration_assump
            )  # 当前单位s

    # 大拐弯的额外时间
    if line["turning_angle"]:
        if line["turning_angle"] >= 90:
            output_dict["time_turning"] = angle_slowdown_diff  # 当前单位s
            if output_dict["time_physical"]:
                output_dict["time_physical"] += angle_slowdown_diff
            else:
                output_dict["time_physical"] = angle_slowdown_diff

    return pd.Series({**line, **output_dict})


def calculate_cycle_time(parsed_df):
    """
    计算加工总时间，并将每行的预测时间和参数写入 Excel 文件
    """

    # 添加新的列，用于存储上一行的坐标和方向

    parsed_df["X_prev"] = parsed_df["X"].shift(1).fillna(0)
    parsed_df["Y_prev"] = parsed_df["Y"].shift(1).fillna(0)
    parsed_df["Z_prev"] = parsed_df["Z"].shift(1).fillna(99999)

    parsed_df["move_code_prev"] = parsed_df["move_code"].shift(1)
    parsed_df["F_prev"] = parsed_df["F"].shift(1)
    parsed_df["S_prev"] = parsed_df["S"].shift(1)

    parsed_df["X_prev_prev"] = parsed_df["X_prev"].shift(1).fillna(0)
    parsed_df["Y_prev_prev"] = parsed_df["Y_prev"].shift(1).fillna(0)
    parsed_df["Z_prev_prev"] = parsed_df["Z_prev"].shift(1).fillna(99999)

    # 计算直线的转角度数
    parsed_df["turning_angle"] = parsed_df.apply(_calculate_turning_angle, axis=1)

    # 计算路径长度
    parsed_df["path_length"] = parsed_df.apply(_calculate_path_length, axis=1)

    # 计算各种时间
    parsed_df = parsed_df.apply(_calculate_time, axis=1)
    parsed_df["time_physical_acc"] = parsed_df["time_physical"].cumsum()

    return parsed_df


def run_calculate_cycle_time(conf):

    # 解析机台代码
    print("[INFO] STEP 0: Parse Code & Merge Simulation Result")
    parsed_df = run_code_parsing(conf, save_output=False)
    parsed_df["sub_program"] = parsed_df["sub_program"].astype(str)
    parsed_df["function"] = parsed_df["function"].fillna("__unknown__")
    # parsed_df = pd.read_excel(f"{conf['path']['dir_app']}/{conf['clamping_name']}/{conf['path']['dir_parsed_line']}")

    # 填充 NaN 值以便正確分組，生成 index_at_this_pos
    group_cols = ["sub_program", "N", "move_code", "X", "Y", "Z"]
    parsed_df_filled = parsed_df.copy()
    for col in group_cols:
        if col in parsed_df.columns:
            # 使用特殊值填充 NaN，確保相同的 NaN 值被視為相同的組
            parsed_df_filled[col] = parsed_df_filled[col].fillna("__NAN__")
    parsed_df["index_at_this_pos"] = parsed_df_filled.groupby(group_cols).cumcount()

    all_res_path = f"{conf['path']['dir_app']}/{conf['clamping_name']}/{conf['path']['dir_simulation']}/all_simulated.xlsx"

    simulated_df = pd.read_excel(all_res_path)
    simulated_df["sub_program"] = simulated_df["sub_program"].astype(str).str.zfill(4)
    # except:
    #     simulated_dfs = []
    #     for idx, sub_program in enumerate(conf["sub_programs"].keys(), start=1):
    #         # 读取仿真结果
    #         path = f"{conf['path']['dir_app']}/{conf['clamping_name']}/{conf['path']['dir_simulation']}/{sub_program}.xlsx"
    #         simulated_sub_df = pd.read_excel(path)
    #         simulated_sub_df["sub_program"] = str(sub_program)
    #         simulated_sub_df["sub_program_key"] = str(idx).zfill(2) + "-" + sub_program
    #         simulated_sub_df["sub_program_seq"] = idx
    #         simulated_dfs.append(simulated_sub_df)
    #     simulated_df = pd.concat(simulated_dfs, axis=0)
    #     simulated_df.to_excel(all_res_path, index=False)

    # 填充 NaN 值以便正確分組，生成 index_at_this_pos
    simulated_df_filled = simulated_df.copy()
    for col in group_cols:
        if col in simulated_df.columns:
            simulated_df_filled[col] = simulated_df_filled[col].fillna("__NAN__")
    simulated_df["index_at_this_pos"] = simulated_df_filled.groupby(
        group_cols
    ).cumcount()

    merge_keys = ["sub_program", "N", "move_code", "X", "Y", "Z", "index_at_this_pos"]

    # 選擇 simulated_df 中不在 parsed_df 中的列
    parsed_col = list(parsed_df.columns)
    simulated_col = list(simulated_df.columns)
    additional_cols = [
        x for x in simulated_col if x not in parsed_col and x not in merge_keys
    ]

    # print(sub_program)
    # print("parsed_sub_df", parsed_sub_df.shape, parsed_sub_df.F.mean())
    # print("simulated_df", simulated_df.shape, simulated_df.F.mean())

    try:
        simulated_df = parsed_df.merge(
            simulated_df[merge_keys + additional_cols],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
    except:
        merge_keys = ["sub_program", "move_code", "X", "Y", "Z", "index_at_this_pos"]
        simulated_df = parsed_df.merge(
            simulated_df[merge_keys + additional_cols],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )

    simulated_df = simulated_df.drop_duplicates(merge_keys, keep="last").reset_index(
        drop=True
    )
    simulated_df["src"] = simulated_df["src"].fillna("")

    print("[INFO] STEP 1: Calculate Cycle Time")
    simulated_df = calculate_cycle_time(simulated_df)

    return simulated_df


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    conf = load_config_v1("./cnc_genai/conf/v1_config.yaml")

    all_code_df = run_calculate_cycle_time(conf)
    out_dir = f'{conf["path"]["dir_intermediate"]}/{conf["clamping_name"]}/{conf["output_path"]["calc_time"]}'
    os.makedirs(out_dir, exist_ok=True)
    all_code_df.to_excel(f"{out_dir}/time_analysis.xlsx", index=False)

    print("Time analysis completed!", all_code_df.time_physical.sum())
