import datetime
import time
import re
import pandas as pd
import numpy as np
import os
import yaml
import copy
from cnc_genai.src.code_parsing.macro import MacroManager
from cnc_genai.src.code_parsing.extract_patterns import (
    extract_G04_pattern,
    extract_G05_pattern,
    extract_G10_pattern,
    extract_G30_pattern,
    extract_G53p1_pattern,
    extract_G54p1_pattern,
    extract_G68_pattern,
    extract_G68p2_pattern,
    extract_G81_pattern,
    extract_G82_pattern,
    extract_G83_pattern,
    extract_G84_pattern,
    extract_M98_pattern,
    extract_G100_pattern,
)
from cnc_genai.src.code_parsing.models import CurrentStatus
from cnc_genai.src.utils.utils import load_config_v1, read_gcode_from_json
from cnc_genai.src.utils.utils import load_raw_gcodes
import warnings

warnings.filterwarnings("ignore")


class CodeParser:
    def __init__(self, macro={}, verbose=False):
        """
        初始化CodeParser，设置初始状态，并创建MacroManager实例
        """
        self.current_status = CurrentStatus()
        self.macro_manager = MacroManager(macro, verbose=verbose)
        # 初始化刀具位置，通常用於G28指令（機床原點）
        self.init_tools_pos = {"X": 0, "Y": 0, "Z": 99999}
        # 第二參考點位置，通常用於G30指令（刀具更換位置）
        self.second_ref_pos = {"X": 0, "Y": 0, "Z": 99999}  # Z軸設定為較高的安全位置
        self.verbose = verbose

    @staticmethod
    def _resolve_value(value_str):
        """
        尝试将给定的字符串转换为浮点数，不成功则返回0.0
        """
        try:
            return float(value_str)
        except Exception:
            return 0.0

    def remove_balanced_parentheses(self, s: str) -> str:
        """
        從字串中移除所有完整匹配的括號及其內部內容，
        若遇到不匹配的括號則保留原有符號。

        例如：
          "df(sdfss)" 輸出 "df"
          "(dsdf(sdfss)dds" 輸出 "(dsdfdds"
        """
        # 建立一個和字串長度相同的布林清單，預設每個位置皆保留（True）
        keep = [True] * len(s)
        # 使用堆疊記錄已出現的開括號的索引
        stack = []
        for i, char in enumerate(s):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    start = stack.pop()
                    # 將從開始到目前索引的所有字元標記為 False，即移除
                    for j in range(start, i + 1):
                        keep[j] = False
        # 組合保留的字元並去除首尾空白
        return "".join(s[i] for i in range(len(s)) if keep[i]).strip()

    def indentify_valid_code(self, gcode_line):
        """
        判断该代码行是否无效：
          - 长度为0；
          - 以"if"开头（忽略大小写）；
          - 以'%'开头
          - 以'#'开头，先解析宏变量在判断无效
          - 全部在一對括號內
        """
        if not len(gcode_line):
            return False
        # TODO 暂时只去除了IF行，没有去除对应的代码
        if (gcode_line[:2]).lower() == "if":
            return False
        if gcode_line[0] in "%":
            return False
        if gcode_line[0] in "#":
            return False
        if gcode_line[0] in "/":
            return False
        if "GOTO" in gcode_line:
            return False
        return True

    def resolve_value_in_brackets(self, gcode_line):
        """
        提取計算方括號內的值，例如：
        - X[35+#3] => X[35+7] => X42
        - Z[95.947+-1.0] => Z94.947
        - Y[100.0--2.0] => Y102.0

        宏變量的調用會在此函數之前完成，所以這裡只需要處理數學運算。

        參數:
            gcode_line (str): 包含方括號的G代碼行

        返回:
            str: 處理後的G代碼行，方括號內的運算已被計算結果替換
        """
        # 使用正則表達式找出所有方括號內的內容
        pattern = r"\[(.*?)\]"

        def evaluate_expression(match):
            try:
                # 獲取表達式
                expr = match.group(1).strip()

                # 處理連續的運算符號
                # 將 '+-' 替換為 '-'
                # 將 '--' 替換為 '+'
                while "+-" in expr or "--" in expr:
                    expr = expr.replace("+-", "-")
                    expr = expr.replace("--", "+")

                # 計算表達式
                result = float(eval(expr))

                # 如果結果是整數，返回整數字符串
                if result.is_integer():
                    return str(int(result))
                # 否則返回浮點數字符串，保留3位小數
                return f"{result:.3f}".rstrip("0").rstrip(".")

            except Exception as e:
                # 如果計算失敗，返回原始字符串
                return match.group(0)

        # 替換所有方括號內的表達式為計算結果
        return re.sub(pattern, evaluate_expression, gcode_line)

    def parse_line(
        self,
        row_id,
        gcode_line,
        acc_rotation=[],
        init_rotation=[],
        program_id_list=[],
        verbose=False,
    ):
        """
        解析单行G-code代码：
          - 使用MacroManager替换宏变量（并计算表达式）
          - 按正则表达式解析各个片段，更新self.current_status

        參數：
          - row_id: 行號
          - gcode_line: 原始G-code行
          - acc_rotation: 累積旋轉角度，用於計算旋轉後的坐標
          - verbose: 是否打印詳細信息

        返回：
          - 一個包含多個動作的列表，每個動作包含當前行號、原始代碼、當前片段及部分狀態
          - 累積旋轉角度
          - 當前刀具號碼
        """
        original_line = gcode_line  # 保留原始代码以便记录

        # 去掉空格和括号
        gcode_line = gcode_line.strip().replace(" ", "")
        gcode_line = self.remove_balanced_parentheses(gcode_line)

        # 特殊替换 todo configuratble
        if "G100" in gcode_line:
            gcode_line = extract_G100_pattern(gcode_line)

        # 解析宏变量赋值
        self.macro_manager.parse_macro_assignment(gcode_line)

        # 空行直接返回
        if original_line is None:
            return [], acc_rotation, self.current_status.T

        # 如果是无效行，直接返回一行现状
        current_returning = (
            [
                # getattr(self.current_status, "O", None),
                program_id_list[row_id],
                row_id,
                original_line if original_line is not None else "",
                None,
            ]
            + [
                getattr(self.current_status, field, None)
                for field in [
                    "rotation",
                    "coordinates_abs_rel",
                    "coordinates_sys",
                    "unit",
                    "precision_mode",
                    "move_code",
                    "panel_selected",
                    "call_func",
                    "G04_time",
                    "G54p1_P",
                    "G54p1_X",
                    "G54p1_Y",
                    "G81_Z",
                    "G82_Z",
                    "G82_P",
                    "G83_Z",
                    "G83_Q",
                ]
            ]
            + [
                getattr(self.current_status, each, None)
                for each in "ONGMXYZSFTHDIJKABC"
            ]
        )
        if not self.indentify_valid_code(gcode_line):
            return [current_returning], acc_rotation, self.current_status.T

        # 使用MacroManager替换宏变量为真实值
        gcode_line = self.macro_manager.replace_macros(gcode_line)

        # 去除注释和不需要的字符
        gcode_line = re.sub(r"\(.*?\)", "", gcode_line)

        # 進行[]內的數值替換
        gcode_line = self.resolve_value_in_brackets(gcode_line)

        # 初始化
        actions = []
        skip_count = 0
        parts = re.findall(r"([A-Z](?:#[0-9]+|[-+]?[0-9]*(?:\.[0-9]*)?))", gcode_line)

        # 初始化旋轉
        self.current_status.rotation = copy.deepcopy(acc_rotation)

        for part in parts:

            # 如果需要跳过part
            if skip_count > 0:
                skip_count -= 1
                continue

            # 收集G字段
            if part[0] == "G":
                self.current_status.G = part

            # 高精度加工模式
            if part in ["G05", "G05.1", "G5", "G5.1"]:
                self.current_status.precision_mode, skip_count = extract_G05_pattern(
                    gcode_line
                )

            # 設定偏移
            elif part == "G10":
                self.current_status.G10_L, skip_count = extract_G10_pattern(gcode_line)

            # 取消設定偏移
            elif part == "G11":
                self.current_status.G10_L = None

            # 停驻时间
            elif part[:3] == "G04":
                self.current_status.G04_time, skip_count = extract_G04_pattern(
                    gcode_line
                )

            # 回坐标原点
            elif part == "G28":
                self.current_status.move_code = "G00"
                self.current_status.X = self.init_tools_pos["X"]
                self.current_status.Y = self.init_tools_pos["Y"]
                self.current_status.Z = self.init_tools_pos["Z"]

            # 移動到第二參考點
            elif part == "G30":
                x_val, y_val, z_val, skip_count = extract_G30_pattern(gcode_line)
                self.current_status.move_code = "G00"
                self.current_status.X = self.second_ref_pos["X"]
                self.current_status.Y = self.second_ref_pos["Y"]
                self.current_status.Z = self.second_ref_pos["Z"]

            # 相对坐标，绝对坐标切换
            elif part == "G90":
                self.current_status.coordinates_abs_rel = "absolute"
            elif part == "G91":
                self.current_status.coordinates_abs_rel = "relative"

            # 切換到幾台坐標
            elif part == "G53":
                # TODO 需要知道機台坐標原點相對於工件物理坐標原點的translation
                pass

            # 坐标系切换
            elif part in ["G54", "G55", "G56", "G57", "G58", "G59"]:
                self.current_status.coordinates_sys = part
            elif part == "G54.1":
                # TODO 需要知道機台坐標原點相對於工件物理坐標原點的translation
                p_val, x_val, y_val, z_val, skip_count = extract_G54p1_pattern(
                    gcode_line
                )
                self.current_status.G54p1_P = p_val
                self.current_status.G54p1_X = x_val
                self.current_status.G54p1_Y = y_val
                self.current_status.G54p1_Z = z_val
                skip_count = skip_count
            elif part == "G53.1":
                # TODO 需要知道機台坐標原點相對於工件物理坐標原點的translation
                p_val, x_val, y_val, z_val, skip_count = extract_G53p1_pattern(
                    gcode_line
                )
                self.current_status.G53p1_P = p_val
                self.current_status.G53p1_X = x_val
                self.current_status.G53p1_Y = y_val
                self.current_status.G53p1_Z = z_val
                skip_count = skip_count

            # 加工平面切换
            elif part == "G17":
                self.current_status.panel_selected = "XY"
            elif part == "G18":
                self.current_status.panel_selected = "XZ"
            elif part == "G19":
                self.current_status.panel_selected = "YZ"

            # 单位切换
            elif part == "G20":
                self.current_status.unit = "英制單位"
            elif part == "G21":
                self.current_status.unit = "公制單位"

            # 刀具补偿，暂未开发
            elif part == "G40":
                pass  # 刀具半徑補償關閉
            elif part == "G41":
                pass  # 刀具半徑補償左
            elif part == "G42":
                pass  # 刀具半徑補償右
            elif part == "G43":
                pass  # 刀具長度補償正
            elif part == "G44":
                pass  # 刀具長度補償負
            elif part == "G49":
                pass  # 刀具長度補償關閉

            # 工件旋转
            elif part == "G68":
                x_val, y_val, r_val, skip_count = extract_G68_pattern(gcode_line)
                # 累计一套绕Z轴正方向逆时针的R角度旋转

                # 检查R角度是否为0
                r_angle = CodeParser._resolve_value(r_val) if r_val is not None else 0
                if r_angle != 0:
                    center = [
                        (CodeParser._resolve_value(x_val) if x_val is not None else 0),
                        (CodeParser._resolve_value(y_val) if y_val is not None else 0),
                        0,
                    ]

                    rotation_config = {"Z": {"center": center, "angle": r_angle}}

                    acc_rotation.append(rotation_config)

                self.current_status.rotation = copy.deepcopy(acc_rotation)
            elif part == "G68.2":
                p_val, x_val, y_val, z_val, i_val, j_val, k_val, skip_count = (
                    extract_G68p2_pattern(gcode_line)
                )
                # 累计一套绕XYZ轴正方向逆时针的R角度旋转

                # 计算中心点坐标
                center = [
                    (CodeParser._resolve_value(x_val) if x_val is not None else 0),
                    (CodeParser._resolve_value(y_val) if y_val is not None else 0),
                    (CodeParser._resolve_value(z_val) if z_val is not None else 0),
                ]

                # 构建旋转配置，只添加angle不为0的轴
                rotation_config = {}

                # X軸旋轉 (I值)
                i_angle = CodeParser._resolve_value(i_val) if i_val is not None else 0
                if i_angle != 0:
                    rotation_config["X"] = {"center": center, "angle": i_angle}

                # Y軸旋轉 (J值)
                j_angle = CodeParser._resolve_value(j_val) if j_val is not None else 0
                if j_angle != 0:
                    rotation_config["Y"] = {"center": center, "angle": j_angle}

                # Z軸旋轉 (K值)
                k_angle = CodeParser._resolve_value(k_val) if k_val is not None else 0
                if k_angle != 0:
                    rotation_config["Z"] = {"center": center, "angle": k_angle}

                # 只有當有非零旋轉時才添加到acc_rotation
                if rotation_config:
                    acc_rotation.append(rotation_config)

                self.current_status.rotation = copy.deepcopy(acc_rotation)
            elif part == "G69":
                self.current_status.rotation = copy.deepcopy(init_rotation)
                acc_rotation = copy.deepcopy(init_rotation)

            # 四轴机台通过ABC旋转
            # TODO to revise 与MAC3的逻辑协同，包括center在mac3并不一定为0,0,0
            elif part[0] in "ABC":
                setattr(
                    self.current_status, part[0], CodeParser._resolve_value(part[1:])
                )
                rotation_config = {}
                if self.current_status.A != 0 and self.current_status.A != None:
                    rotation_config["X"] = {
                        "center": [0, 0, 0],
                        "angle": self.current_status.A,
                    }
                if self.current_status.B != 0 and self.current_status.B != None:
                    rotation_config["Y"] = {
                        "center": [0, 0, 0],
                        "angle": self.current_status.B,
                    }
                if self.current_status.C != 0 and self.current_status.C != None:
                    rotation_config["Z"] = {
                        "center": [0, 0, 0],
                        "angle": self.current_status.C,
                    }
                if rotation_config != {}:
                    acc_rotation = [rotation_config]
                self.current_status.rotation = copy.deepcopy(acc_rotation)

            # 加工
            elif part in [
                "G0",
                "G1",
                "G2",
                "G3",
                "G4",
                "G00",
                "G01",
                "G02",
                "G03",
                "G04",
            ]:
                self.current_status.move_code = (
                    part if len(part) == 3 else f"G0{part[1]}"
                )

            elif part == "G81":  # 鑽孔
                self.current_status.move_code = part
                self.current_status.G98_Z = self.current_status.Z_prev
                x_val, y_val, origin_z, final_z, r_val, f_val, skip_count = (
                    extract_G81_pattern(gcode_line, self.current_status)
                )
                self.current_status.G81_Z = (
                    CodeParser._resolve_value(origin_z)
                    if origin_z is not None
                    else None
                )
                self.current_status.X = (
                    CodeParser._resolve_value(x_val) if x_val is not None else None
                )
                self.current_status.Y = (
                    CodeParser._resolve_value(y_val) if y_val is not None else None
                )
                self.current_status.Z = (
                    CodeParser._resolve_value(final_z) if final_z is not None else None
                )
                self.current_status.F = (
                    CodeParser._resolve_value(f_val) if f_val is not None else None
                )

            elif part == "G82":  # 沉孔
                self.current_status.move_code = part
                self.current_status.G98_Z = self.current_status.Z_prev
                x_val, y_val, origin_z, final_z, r_val, p_val, f_val, skip_count = (
                    extract_G82_pattern(gcode_line, self.current_status)
                )
                self.current_status.G82_Z = (
                    CodeParser._resolve_value(origin_z)
                    if origin_z is not None
                    else None
                )
                self.current_status.X = (
                    CodeParser._resolve_value(x_val) if x_val is not None else None
                )
                self.current_status.Y = (
                    CodeParser._resolve_value(y_val) if y_val is not None else None
                )
                self.current_status.Z = (
                    CodeParser._resolve_value(final_z) if final_z is not None else None
                )
                self.current_status.F = (
                    CodeParser._resolve_value(f_val) if f_val is not None else None
                )
                # 特殊参数P，停留时间
                self.current_status.G82_P = (
                    CodeParser._resolve_value(p_val) if p_val is not None else None
                )

            elif part == "G83":  # 深孔
                self.current_status.move_code = part
                self.current_status.G98_Z = self.current_status.Z_prev
                x_val, y_val, origin_z, final_z, r_val, q_val, f_val, skip_count = (
                    extract_G83_pattern(gcode_line, self.current_status)
                )
                self.current_status.G83_Z = (
                    CodeParser._resolve_value(origin_z)
                    if origin_z is not None
                    else None
                )
                self.current_status.X = (
                    CodeParser._resolve_value(x_val) if x_val is not None else None
                )
                self.current_status.Y = (
                    CodeParser._resolve_value(y_val) if y_val is not None else None
                )
                self.current_status.Z = (
                    CodeParser._resolve_value(final_z) if final_z is not None else None
                )
                self.current_status.F = (
                    CodeParser._resolve_value(f_val) if f_val is not None else None
                )
                # 特殊参数Q，每次切入深度
                self.current_status.G83_Q = (
                    CodeParser._resolve_value(q_val) if q_val is not None else None
                )

            elif part == "G84":
                self.current_status.move_code = part
                self.current_status.G98_Z = self.current_status.Z_prev
                x_val, y_val, origin_z, final_z, r_val, q_val, f_val, skip_count = (
                    extract_G84_pattern(gcode_line, self.current_status)
                )
                self.current_status.G84_Z = (
                    CodeParser._resolve_value(origin_z)
                    if origin_z is not None
                    else None
                )
                self.current_status.X = (
                    CodeParser._resolve_value(x_val) if x_val is not None else None
                )
                self.current_status.Y = (
                    CodeParser._resolve_value(y_val) if y_val is not None else None
                )
                self.current_status.Z = (
                    CodeParser._resolve_value(final_z) if final_z is not None else None
                )
                self.current_status.F = (
                    CodeParser._resolve_value(f_val) if f_val is not None else None
                )
                # 特殊参数Q，每次切入深度
                self.current_status.G84_Q = (
                    CodeParser._resolve_value(q_val) if q_val is not None else None
                )

            elif part == "G80":
                pass  # 終止固定循環

            elif part == "G98":  # 控制鑽孔循環結束後的Z軸返回位置
                self.current_status.move_code = "G00"
                self.current_status.Z = self.current_status.G98_Z

            elif part == "G99":  # 控制鑽孔循環中，讓刀具只返回到R平面（快速進給平面）
                self.current_status.move_code = "G00"
                self.current_status.Z = self.current_status.G99_R

            # G01G02G03状态下移动位置
            elif part[0] in "XYZ":
                # assert self.current_status.move_code in [
                #     "G00",
                #     "G01",
                #     "G02",
                #     "G03",
                # ], f"Error move_code {self.current_status.move_code} in {gcode_line}"
                if self.current_status.coordinates_abs_rel == "absolute":
                    setattr(
                        self.current_status,
                        part[0],
                        CodeParser._resolve_value(part[1:]),
                    )
                else:
                    current_val = getattr(self.current_status, part[0], 0) or 0
                    setattr(
                        self.current_status,
                        part[0],
                        current_val + CodeParser._resolve_value(part[1:]),
                    )

            # 函数段切换和标记
            elif part[0] in "ONM":
                setattr(self.current_status, part[0], part)

            # 刀具更换 - 必須有M06才能更換刀具
            elif part[0] in "T":
                # 檢查同一行是否包含M06指令
                if "M06" in gcode_line or "M6" in gcode_line:
                    try:
                        # 提取刀具號碼並格式化
                        tool_number = part[1:]
                        setattr(
                            self.current_status,
                            "T",
                            "T" + tool_number,
                        )
                        print(f"[INFO] 更換刀具{self.current_status.T} ({gcode_line})")
                    except:
                        print(f"[ERROR] 刀具更換指令格式錯誤: {part} in {gcode_line}")
                else:
                    if verbose:
                        print(f"[INFO] 沒有M06，預換刀: {part} in {gcode_line}")

            # 其他标志字段
            elif part[0] in "IJKSFHD":
                setattr(
                    self.current_status, part[0], CodeParser._resolve_value(part[1:])
                )

            # 提示未知字段
            else:
                # pass
                if verbose:
                    print(f"[INFO] 暂不解析的指令: {part} in {gcode_line}")

            actions.append(
                # [getattr(self.current_status, "O", None), row_id, original_line, part]
                [program_id_list[row_id], row_id, original_line, part]
                + [
                    getattr(self.current_status, field, None)
                    for field in [
                        "rotation",
                        "coordinates_abs_rel",
                        "coordinates_sys",
                        "unit",
                        "precision_mode",
                        "move_code",
                        "panel_selected",
                        "call_func",
                        "G04_time",
                        "G54p1_P",
                        "G54p1_X",
                        "G54p1_Y",
                        "G81_Z",
                        "G82_Z",
                        "G82_P",
                        "G83_Z",
                        "G83_Q",
                    ]
                ]
                + [
                    getattr(self.current_status, each, None)
                    for each in "ONGMXYZSFTHDIJKABC"
                ]
            )

        # 更新上一组XYZ坐标
        for each in "XYZ":
            setattr(
                self.current_status,
                f"{each}_prev",
                getattr(self.current_status, each, None),
            )
        return actions, acc_rotation, self.current_status.T

    def recursively_replace_sub_program(
        self, gcode, gcodes_dict, current_subprogram_id=None
    ):
        """
        遞迴替換子程序:
        如果遇到一行符合 "M98Pxxx" 格式的子程序呼叫，
        從指定目錄中讀取對應的 "xxx.txt" 檔案，
        將該行替換為檔案內容的各行（按換行符拆分）。
        如果替換後的內容中還有子程序呼叫，也會進行遞迴處理。

        參數:
          gcode (str): 原始的G-code字串
          gcodes_dict (dict): 供调用的子程序
          current_subprogram_id (str): 當前子程式ID

        回傳:
          str: 替換後的完整G-code字串
          list: 每一行代码的src来自于哪个子程式/主程式的程式号
        """
        lines = gcode.splitlines()
        final_lines = []
        subprogram_id_list = []
        for line in lines:
            # 使用 extract_M98_pattern 檢查是否為完整的 M98Pxxx 呼叫行
            subprogram_id = extract_M98_pattern(line)
            if subprogram_id:
                try:
                    subprogram_code = gcodes_dict[f"O{subprogram_id.zfill(4)}"]
                except:
                    print(f"[ERROR] 子程序O{subprogram_id.zfill(4)}不存在")
                    subprogram_code = ""
                # 遞迴處理子程序內容，以替換其中可能存在的其他子程序呼叫
                replaced_subprogram_code, replaced_subprogram_id_list = (
                    self.recursively_replace_sub_program(
                        subprogram_code, gcodes_dict, f"O{subprogram_id.zfill(4)}"
                    )
                )
                # 將替換後的子程序內容逐行加入最終結果
                final_lines.extend(replaced_subprogram_code.splitlines())
                # 將對應的子程式ID列表加入
                subprogram_id_list.extend(replaced_subprogram_id_list)
                if self.verbose:
                    print(
                        f"[INFO] 将M98P{subprogram_id}替换为{len(replaced_subprogram_code.splitlines())}行 | {line}"
                    )
            else:
                final_lines.append(line)
                # 為每一行添加對應的子程式ID
                subprogram_id_list.append(current_subprogram_id)
        return "\n".join(final_lines), subprogram_id_list


def structure_cnc_code(
    gcode,
    gcodes_dict,
    inspect_sub_program=False,
    main_program_id=None,
    macro={},
    init_rotation=[],
    verbose=False,
    output_df=False,
    **default_settings,
):
    """
    解析和结构化CNC代码。

    接收一个包含CNC代码的字符串，输出一个结构化的DataFrame，其中包含了代码的详细分解。

    参数:
    gcode (str): 一个包含CNC代码的字符串，主程式。
    gcodes_dict (dict): 供调用的子程序，字典。
    inspect_sub_program (bool): 是否递归替换子程序，默认为False。
    main_program_id (str): 主程式ID，用于在DataFrame中添加一列"sub_program"。
    macro (dict): 初始宏变量，默认为空。
    init_rotation (list): 初始旋转，由程式单定义。
    verbose (bool): 是否输出详细信息，默认为False。
    output_df (bool): 是否输出DataFrame，默认为False，用于在解析主程式时节约时间。
    **default_settings: 其他默认设置。

    返回:
    DataFrame: 一个结构化的DataFrame，包含了代码的详细分解。
    """

    code_parser = CodeParser(macro=macro)

    # 如果提供了子程序所在目錄，則先進行遞迴替換子程序
    if inspect_sub_program:
        gcode, program_id_list = code_parser.recursively_replace_sub_program(
            gcode, gcodes_dict, current_subprogram_id=main_program_id
        )
        length = len(gcode.split("\n"))
        if verbose:
            print(f"[INFO] 替换所有M98，最终得到{length}行代码")
    else:
        program_id_list = [main_program_id] * len(gcode.split("\n"))
    assert len(program_id_list) == len(
        gcode.split("\n")
    ), f"[ERROR] program_id_list length"

    lines = gcode.split("\n")
    structured_code = []

    # 出现的刀具列表
    tool_list = []

    # 每个子程式/主程式的初始旋转，由程式单定义。
    acc_rotation = copy.deepcopy(init_rotation)

    # 逐行解析代码，并保存到structured_code列表中

    start_time = time.time()
    last_report_time = start_time

    for row_id, gcode_line in enumerate(lines):
        actions, acc_rotation, current_tool = code_parser.parse_line(
            row_id,
            gcode_line,
            acc_rotation,
            init_rotation,
            program_id_list,
            verbose=verbose,
        )
        if current_tool not in tool_list:
            tool_list.append(current_tool)
        if output_df:
            structured_code.extend(actions)

        # 每1000行報告一次時間
        if (row_id + 1) % 1000 == 0:
            current_time = time.time()
            elapsed_since_last = current_time - last_report_time
            total_elapsed = current_time - start_time
            avg_time_per_1000 = elapsed_since_last
            if verbose:
                print(
                    f"[INFO] 已處理 {row_id + 1} 行，本批次耗時: {avg_time_per_1000:.2f}秒，總耗時: {total_elapsed:.2f}秒，平均速度: {1000/avg_time_per_1000:.1f}行/秒"
                )
            last_report_time = current_time

    # 最終時間統計
    if verbose:
        final_time = time.time()
        total_time = final_time - start_time
        total_lines = len(lines)
        avg_speed = total_lines / total_time if total_time > 0 else 0
        print(
            f"[INFO] 代碼解析完成，總共處理 {total_lines} 行，總耗時: {total_time:.2f}秒，平均速度: {avg_speed:.1f}行/秒"
        )

    if output_df:
        df = pd.DataFrame(
            structured_code,
            columns=[
                "sub_program",
                "row_id",
                "src",
                "command",
                "rotation",
                "coordinates_abs_rel",
                "coordinates_sys",
                "unit",
                "precision_mode",
                "move_code",
                "panel_selected",
                "call_func",
                "G04_time",
                "G54p1_P",
                "G54p1_X",
                "G54p1_Y",
                "G81_Z",
                "G82_Z",
                "G82_P",
                "G83_Z",
                "G83_Q",
            ]
            + [x for x in "ONGMXYZSFTHDIJKABC"],
        )
        for each in "XYZFSHDIJKABC":
            df[each] = df[each].astype(float)
        df["command_code"] = df["command"].str[0]

        # 假設初始化刀頭位置
        df["X"] = df["X"].fillna(code_parser.init_tools_pos["X"])
        df["Y"] = df["Y"].fillna(code_parser.init_tools_pos["Y"])
        df["Z"] = df["Z"].fillna(
            code_parser.init_tools_pos["Z"]
        )  # 无限高的初始抬刀距离

        df["F"] = np.where(
            df["move_code"] == "G00",
            default_settings.get("rapid_speed", 15000),
            df["F"],
        )
    else:
        # 如果不需要DataFrame，創建空的DataFrame
        df = pd.DataFrame()

    return df, code_parser.macro_manager, tool_list


def prepare_parsing(config, verbose=False):
    clamping = config["clamping_name"]
    print("-" * 30)
    print(
        f"[INFO] 準備代碼解析使用的夾位名稱: {clamping} - {config['path']['dir_machine_folder']}"
    )
    gcodes_dict = load_raw_gcodes(config, verbose=verbose)

    if config["path"]["dir_machine_folder"] == "製工標準":
        base_path = f"{config['path']['dir_app']}/{clamping}"
        # ../app/mac1/simulation_master/X2867-CNC2
    else:
        base_path = os.path.dirname(config["path"]["dir_machine_data"])
        # ../cnc_data/mac3/nc_code/B05-2F#Diamond-Cell#CNC6#P13#2025-02-26

    os.makedirs(
        f"{base_path}/{config['path']['dir_parsed']}",
        exist_ok=True,
    )
    os.makedirs(
        f"{base_path}/{config['path']['dir_macro']}",
        exist_ok=True,
    )

    # 初始宏变量
    try:
        machine_macro_df = pd.read_excel(
            f"{base_path}/{config['path']['machine_macro_path']}"
        )
        machine_macro_df.columns = ["宏變量", "取值"]
        machine_macro_df = machine_macro_df[
            machine_macro_df["宏變量"].str.contains("#")
        ]

        machine_macro = dict(zip(machine_macro_df["宏變量"], machine_macro_df["取值"]))
        if verbose:
            print("[INFO] 讀取設備記錄的宏變量")
    except:
        if verbose:
            print("[WARN] 未讀取設備記錄的宏變量")
        machine_macro = {}

    # 第一段子程序前宏变量的变化
    try:
        gcode = open(
            f"{base_path}/{config['path']['init_macro_code_path']}",
            "r",
        ).read()

        _, macro, _ = structure_cnc_code(
            gcode,
            gcodes_dict=gcodes_dict,
            macro=machine_macro,
            inspect_sub_program=True,
            init_rotation=[],
            main_program_id=None,
            verbose=verbose,
            output_df=False,
        )
        macro.save_to_yaml(f"{base_path}/{config['path']['dir_macro']}/init.yaml")
        if verbose:
            print("[INFO] 運行第一個子程式之前代碼，並保存宏變量到yaml...")
    except:
        macro = machine_macro
        if verbose:
            print("[WARN] 未運行第一個子程式之前的代碼，使用宏變量初始化...")

    print(f"[INFO] 準備代碼解析完成")
    print("-" * 30)

    return base_path, macro


def run_code_parsing(config, verbose=True, save_output=True):
    config["path"]["dir_app"] = config["path"]["dir_app"].format(
        department=config["department"]
    )

    clamping = config["clamping_name"]
    base_path, macro = prepare_parsing(config, verbose=verbose)

    # 准备原始代码
    funcs = pd.read_excel(
        f"{config['path']['dir_app']}/{clamping}/{config['path']['master_path']}"
    )
    funcs["rotation_Y_axis"] = None

    # 检查sub_program是否在列名中
    if "sub_program" not in funcs.columns:
        # 如果不在，则用第二行作为表头，跳过第一行
        funcs = pd.read_excel(
            f"{config['path']['dir_app']}/{clamping}/{config['path']['master_path']}",
            header=1,
        )

    # 格式化子程式名稱
    funcs["sub_program"] = funcs["sub_program"].astype(int).astype(str).str.zfill(4)
    funcs = funcs.sort_values(by=["sub_program_key"])
    funcs["sub_program_last"] = funcs["sub_program"].shift(1)

    # 加載子程式代碼
    if verbose:
        print(f"[INFO] 正式開始解析代碼，共{len(funcs)}個子程式")
    gcodes_dict = load_raw_gcodes(config, verbose=verbose)
    dfs = []
    unsolved_macros = {}

    for idx, row in funcs.iterrows():
        if verbose:
            print("-" * 30)
            print(f"[INFO] 解析代碼: {row['sub_program']} - {row['function']}")
        try:
            gcode = gcodes_dict[f'O{row["sub_program"]}']
        except:
            raise ValueError(f'[ERROR] 找不到子程式O{row["sub_program"]}')

        # 初始化旋轉，即代碼外的治具擺放方向
        # FXN的规则：
        # 制工口径四轴旋转正方向，即工件绕X正方向顺时针旋转，等价于点绕X正方向逆时针旋转，所以需要取相反
        # 制工口径0.5轴旋转正方向，即工件绕Z正方向逆时针旋转，等价于点绕Z正方向顺时针旋转，不需要取相反
        if "rotation_Y_axis" not in row.index:
            row["rotation_Y_axis"] = None

        def parse_rotation_param(param_value, axis_multiplier=1):
            """
            解析旋轉參數，格式為 "angle@x/y/z"
            返回 {"center": [x, y, z], "angle": angle} 或 None（如果angle為0）
            """
            if pd.notna(param_value):
                angle = float(param_value.split("@")[0]) * axis_multiplier
                center = [float(x) for x in param_value.split("@")[1].split("/")]
                return {"center": center, "angle": angle} if angle != 0 else None
            return None

        # 构建旋轉配置，過濾掉angle為0的軸
        rotation_config = {}

        # X軸：四軸旋轉（需要取相反）
        x_rotation = parse_rotation_param(row["rotation_4th_axis"], -1)
        if x_rotation:
            rotation_config["X"] = x_rotation

        # Y軸：Y軸旋轉
        y_rotation = parse_rotation_param(row["rotation_Y_axis"], 1)
        if y_rotation:
            rotation_config["Y"] = y_rotation

        # Z軸：0.5軸旋轉
        z_rotation = parse_rotation_param(row["rotation_0.5_axis"], 1)
        if z_rotation:
            rotation_config["Z"] = z_rotation

        # 如果所有軸的angle都為0，則返回空列表；否則返回包含配置的列表
        init_rotation = [rotation_config] if rotation_config else []

        # 解析代碼
        df, macro_manager, _ = structure_cnc_code(
            gcode,
            gcodes_dict=gcodes_dict,
            macro=macro,
            init_rotation=init_rotation,
            main_program_id=row["sub_program"],
            verbose=verbose,
            output_df=True,
        )

        # 保存子程式宏變量到yaml
        macro_manager.save_to_yaml(
            f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_macro']}/{row['sub_program']}.yaml"
        )
        unsolved_macros[row["sub_program"]] = macro_manager.unsolved_macros
        macro = macro_manager.macros
        df["sub_program"] = str(row["sub_program"])
        df["function"] = row["function"]
        df["real_ct"] = row["real_ct"]

        dfs.append(df)

    # 合并所有子程式代碼
    df_out = pd.concat(dfs, axis=0).reset_index(drop=True)
    df_line = df_out.drop_duplicates(["sub_program", "row_id", "src"], keep="last")

    # 打印XYZ范圍
    if verbose:
        print("-" * 30)
        print("[INFO] XYZ Range")
        print("X", df_line["X"].min(), df_line["X"].max())
        print("Y", df_line["Y"].min(), df_line["Y"].max())
        print("Z", df_line["Z"].min(), df_line[df_line["Z"] < 99999]["Z"].max())
        print("-" * 30)

    # 生成代碼段列表
    df_program_segments = df_out[["sub_program", "function", "N"]].drop_duplicates(
        ["sub_program", "N"], keep="last"
    )
    df_program_segments = df_program_segments[~df_program_segments["N"].isna()]
    df_program_segments.columns = ["sub_program", "function", "n"]

    if save_output:

        # 保存代碼段到excel
        df_program_segments.to_excel(
            f"{config['path']['dir_app']}/{clamping}/{config['path']['program_segments_path']}",
            index=False,
        )
        if verbose:
            print(
                f"[INFO] 保存代碼段到{config['path']['dir_app']}/{clamping}/{config['path']['program_segments_path']}"
            )

        # 保存代碼指令解析結果到excel
        try:
            if not os.path.exists(
                f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_command']}"
            ):
                df_out.to_excel(
                    f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_command']}",
                    index=False,
                )
                if verbose:
                    print(
                        f"[INFO] 保存代碼指令解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_command']}"
                    )
        except Exception as e:
            print(
                f"[ERROR] 保存代碼指令解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_command']} 錯誤: {e}"
            )

        # 保存代碼指令解析結果到parquet
        try:
            df_out.to_parquet(
                f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_command'].replace('.xlsx', '.parquet')}",
                index=False,
            )
            if verbose:
                print(
                    f"[INFO] 保存代碼指令解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_command'].replace('.xlsx', '.parquet')}"
                )
        except Exception as e:
            print(
                f"[ERROR] 保存代碼指令解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_command'].replace('.xlsx', '.parquet')} 錯誤: {e}"
            )

        # 保存代碼行解析結果到excel
        try:
            if not os.path.exists(
                f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_line']}"
            ):
                df_line.to_excel(
                    f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_line']}",
                    index=False,
                )
                if verbose:
                    print(
                        f"[INFO] 保存代碼行解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_line']}"
                    )
        except Exception as e:
            print(
                f"[ERROR] 保存代碼行解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_line']} 錯誤: {e}"
            )

        # 保存代碼行解析結果到parquet
        try:
            df_line.to_parquet(
                f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_line'].replace('.xlsx', '.parquet')}",
                index=False,
            )
            if verbose:
                print(
                    f"[INFO] 保存代碼行解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_line'].replace('.xlsx', '.parquet')}"
                )
        except Exception as e:
            print(
                f"[ERROR] 保存代碼行解析結果到{config['path']['dir_app']}/{clamping}/{config['path']['dir_parsed_line'].replace('.xlsx', '.parquet')} 錯誤: {e}"
            )

        # 保存未解決的宏變量到yml
        with open(
            f"{config['path']['dir_app']}/{clamping}/{config['path']['dir_macro']}/unsolved_macro.yml",
            "w",
            encoding="utf-8",
        ) as f:
            yaml.dump(unsolved_macros, f, allow_unicode=True)
        if verbose:
            print(
                f"[INFO] 保存未解決的宏變量到{config['path']['dir_app']}/{clamping}/{config['path']['dir_macro']}/unsolved_macro.yml"
            )

    return df_line


if __name__ == "__main__":

    conf = load_config_v1("./cnc_genai/conf/v1_config.yaml")
    _ = run_code_parsing(conf)
