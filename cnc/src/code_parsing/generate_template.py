import pandas as pd
import re
import os
import time
from typing import List
from cnc_genai.src.utils.utils import load_raw_gcodes
from cnc_genai.src.code_parsing.macro import MacroManager
from cnc_genai.src.code_parsing.utils import find_tools_spec
from cnc_genai.src.code_parsing.code_parsing import CodeParser


def generate_sub_program_master(raw_main: str = "some_path/O5000.txt") -> pd.DataFrame:
    """
    生成表一，作为模版让制工下载填写
    - 只考虑主入口的子程序号
    - 表头：
    sub_program_key | sub_program | func | mode | tool | tool_offset | tool_spec | rotation | rotation_4th_axis | rotation_0.5_axis | ct_macro | real_ct | is_supporting | break_tool_check | chip_detection | speed | feed_rate | coordinate_system | tool_length | tool_holder | remark
    """

    # 表頭說明
    hint_sub_program_key = "[無需填寫]自動生成"
    hint_sub_program = "[無需填寫]子程式編號，從主程序解析自動生成"
    hint_func = '[選填]請填寫該段子程序的功能，例如"內腔開粗"等'
    hint_mode = "[選填]請填寫該段子程序的加工模式"
    hint_tool = (
        '[選填]請填寫該段子程序使用的刀具編號，例如"T02"，注意盡量寫成"T02"而非"T2"'
    )
    hint_tool_offset = "[選填]請填寫刀具補正"
    hint_tool_spec = "[選填]請填寫刀具規格型號"
    hint_rotation_4th_axis = "[必填]請填寫治具繞第4軸旋轉的角度和旋轉中心，以沿X軸正方向視角逆時針為正，例如繞點(0,0,0)為中心，沿X軸正方向逆時針旋轉90度，則填入90@0/0/0"
    hint_rotation_0p5_axis = "[必填]請填寫治具繞第0.5軸旋轉的角度和旋轉中心，以沿Z軸正方向視角順時針為正，例如繞點(0,0,0)為中心，沿Z軸正方向逆時針旋轉90度，則填入-90@0/0/0"
    hint_rotation = "[選填]新版功能，用一個列表填寫XYZ各軸的初始旋轉位置，例如[{'X': {'center': [0, 0, 0], 'angle': 0}, 'Y': {'center': [0, 0, 0], 'angle': 0}, 'Z': {'center': [0, 0, 0], 'angle': 0}}]"
    hint_ct_macro = "[選填]標准代碼中的CT宏變量"
    hint_ct = "[選填]標準CT時間，用以計算提升量"
    hint_is_supporting = "[必填]請填寫該段子程序是否為輔助子程式"
    hint_break_tool_check = "[選填]請填寫斷刀檢測"
    hint_chip_detection = "[選填]請填寫夾屑檢測"
    hint_speed = "[選填]請填寫主軸轉速"
    hint_feed_rate = "[選填]請填寫進給速度"
    hint_coordinate_system = "[選填]請填寫坐標系"
    hint_tool_length = "[選填]請填寫刀具伸出長度"
    hint_tool_holder = "[選填]請填寫刀把型號"
    hint_remark = "[選填]請填寫備註"

    if not os.path.exists(raw_main):
        print(f"文件 {raw_main} 不存在")
        return (
            pd.DataFrame(
                {
                    "sub_program_key": [hint_sub_program_key, "sub_program_key"],
                    "sub_program": [hint_sub_program, "sub_program"],
                    "function": [hint_func, "function"],
                    "mode": [hint_mode, "mode"],
                    "tool": [hint_tool, "tool"],
                    "tool_offset": [hint_tool_offset, "tool_offset"],
                    "tool_spec": [hint_tool_spec, "tool_spec"],
                    "rotation": [hint_rotation, "rotation"],
                    "rotation_4th_axis": [hint_rotation_4th_axis, "rotation_4th_axis"],
                    "rotation_0.5_axis": [hint_rotation_0p5_axis, "rotation_0.5_axis"],
                    "ct_macro": [hint_ct_macro, "ct_macro"],
                    "real_ct": [hint_ct, "real_ct"],
                    "is_supporting": [hint_is_supporting, "is_supporting"],
                    "break_tool_check": [hint_break_tool_check, "break_tool_check"],
                    "chip_detection": [hint_chip_detection, "chip_detection"],
                    "speed": [hint_speed, "speed"],
                    "feed_rate": [hint_feed_rate, "feed_rate"],
                    "coordinate_system": [hint_coordinate_system, "coordinate_system"],
                    "tool_length": [hint_tool_length, "tool_length"],
                    "tool_holder": [hint_tool_holder, "tool_holder"],
                    "remark": [hint_remark, "remark"],
                }
            ),
            None,
        )

    # 讀取文件內容
    with open(raw_main, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 使用正則表達式找出所有 M98P 後面的數字
    pattern = re.compile(r"M98P(\d+)")
    matches = pattern.findall(content)

    # 不去重不排序
    sub_programs = list(matches)

    # 將子程序號轉換為四位字符串，不夠四位則在前面補零
    sub_programs = [str(prog).zfill(4) for prog in sub_programs]
    sub_programs = [x for x in sub_programs if not x.startswith("0")]
    sub_program_key = [
        f"{str(idx+1).zfill(2)}-{str(sub_programs[idx]).zfill(4)}"
        for idx in range(len(sub_programs))
    ]

    # 從原始路徑中提取主程序號
    basename = os.path.basename(raw_main)
    main_program = basename.replace("O", "").replace(".txt", "")

    # 尝试在住程序中找到子程序使用的刀具信息
    tools = []
    tool_specs = []
    is_supportings = []
    for each in sub_programs:
        # 構建子程序路徑，用子程序號替換主程序號
        sub_program_path = raw_main.replace(f"O{main_program}.txt", f"O{each}.txt")
        is_supporting = check_is_supporting(sub_program_path)
        tool, tool_spec = find_tools_spec(each, content)
        tools.append(tool)
        tool_specs.append(tool_spec)
        is_supportings.append(is_supporting)
    # 創建 DataFrame
    df = pd.DataFrame(
        {
            "sub_program_key": [hint_sub_program_key]
            + ["sub_program_key"]
            + sub_program_key,
            "sub_program": [hint_sub_program] + ["sub_program"] + sub_programs,
            "function": [hint_func] + ["function"] + [""] * len(sub_programs),
            "mode": [hint_mode] + ["mode"] + [""] * len(sub_programs),
            "tool": [hint_tool] + ["tool"] + tools,
            "tool_offset": [hint_tool_offset]
            + ["tool_offset"]
            + [""] * len(sub_programs),
            "tool_spec": [hint_tool_spec] + ["tool_spec"] + tool_specs,
            "rotation_4th_axis": [hint_rotation_4th_axis]
            + ["rotation_4th_axis"]
            + [""] * len(sub_programs),
            "rotation_0.5_axis": [hint_rotation_0p5_axis]
            + ["rotation_0.5_axis"]
            + [""] * len(sub_programs),
            "ct_macro": [hint_ct_macro] + ["ct_macro"] + [""] * len(sub_programs),
            "real_ct": [hint_ct] + ["real_ct"] + [""] * len(sub_programs),
            "is_supporting": [hint_is_supporting] + ["is_supporting"] + is_supportings,
            "break_tool_check": [hint_break_tool_check]
            + ["break_tool_check"]
            + [""] * len(sub_programs),
            "chip_detection": [hint_chip_detection]
            + ["chip_detection"]
            + [""] * len(sub_programs),
            "speed": [hint_speed] + ["speed"] + [""] * len(sub_programs),
            "feed_rate": [hint_feed_rate] + ["feed_rate"] + [""] * len(sub_programs),
            "coordinate_system": [hint_coordinate_system]
            + ["coordinate_system"]
            + [""] * len(sub_programs),
            "tool_length": [hint_tool_length]
            + ["tool_length"]
            + [""] * len(sub_programs),
            "tool_holder": [hint_tool_holder]
            + ["tool_holder"]
            + [""] * len(sub_programs),
            "remark": [hint_remark] + ["remark"] + [""] * len(sub_programs),
        }
    )

    # 刀具规格映射表
    tool_mapper = dict(zip(tools, tool_specs))

    return df, tool_mapper


def generate_macros_and_tools(
    config, macros: pd.DataFrame, main_func: str, funcs: str, tool_mapper: dict
):

    # 初始宏变量
    try:
        macro = dict(zip(macros["宏變量"], macros["取值"]))
        print("[INFO] 讀取設備記錄的宏變量...")
    except:
        print("[WARN] 未讀取設備記錄的宏變量...")
        macro = {}

    gcode_main = open(main_func, "r").read()

    gcodes_dict = load_raw_gcodes(config)

    print("[INFO] 解析遞歸後的主程式")
    t0 = time.time()
    # TODO 提速
    # _, macro_manager, tool_list = structure_cnc_code(
    #     gcode_main,
    #     gcodes_dict=gcodes_dict,
    #     macro=macro,
    #     inspect_sub_program=True,  # 審查所有被call的子程序
    #     verbose=True,
    #     output_df=False,
    # )
    macro_manager = CodeParser().macro_manager
    tool_list = []
    t1 = time.time()
    print(f"[INFO] 解析遞歸後的主程式完成，耗時{t1-t0}秒")

    macros_df = generate_macros_needed(macro_manager)
    tool_df = generate_tools(tool_list, tool_mapper)

    return macros_df, tool_df


def generate_macros_needed(macro_manager: MacroManager) -> pd.DataFrame:
    """
    生成宏變量模板表格
    - 包含所有宏變量
    - 先寫所有未解析的宏變量，value為空
    - 再寫所有已知宏變量，包含k-v對
    - 表頭：macro | value

    Args:
        macro_manager: MacroManager 實例，包含已解析和未解析的宏變量

    Returns:
        pd.DataFrame: 包含宏變量的DataFrame
    """

    hint_macro = "[無需填寫]\n解析得到代碼中出現過的宏變量"
    hint_value = "[選填]\n該宏變量的取值"

    # 處理未解析的宏變量
    unsolved_macros_list = list(macro_manager.unsolved_macros)
    df_unsolved = pd.DataFrame(
        {"macro": unsolved_macros_list, "value": [None] * len(unsolved_macros_list)}
    )

    # 處理已解析的宏變量
    solved_macros = macro_manager.macros
    if solved_macros:
        # 將字典轉換為DataFrame
        df_solved = pd.DataFrame(
            {"macro": list(solved_macros.keys()), "value": list(solved_macros.values())}
        )
    else:
        df_solved = pd.DataFrame(columns=["macro", "value"])

    # 合併兩個DataFrame
    out_df = pd.concat([df_unsolved, df_solved], ignore_index=True)
    macro_df = pd.DataFrame(
        {
            "宏變量": [hint_macro, "宏變量"] + list(out_df["macro"]),
            "取值": [hint_value, "取值"] + list(out_df["value"]),
        }
    )
    return macro_df


def generate_tools(tool_list: list, tool_mapper: dict) -> pd.DataFrame:
    """
    生成表三，刀具
    - 包含所有换刀刀具
    - optional 刀具规格根据括号提示
    - 表头：
    刀號 | 刀具規格 | 刀頭直徑 | 刀頭高度 | 是否為T刀 | 是否有R角
    """

    hint_刀號 = "[無需填寫]\n從主程序換刀指令提取的刀號"
    hint_刀具規格 = (
        "[選填]\n使用該刀具的道具規格，例如：W-DJ5232-2(A)(d1.73H5.8D4A135*2F*75)"
    )
    hint_刀頭直徑 = "[必填]\n該刀具切割部分的直徑，單位mm"
    hint_刀頭高度 = "[必填]\n該刀具切割部分的高度，即刃長，單位mm"
    hint_是否為T刀 = "[選填]\n該刀具是否為T形刀，如果是則填1，如果不是則填0"
    hint_是否有R角 = "[選填]\n該刀具是否有R角，如果是則填1，如果不是則填0"
    hint_刃數 = "[選填]\n該刀具有幾刃，例如：3"

    # 獲取所有不為None的刀號
    tools = [t for t in tool_list if t is not None]
    formatted_tools = sorted([f"T{x[1:]}" for x in tools])
    tool_specs = [tool_mapper.get(t) for t in formatted_tools]

    # 創建DataFrame
    tool_df = pd.DataFrame(
        {
            "刀號": [hint_刀號, "刀號"] + formatted_tools,
            "規格型號": [hint_刀具規格, "規格型號"] + tool_specs,
            "刀頭直徑": [hint_刀頭直徑, "刀頭直徑"] + [""] * len(formatted_tools),
            "刀頭高度": [hint_刀頭高度, "刀頭高度"] + [""] * len(formatted_tools),
            "是否為T刀": [hint_是否為T刀, "是否為T刀"] + [""] * len(formatted_tools),
            "是否有R角": [hint_是否有R角, "是否有R角"] + [""] * len(formatted_tools),
            "刃數": [hint_刃數, "刃數"] + [""] * len(formatted_tools),
        }
    )

    return tool_df


def check_is_supporting(sub_program_path: str) -> bool:
    """
    檢查是否為輔助子程序

    如果子程式內容有以下模式之一，則視為切割子程序（返回False）：
    1. 包含換刀代碼，如T7M06
    2. 包含G43H代碼
    3. 包含S12000M303或類似的代碼

    Args:
        gcode: 子程序代碼內容

    Returns:
        bool: True表示是輔助子程序，False表示是切割子程序
    """
    # 定義需要匹配的模式
    patterns = [
        r"T\d+M0?6",  # 匹配換刀代碼，如T7M06, T12M6等
        r"G43H\d+",  # 匹配G43H代碼
        r"S\d+M[03]0?[34]?",  # 匹配主軸啟動代碼，如S12000M303, S12000M04等
    ]

    try:
        content = open(sub_program_path, "r").read()
    except:
        return "子程式文件未找到"  # 文件不存在，視為輔助子程序

    # 檢查所有模式是否均出現
    all_patterns_found = True
    for pattern in patterns:
        if not re.search(pattern, content):
            all_patterns_found = False
            break

    if all_patterns_found:
        return "切割子程式"  # 所有模式都找到匹配，是切割子程序

    return "輔助子程式"  # 未能匹配所有模式，是輔助子程序


def parse_sub_program_master(df, template_df) -> pd.DataFrame:
    """
    解析程式單並填入template_df

    Args:
        df: 輸入的DataFrame，包含程式單資料
        template_df: 模板DataFrame，將被填充

    Returns:
        填充後的template_df
    """
    # 處理合併單元格: 先將df轉換為字符串類型並替換NaN為空字符串
    # df = df.astype(str).replace('nan', '')

    # # 對每一列進行處理，填充空值
    # for col in df.columns:
    #     # 向下填充相同值
    #     df[col] = df[col].replace('', None).ffill()

    # 創建一個template_df的副本，避免修改原始資料
    result_df = template_df.copy()

    # 檢查df是否為空
    if df.empty:
        return result_df

    # 1. 找到有'加工内容'的一行，以這一行作為表頭
    header_row_idx = None
    for idx, row in df.iterrows():
        if "加工内容" in row.values:
            header_row_idx = idx
            break

    if header_row_idx is None:
        return result_df  # 如果找不到表頭行，返回原始模板

    # 將這一行設為表頭，將這一行以下的部分作為內容
    content_df = df.iloc[header_row_idx:].copy()
    headers = content_df.iloc[0].tolist()

    # 清理表頭中的空格和換行符
    headers = [
        h.strip().replace(" ", "").replace("\n", "") if isinstance(h, str) else h
        for h in headers
    ]

    content_df.columns = headers
    content_df = content_df.iloc[1:]  # 排除表頭行

    # 將所有列轉換為字符串類型，處理NaN值
    content_df = content_df.astype(str)
    content_df = content_df.replace("nan", "")

    # 2. 反向映射：將template_df的所有列映射到df
    mapping = {
        "新程式": "sub_program",
        "加工内容": "function",
        "刀号": "tool",
        "刀补": "tool_offset",
        "刀具规格型号": "tool_spec",
        "加工模式": "mode",
        "断刀检测": "break_tool_check",
        "夹屑检测": "chip_detection",
        "转速(S)": "speed",
        "进给(F)": "feed_rate",
        "坐标系": "coordinate_system",
        "刀具伸出长度范围": "tool_length",
        "刀把型号": "tool_holder",
        "CT宏變量": "ct_macro",
        "CT": "real_ct",
        "备注": "remark",
        "4轴角度": "rotation_4th_axis",
        "0.5轴角度": "rotation_0.5_axis",
    }

    # 同樣清理映射字典中的鍵，去除空格和換行符
    cleaned_mapping = {
        k.strip().replace(" ", "").replace("\n", ""): v for k, v in mapping.items()
    }
    mapping = cleaned_mapping

    # 只保留模板的前兩行（說明和列名）
    header_rows = result_df.iloc[:2].copy()
    template_columns = result_df.columns.tolist()

    # 創建新資料列的字典
    new_data = []

    # 找出df中所有列
    df_columns = content_df.columns.tolist()

    # 檢查CT列的處理情況
    has_ct_column = "CT" in df_columns

    # 如果有CT列並且下方有兩個子列
    if has_ct_column and content_df.shape[1] > df_columns.index("CT") + 1:
        # 獲取CT列的索引
        ct_index = df_columns.index("CT")
        # 如果CT列下有兩個子列，則將CT列右邊的列視為real_ct
        ct_columns = ["ct_macro", "real_ct"]
    else:
        ct_columns = []

    # 為每一行創建新的行數據
    for _, row in content_df.iterrows():
        new_row = {}

        # 先處理template_df中已有的列
        for template_col in template_columns:
            new_row[template_col] = ""  # 初始化為空

        # 對df中的每一列找對應的template列或創建新列
        for i, df_col in enumerate(df_columns):
            # 特殊處理CT列
            if df_col == "CT" and len(ct_columns) == 2:
                # 獲取CT列的值（左邊子列，宏變量）
                new_row["ct_macro"] = row[df_col]

                # 嘗試獲取右邊子列的值（實際CT時間）
                if i + 1 < len(df_columns):
                    next_col = df_columns[i + 1]
                    new_row["real_ct"] = row[next_col]
                    # 跳過下一列的處理，因為已經處理過了
                    continue
            # 如果是CT右邊的列且已經在CT處理中處理過了，則跳過
            elif i > 0 and df_columns[i - 1] == "CT" and len(ct_columns) == 2:
                continue
            # 一般映射處理
            elif df_col in mapping and mapping[df_col] in template_columns:
                template_col = mapping[df_col]
                new_row[template_col] = row[df_col]
            # 如果沒有對應關係，直接使用原列名
            else:
                # 檢查是否已經在template_columns中
                if df_col not in template_columns:
                    # 如果不在，加入到template_columns
                    # template_columns.append(df_col)
                    pass

                # 無論如何，加入到新行數據中
                new_row[df_col] = row[df_col]

        # 特殊處理sub_program_key
        if "sub_program" in new_row and new_row["sub_program"]:
            # 假設sub_program_key的格式為"序號-程式號"
            prog_num = new_row["sub_program"].zfill(4)
            idx = len(new_data) + 1
            new_row["sub_program_key"] = f"{str(idx).zfill(2)}-{prog_num}"

        new_data.append(new_row)

    # 創建新的DataFrame，包含所有列
    content_result_df = pd.DataFrame(new_data, columns=template_columns)
    content_result_df["sub_program"] = content_result_df["sub_program"].str.replace(
        "O", ""
    )
    content_result_df["sub_program_key"] = content_result_df[
        "sub_program_key"
    ].str.replace("O", "")
    content_result_df["break_tool_check"] = content_result_df[
        "break_tool_check"
    ].str.replace("/", "")
    content_result_df["chip_detection"] = content_result_df[
        "chip_detection"
    ].str.replace("/", "")

    content_result_df["rotation_4th_axis"] = (
        content_result_df["rotation_4th_axis"].astype(str).replace("", None).ffill()
    )
    content_result_df["rotation_0.5_axis"] = (
        content_result_df["rotation_0.5_axis"].astype(str).replace("", None).ffill()
    )
    content_result_df["is_supporting"] = content_result_df["sub_program"].apply(
        check_is_supporting
    )

    # # 為新增的列加入說明行
    # for col in template_columns:
    #     if col not in header_rows.columns:
    #         header_rows[col] = [f"[選填]來自程式單的{col}", col]

    # 合併表頭和內容
    result_df = pd.concat([header_rows, content_result_df], ignore_index=True)

    return result_df
