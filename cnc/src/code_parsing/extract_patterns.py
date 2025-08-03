import re


def extract_subline(gcode_line, command):
    """
    提取command指令的參數字符串

    - 不包含command本身
    - 紧跟在command后面的字符串
    - 直到遇到下一個G指令或行尾
    """
    # 找到command的位置
    command_pos = gcode_line.find(command)
    if command_pos == -1:
        return None

    # 找到command指令結束的位置（下一個G指令的開始，或者行尾）
    command_part = gcode_line[command_pos:]
    next_g_pos = -1
    for i in range(len(command), len(command_part)):  # 從command之後開始查找
        if (
            command_part[i] == "G"
            and i + 1 < len(command_part)
            and command_part[i + 1].isdigit()
        ):
            next_g_pos = i
            break

    if next_g_pos != -1:
        command_params = command_part[:next_g_pos]
    else:
        command_params = command_part
    return command_params


# 提取 G100 pattern
def extract_G100_pattern(gcode_line):
    """
    提取 G100 指令的參數。

    支持的格式：
    - G100V#1000 / G100U#1000：設置進給速度和主軸轉速
    - G100V#1000 / G100U#1000：設置進給速度和主軸轉速
    """
    gcode_line = gcode_line.strip().replace("G100V", "F")
    gcode_line = gcode_line.strip().replace("G100U", "S")
    return gcode_line


# 提取 M98 pattern
def extract_M98_pattern(gcode_line):
    """
    提取 M98 指令的參數。

    支持的格式：
    - M98P1000：調用子程序
    """
    pattern = re.compile(r"M98" r"P(?P<p>(?:[0-9]*)?)")
    match = pattern.search(gcode_line)
    if match:
        return match.group("p")
    else:
        return None


def extract_G04_pattern(gcode_line):
    """
    提取 G04 指令的參數。

    支持的格式：
    - G04P1000：暫停 1000 毫秒
    - G04X1.0：暫停 1.0 秒

    注意: G04 只是一個暫停（dwell）指令，後面的 P 或 X 參數表示暫停時間，
    並非代表軸的移動
    """
    pattern = re.compile(
        r"G04"
        r"(?P<mode>[PX])?"  # P改为可选参数
        r"(?P<time>(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?))"  # X改为可选参数
    )
    match = pattern.search(gcode_line)
    if match:
        mode = match.group("mode") if match.group("mode") else None
        time = match.group("time") if match.group("time") else 0

        # 計算需要跳過的指令數 - 只計算實際匹配到的參數
        skip_count = 0  # 不跳過G04本身，因為它已經被處理了

        # 檢查各個參數是否在匹配中存在
        if mode:  # P/X參數
            skip_count += 1
            t = float(time)
            if mode == "P":
                t = t * 0.001  # 將毫秒轉換為秒
        else:
            t = None
        return t, skip_count
    return None, 0  # 沒有匹配到，返回None和0


# 提取 G05 FANUC 機台高精加工模式
def extract_G05_pattern(gcode_line):
    """
    解析G05和G5.1的高精度加工模式指令。

    支持的格式：
    - G05P20000 / G05P0：高精度模式開啟/關閉
    - G05.1Q1 / G05.1Q0：高精度模式開啟/關閉
    - G5.1Q1 / G5.1Q0：高精度模式開啟/關閉（簡寫形式）

    返回：
    - (bool, int): (是否開啟高精度模式, 需要跳過的指令數)
    """
    # 先將G5.1標準化為G05.1
    gcode_line = gcode_line.replace("G5.1", "G05.1")

    # 標準G05格式
    if "G05P20000" in gcode_line:
        return True, 1
    elif "G05P0" in gcode_line:
        return False, 1

    # G05.1格式
    if "G05.1Q1" in gcode_line:
        return True, 1
    elif "G05.1Q0" in gcode_line:
        return False, 1

    # 未知格式
    print(f"未知的G05/G5.1指令格式: {gcode_line}")
    return None, 0


# 提取 G10 pattern
def extract_G10_pattern(gcode_line):
    """
    提取 G10 指令的參數。

    支持的格式：
    - G10L1：移動到第一參考點
    - G10L2：移動到第二參考點
    - G10L3：移動到第三參考點
    """
    pattern = re.compile(r"G10" r"L(?P<l>(?:[0-9]*)?)")
    match = pattern.search(gcode_line)
    if match:
        return match.group("l"), 1
    else:
        print(f"New G10 pattern {gcode_line}")
        return None, 1


def extract_G30_pattern(gcode_line):
    """
    提取 G30 指令的參數，主要用于刀具更換位置、安全停車位置、程序結束位置

    支持的格式：
    - G30：移動到第二參考點
    - G30X100Y100Z100：移動到指定的中間位置，然後移動到第二參考點

    G30指令功能說明：
    1. 將機床快速移動到預設的第二參考點（secondary reference position）
    2. 主要用途：刀具更換位置、安全停車位置、程序結束位置
    3. 語法格式：G30 [X__] [Y__] [Z__]
       - 無參數：直接移動到第二參考點
       - 有參數：先移動到指定的中間位置，然後移動到第二參考點
    # 4. 與G28的區別：G28回到機床原點，G30回到用戶定義的第二參考點
    """

    # 通用參數提取模式：支持宏變量（#開頭）和普通數值（可帶正負號）
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 使用更精確的正則表達式，只匹配G30指令及其直接相關的參數
    g30_pattern = re.compile(
        r"G30"  # 匹配G30
        r"(?:X(" + param_pattern + r"))?"  # 可選X參數
        r"(?:Y(" + param_pattern + r"))?"  # 可選Y參數
        r"(?:Z(" + param_pattern + r"))?"  # 可選Z參數
    )

    match = g30_pattern.search(gcode_line)
    if match:
        # 獲取參數值
        x = match.group(1) if match.group(1) else None
        y = match.group(2) if match.group(2) else None
        z = match.group(3) if match.group(3) else None

        # 計算需要跳過的指令數 - 只計算實際匹配到的參數
        skip_count = 0  # 不跳過G30本身，因為它已經被處理了

        # 檢查各個參數是否在匹配中存在
        if match.group(1):  # X參數
            skip_count += 1
        if match.group(2):  # Y參數
            skip_count += 1
        if match.group(3):  # Z參數
            skip_count += 1

        # 返回參數值和跳過指令數
        if x is not None or y is not None or z is not None:
            # 有參數的情況：返回所有參數值和跳過數
            return x, y, z, skip_count
        else:
            # 無參數的情況：僅返回G30指令本身
            return None, None, None, 0
    else:
        # 沒有匹配到，可能是單獨的G30指令
        return None, None, None, 0


# 提取 G68 旋转坐标 pattern
def extract_G68_pattern(gcode_line):
    """
    提取 G68 指令的參數。

    支持的格式：
    - G68X100Y100R600：旋轉坐標系統
    - G68X100Y100R600：旋轉坐標系統

    G68指令功能說明：
    1. 旋轉坐標系統
    2. 語法格式：G68X__Y__R__
    3. 旋轉角度R為必須參數，X和Y為可選參數，順序可變
    4. 旋轉角度R為以 '#' 開頭的宏變量或者是整數/小數數字（可帶正負號）。
    5. 旋轉角度R為以 '#' 開頭的宏變量或者是整數/小數數字（可帶正負號）。

    XYR为可选参数，顺序可变，x, y, r 可為以 '#' 開頭的宏變量或者是整數/小數數字（可帶正負號）。
    """
    # 檢查是否包含G68指令
    if "G68" not in gcode_line:
        return None, None, None, None, 0

    # 找到G68指令結束的位置（下一個G指令的開始，或者行尾）
    g68_params = extract_subline(gcode_line, "G68")
    if g68_params is None:
        return None, None, None, None, 0

    # 通用參數提取模式：支持宏變量（#開頭）和普通數值（可帶正負號）
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 只在G68參數部分中提取X、Y、R參數
    x_match = re.search(r"X(" + param_pattern + r")", g68_params)
    y_match = re.search(r"Y(" + param_pattern + r")", g68_params)
    r_match = re.search(r"R(" + param_pattern + r")", g68_params)

    # 獲取參數值
    x = x_match.group(1) if x_match else None
    y = y_match.group(1) if y_match else None
    r = r_match.group(1) if r_match else None

    # 計算需要跳過的指令數
    skip_count = 0  # 基礎計數：G68本身

    # 檢查各個參數是否存在
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if r_match:  # R參數
        skip_count += 1

    return x, y, r, skip_count


def extract_G68p2_pattern(gcode_line):
    """
    提取 G68.2 指令的參數。

    支持的格式：
    - G68.2P1X100Y100Z100I100J100K100：旋轉坐標系統
    - G68.2P1X100Y100Z100I100J100K100：旋轉坐標系統

    G68.2指令功能說明：
    1. 旋轉坐標系統
    2. 語法格式：G68.2P1X__Y__Z__I__J__K__
    3. P參數用於指定旋轉坐標系統的編號（如P1-P48等）
    4. X、Y、Z、I、J、K為可選參數，順序可變
    5. X、Y、Z、I、J、K為以 '#' 開頭的宏變量或者是整數/小數數字（可帶正負號）。

    """
    # 檢查是否包含G68.2指令
    if "G68.2" not in gcode_line:
        return None, None, None, None, None, None, None, 0

    # 提取G68.2參數
    g68_params = extract_subline(gcode_line, "G68.2")
    if g68_params is None:
        return None, None, None, None, None, None, None, 0

    # 通用參數提取模式：支持宏變量（#開頭）和普通數值（可帶正負號）
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 只在G68.2參數部分中提取P、X、Y、Z、I、J、K參數
    p_match = re.search(r"P(" + param_pattern + r")", g68_params)
    x_match = re.search(r"X(" + param_pattern + r")", g68_params)
    y_match = re.search(r"Y(" + param_pattern + r")", g68_params)
    z_match = re.search(r"Z(" + param_pattern + r")", g68_params)
    i_match = re.search(r"I(" + param_pattern + r")", g68_params)
    j_match = re.search(r"J(" + param_pattern + r")", g68_params)
    k_match = re.search(r"K(" + param_pattern + r")", g68_params)

    # 獲取參數值
    p = p_match.group(1) if p_match else None
    x = x_match.group(1) if x_match else None
    y = y_match.group(1) if y_match else None
    z = z_match.group(1) if z_match else None
    i = i_match.group(1) if i_match else None
    j = j_match.group(1) if j_match else None
    k = k_match.group(1) if k_match else None

    # 計算需要跳過的指令數
    skip_count = 0  # 基礎計數：G68.2本身

    # 檢查各個參數是否存在
    if p_match:  # P參數
        skip_count += 1
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if z_match:  # Z參數
        skip_count += 1
    if i_match:  # I參數
        skip_count += 1
    if j_match:  # J參數
        skip_count += 1
    if k_match:  # K參數
        skip_count += 1

    return p, x, y, z, i, j, k, skip_count


def extract_G53p1_pattern(gcode_line):
    # G53.1P1 - 機床坐標系統的擴展指令
    # ================================
    # G53.1P1指令功能說明：
    # 1. 機床坐標系統的擴展版本，類似於G53但具有額外參數
    # 2. P參數用於指定特定的機床參考系統或配置
    # 3. 語法格式：G53.1P1 [X__] [Y__] [Z__]
    # 4. 與G53類似，會忽略所有活動的工件坐標偏移
    # 5. 主要用於複雜機床配置或特殊定位需求
    # ================================

    # 檢查是否包含G53.1指令
    if "G53.1" not in gcode_line:
        return None

    # 通用參數提取模式：支持宏變量（#開頭）和普通數值（可帶正負號）
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 提取P參數
    p_match = re.search(r"P(" + param_pattern + r")", gcode_line)

    # 提取X、Y、Z參數，不限制順序
    x_match = re.search(r"X(" + param_pattern + r")", gcode_line)
    y_match = re.search(r"Y(" + param_pattern + r")", gcode_line)
    z_match = re.search(r"Z(" + param_pattern + r")", gcode_line)

    # 獲取參數值
    p = p_match.group(1) if p_match else None
    x = x_match.group(1) if x_match else None
    y = y_match.group(1) if y_match else None
    z = z_match.group(1) if z_match else None

    # 計算需要跳過的指令數
    skip_count = 0  # 基礎計數：G53.1本身

    # 檢查各個參數是否存在
    if p_match:  # P參數
        skip_count += 1
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if z_match:  # Z參數
        skip_count += 1

    # 如果找到任何參數，返回結果
    if p is not None or x is not None or y is not None or z is not None:
        return p, x, y, z, skip_count
    else:
        # 僅有G53.1指令，沒有參數
        return None, None, None, None, 1


# 提取 G54.1P40進行坐標變換
def extract_G54p1_pattern(gcode_line):
    # G54.1P40 - 擴展工件坐標系統指令
    # ================================
    # G54.1P40指令功能說明：
    # 1. 擴展工件坐標系統，支援更多坐標系統配置
    # 2. P參數用於指定坐標系統編號（如P1-P48等）
    # 3. 語法格式：G54.1P40 [X__] [Y__] [Z__]
    # 4. 可以用於設置額外的工件座標系統偏移
    # 5. 支援宏變量和普通數值參數
    # ================================

    # 檢查是否包含G54.1指令
    if "G54.1" not in gcode_line:
        return None

    # 通用參數提取模式：支持宏變量（#開頭）和普通數值（可帶正負號）
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 提取P參數（必須）
    p_match = re.search(r"P(" + param_pattern + r")", gcode_line)

    # 提取X、Y、Z參數，不限制順序
    x_match = re.search(r"X(" + param_pattern + r")", gcode_line)
    y_match = re.search(r"Y(" + param_pattern + r")", gcode_line)
    z_match = re.search(r"Z(" + param_pattern + r")", gcode_line)

    # 獲取參數值
    p = p_match.group(1) if p_match else None
    x = x_match.group(1) if x_match else None
    y = y_match.group(1) if y_match else None
    z = z_match.group(1) if z_match else None

    # 計算需要跳過的指令數
    skip_count = 0  # 基礎計數：G54.1本身

    # 檢查P參數是否存在（必須參數）
    if p_match:
        skip_count += 1

    # 檢查各個可選參數是否存在
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if z_match:  # Z參數
        skip_count += 1

    # 如果找到P參數，返回結果
    if p is not None:
        return p, x, y, z, skip_count
    else:
        # 沒有找到P參數，返回None
        return None, None, None, None, 0


# 提取钻孔 G81 Pattern
def extract_G81_pattern(gcode_line, status):
    # 支持以下格式:
    # 1. G81G99X[#14]Y[#15]Z[0.43+#930]R2.F150
    # 2. G98G81X-49.62Y7.R2.Z0.46F800.
    # 3. G81G99Z103.39R105.5F1000. (省略XY)
    # 4. G81G99X10Z103.39R105.5F1000. (省略Y)
    # 5. G81G99Y20Z103.39R105.5F1000. (省略X)
    # 6. G99G81X170.Y320.Z235.2R237.7F100. (G99在前G81在后)
    # 7. G81X-155.Y250.Z235.2R237.7

    # 檢查是否包含G81鑽孔循環
    if "G81" not in gcode_line:
        return None, 0

    # 檢測返回模式：G98（返回初始平面）或 G99（返回R平面）
    return_mode = None
    if "G98" in gcode_line:
        return_mode = "G98"  # 返回初始平面
    elif "G99" in gcode_line:
        return_mode = "G99"  # 返回R平面
    else:
        # 如果沒有明確指定，根據CNC標準，默認為G99（返回R平面）
        return_mode = "G99"

    # 通用參數提取模式：支持參數任意順序排列
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 提取各個參數，不限制順序
    x_match = re.search(r"X(" + param_pattern + r")", gcode_line)
    y_match = re.search(r"Y(" + param_pattern + r")", gcode_line)
    z_match = re.search(r"Z(" + param_pattern + r")", gcode_line)
    r_match = re.search(r"R(" + param_pattern + r")", gcode_line)
    f_match = re.search(r"F(" + param_pattern + r")", gcode_line)

    # 獲取參數值，如果不存在則使用狀態中的值
    x = x_match.group(1) if x_match else status.X
    y = y_match.group(1) if y_match else status.Y
    origin_z = z_match.group(1) if z_match else status.Z

    # R參數處理：如果沒有R參數，嘗試使用狀態中的模態R值
    if r_match:
        r = r_match.group(1)
    else:
        # 沒有R參數時，使用狀態中的G99_R（模態R值）
        if hasattr(status, "G99_R") and status.G99_R is not None:
            r = status.G99_R
            print(f"G81指令使用模態R值: {r} (來源: {gcode_line})")
        else:
            # 如果也沒有模態R值，使用當前Z位置作為安全高度
            r = status.Z
            print(f"G81指令使用當前Z位置作為默認R值: {r} (來源: {gcode_line})")

    # F參數處理：如果沒有F參數，使用狀態中的模態F值
    if f_match:
        f = f_match.group(1)
    else:
        # 使用狀態中的F值（模態進給速度）
        f = getattr(status, "F", None)

    # 根據返回模式計算執行後的Z位置
    if return_mode == "G98":
        # G98：返回到初始平面，Z位置應該是加工前的初始Z位置
        final_z = status.Z  # 保持當前狀態的Z位置（初始位置）
    else:  # G99
        # G99：返回到R平面，Z位置應該是R參數的值
        final_z = r

    # 動態計算skip_count：根據實際匹配到的參數個數
    skip_count = 0  # 基礎計數：G81本身

    # 檢查返回模式（G98/G99）
    if "G98" in gcode_line or "G99" in gcode_line:
        skip_count += 1

    # 檢查各個參數是否存在
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if z_match:  # Z參數
        skip_count += 1
    if r_match:  # R參數
        skip_count += 1
    if f_match:  # F參數
        skip_count += 1

    return x, y, origin_z, final_z, r, f, skip_count


def extract_G83_pattern(gcode_line, status):
    """
    提取G83深孔鑽削循環的參數。

    支持的格式：
    - G83G99XxYyZzRrQqFf
    - G98G83XxYyZzRrQqFf
    - G99G83XxYyZzRrQqFf (G99在前G83在後)
    - G83XxYyZzRrQqFf (沒有返回模式)

    其中：
    - X,Y: 孔的位置 (可選)
    - Z: 孔的最終深度 (可選)
    - R: 快速進給平面（R平面）(可選，可使用模態值)
    - Q: 每次切入深度 (可選，可使用模態值)
    - F: 進給速度 (可選，可使用模態值)

    例如：
    G83G99X-16.683Y106.137Z91.847R98.217Q6.32F800

    返回：
    - (str, str, str, str, str, str, int):
      (X值, Y值, 執行後Z值, R值, Q值, F值, 需要跳過的指令數)
    """
    # 檢查是否包含G83深孔鑽削循環
    if "G83" not in gcode_line:
        return None, 0

    # 檢測返回模式：G98（返回初始平面）或 G99（返回R平面）
    return_mode = None
    if "G98" in gcode_line:
        return_mode = "G98"  # 返回初始平面
    elif "G99" in gcode_line:
        return_mode = "G99"  # 返回R平面
    else:
        # 如果沒有明確指定，根據CNC標準，默認為G99（返回R平面）
        return_mode = "G99"

    # 通用參數提取模式：支持參數任意順序排列
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 提取各個參數，不限制順序
    x_match = re.search(r"X(" + param_pattern + r")", gcode_line)
    y_match = re.search(r"Y(" + param_pattern + r")", gcode_line)
    z_match = re.search(r"Z(" + param_pattern + r")", gcode_line)
    r_match = re.search(r"R(" + param_pattern + r")", gcode_line)
    q_match = re.search(r"Q(" + param_pattern + r")", gcode_line)
    f_match = re.search(r"F(" + param_pattern + r")", gcode_line)

    # 獲取參數值，如果不存在則使用狀態中的值
    x = x_match.group(1) if x_match else status.X
    y = y_match.group(1) if y_match else status.Y
    origin_z = z_match.group(1) if z_match else status.Z

    # R參數處理：如果沒有R參數，嘗試使用狀態中的模態R值
    if r_match:
        r = r_match.group(1)
    else:
        # 沒有R參數時，使用狀態中的G99_R（模態R值）
        if hasattr(status, "G99_R") and status.G99_R is not None:
            r = status.G99_R
            print(f"G83指令使用模態R值: {r} (來源: {gcode_line})")
        else:
            # 如果也沒有模態R值，使用當前Z位置作為安全高度
            r = status.Z
            print(f"G83指令使用當前Z位置作為默認R值: {r} (來源: {gcode_line})")

    # Q參數處理：如果沒有Q參數，使用狀態中的模態Q值
    if q_match:
        q = q_match.group(1)
    else:
        # 使用狀態中的G83_Q值（模態切入深度）
        q = getattr(status, "G83_Q", None)
        if q is None:
            print(f"G83指令缺少Q參數且無模態Q值: {gcode_line}")

    # F參數處理：如果沒有F參數，使用狀態中的模態F值
    if f_match:
        f = f_match.group(1)
    else:
        # 使用狀態中的F值（模態進給速度）
        f = getattr(status, "F", None)

    # 根據返回模式計算執行後的Z位置
    if return_mode == "G98":
        # G98：返回到初始平面，Z位置應該是加工前的初始Z位置
        final_z = status.Z  # 保持當前狀態的Z位置（初始位置）
    else:  # G99
        # G99：返回到R平面，Z位置應該是R參數的值
        final_z = r

    # 動態計算skip_count：根據實際匹配到的參數個數
    skip_count = 0  # 基礎計數：G83本身

    # 檢查返回模式（G98/G99）
    if "G98" in gcode_line or "G99" in gcode_line:
        skip_count += 1

    # 檢查各個參數是否存在
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if z_match:  # Z參數
        skip_count += 1
    if r_match:  # R參數
        skip_count += 1
    if q_match:  # Q參數
        skip_count += 1
    if f_match:  # F參數
        skip_count += 1

    return x, y, origin_z, final_z, r, q, f, skip_count


def extract_G82_pattern(gcode_line, status):
    """
    提取G82沉孔鑽削循環的參數。

    支持的格式：
    - G82G99XxYyZzRrPpFf
    - G98G82XxYyZzRrPpFf
    - G99G82XxYyZzRrPpFf (G99在前G82在後)
    - G82XxYyZzRrPpFf (沒有返回模式)

    其中：
    - X,Y: 孔的位置 (可選)
    - Z: 孔的最終深度 (可選)
    - R: 快速進給平面（R平面）(可選，可使用模態值)
    - P: 在孔底的停留時間（秒或毫秒） (可選，可使用模態值)
    - F: 進給速度 (可選，可使用模態值)

    例如：
    G82G99X10.5Y20.3Z-5.2R2.0P0.5F150
    G82X10Y20Z-5R2P500F100 (P500毫秒)

    返回：
    - (str, str, str, str, str, str, int):
      (X值, Y值, 執行後Z值, R值, P值, F值, 需要跳過的指令數)
    """
    # 檢查是否包含G82沉孔鑽削循環
    if "G82" not in gcode_line:
        return None, 0

    # 檢測返回模式：G98（返回初始平面）或 G99（返回R平面）
    return_mode = None
    if "G98" in gcode_line:
        return_mode = "G98"  # 返回初始平面
    elif "G99" in gcode_line:
        return_mode = "G99"  # 返回R平面
    else:
        # 如果沒有明確指定，根據CNC標準，默認為G99（返回R平面）
        return_mode = "G99"

    # 通用參數提取模式：支持參數任意順序排列
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 提取各個參數，不限制順序
    x_match = re.search(r"X(" + param_pattern + r")", gcode_line)
    y_match = re.search(r"Y(" + param_pattern + r")", gcode_line)
    z_match = re.search(r"Z(" + param_pattern + r")", gcode_line)
    r_match = re.search(r"R(" + param_pattern + r")", gcode_line)
    p_match = re.search(r"P(" + param_pattern + r")", gcode_line)
    f_match = re.search(r"F(" + param_pattern + r")", gcode_line)

    # 獲取參數值，如果不存在則使用狀態中的值
    x = x_match.group(1) if x_match else status.X
    y = y_match.group(1) if y_match else status.Y
    origin_z = z_match.group(1) if z_match else status.Z

    # R參數處理：如果沒有R參數，嘗試使用狀態中的模態R值
    if r_match:
        r = r_match.group(1)
    else:
        # 沒有R參數時，使用狀態中的G99_R（模態R值）
        if hasattr(status, "G99_R") and status.G99_R is not None:
            r = status.G99_R
            print(f"G82指令使用模態R值: {r} (來源: {gcode_line})")
        else:
            # 如果也沒有模態R值，使用當前Z位置作為安全高度
            r = status.Z
            print(f"G82指令使用當前Z位置作為默認R值: {r} (來源: {gcode_line})")

    # P參數處理：如果沒有P參數，使用狀態中的模態P值
    if p_match:
        p = p_match.group(1)
    else:
        # 使用狀態中的G82_P值（模態停留時間）
        p = getattr(status, "G82_P", None)
        if p is None:
            print(f"G82指令缺少P參數且無模態P值: {gcode_line}")

    # F參數處理：如果沒有F參數，使用狀態中的模態F值
    if f_match:
        f = f_match.group(1)
    else:
        # 使用狀態中的F值（模態進給速度）
        f = getattr(status, "F", None)

    # 根據返回模式計算執行後的Z位置
    if return_mode == "G98":
        # G98：返回到初始平面，Z位置應該是加工前的初始Z位置
        final_z = status.Z  # 保持當前狀態的Z位置（初始位置）
    else:  # G99
        # G99：返回到R平面，Z位置應該是R參數的值
        final_z = r

    # 動態計算skip_count：根據實際匹配到的參數個數
    skip_count = 0  # 基礎計數：G82本身

    # 檢查返回模式（G98/G99）
    if "G98" in gcode_line or "G99" in gcode_line:
        skip_count += 1

    # 檢查各個參數是否存在
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if z_match:  # Z參數
        skip_count += 1
    if r_match:  # R參數
        skip_count += 1
    if p_match:  # P參數
        skip_count += 1
    if f_match:  # F參數
        skip_count += 1

    return x, y, origin_z, final_z, r, p, f, skip_count


def extract_G84_pattern(gcode_line, status):
    """
    提取G84螺紋切削循環（攻絲循環）的參數。

    支持的格式：
    - G84G99XxYyZzRrQqFf
    - G98G84XxYyZzRrQqFf
    - G99G84XxYyZzRrQqFf (G99在前G84在後)
    - G84XxYyZzRrQqFf (沒有返回模式)

    其中：
    - X,Y: 螺紋孔的位置 (可選)
    - Z: 螺紋切削的最終深度 (可選)
    - R: 快速進給平面（R平面）(可選，可使用模態值)
    - Q: 螺紋導程（thread pitch）(可選，可使用模態值)
    - F: 進給速度 (可選，可使用模態值)

    例如：
    G99G84X337.Y26.45Z-11.R2.5F480.Q10.5

    返回：
    - (str, str, str, str, str, str, int):
      (X值, Y值, 執行後Z值, R值, Q值, F值, 需要跳過的指令數)
    """
    # 檢查是否包含G84螺紋切削循環
    if "G84" not in gcode_line:
        return None, 0

    # 檢測返回模式：G98（返回初始平面）或 G99（返回R平面）
    return_mode = None
    if "G98" in gcode_line:
        return_mode = "G98"  # 返回初始平面
    elif "G99" in gcode_line:
        return_mode = "G99"  # 返回R平面
    else:
        # 如果沒有明確指定，根據CNC標準，默認為G99（返回R平面）
        return_mode = "G99"

    # 通用參數提取模式：支持參數任意順序排列
    param_pattern = r"(?:#[0-9]+|[+-]?[0-9]*(?:\.[0-9]*)?)"

    # 提取各個參數，不限制順序
    x_match = re.search(r"X(" + param_pattern + r")", gcode_line)
    y_match = re.search(r"Y(" + param_pattern + r")", gcode_line)
    z_match = re.search(r"Z(" + param_pattern + r")", gcode_line)
    r_match = re.search(r"R(" + param_pattern + r")", gcode_line)
    q_match = re.search(r"Q(" + param_pattern + r")", gcode_line)
    f_match = re.search(r"F(" + param_pattern + r")", gcode_line)

    # 獲取參數值，如果不存在則使用狀態中的值
    x = x_match.group(1) if x_match else status.X
    y = y_match.group(1) if y_match else status.Y
    origin_z = z_match.group(1) if z_match else status.Z

    # R參數處理：如果沒有R參數，嘗試使用狀態中的模態R值
    if r_match:
        r = r_match.group(1)
    else:
        # 沒有R參數時，使用狀態中的G99_R（模態R值）
        if hasattr(status, "G99_R") and status.G99_R is not None:
            r = status.G99_R
            print(f"G84指令使用模態R值: {r} (來源: {gcode_line})")
        else:
            # 如果也沒有模態R值，使用當前Z位置作為安全高度
            r = status.Z
            print(f"G84指令使用當前Z位置作為默認R值: {r} (來源: {gcode_line})")

    # Q參數處理：如果沒有Q參數，使用狀態中的模態Q值
    if q_match:
        q = q_match.group(1)
    else:
        # 使用狀態中的G84_Q值（模態螺紋導程）
        q = getattr(status, "G84_Q", None)
        if q is None:
            print(f"G84指令缺少Q參數且無模態Q值: {gcode_line}")

    # F參數處理：如果沒有F參數，使用狀態中的模態F值
    if f_match:
        f = f_match.group(1)
    else:
        # 使用狀態中的F值（模態進給速度）
        f = getattr(status, "F", None)

    # 根據返回模式計算執行後的Z位置
    if return_mode == "G98":
        # G98：返回到初始平面，Z位置應該是加工前的初始Z位置
        final_z = status.Z  # 保持當前狀態的Z位置（初始位置）
    else:  # G99
        # G99：返回到R平面，Z位置應該是R參數的值
        final_z = r

    # 動態計算skip_count：根據實際匹配到的參數個數
    skip_count = 0  # 基礎計數：G84本身

    # 檢查返回模式（G98/G99）
    if "G98" in gcode_line or "G99" in gcode_line:
        skip_count += 1

    # 檢查各個參數是否存在
    if x_match:  # X參數
        skip_count += 1
    if y_match:  # Y參數
        skip_count += 1
    if z_match:  # Z參數
        skip_count += 1
    if r_match:  # R參數
        skip_count += 1
    if q_match:  # Q參數
        skip_count += 1
    if f_match:  # F參數
        skip_count += 1

    return x, y, origin_z, final_z, r, q, f, skip_count
