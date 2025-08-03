import yaml
import json
import pandas as pd
import os


def load_yaml_config(config_path: str) -> dict:
    """
    Load a YAML configuration file into a dictionary.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def merge_configs(base_config: dict, specialized_config: dict) -> dict:
    """
    Recursively merge two configuration dictionaries, where specialized config overwrites base config.
    Preserves nested structures in base_config that aren't overwritten by specialized_config.

    Args:
        base_config (dict): Base configuration dictionary
        specialized_config (dict): Specialized configuration dictionary that will override base

    Returns:
        dict: Merged configuration dictionary
    """
    merged = base_config.copy()
    for key, value in specialized_config.items():
        if (
            isinstance(value, dict)
            and key in base_config
            and isinstance(base_config[key], dict)
        ):
            merged[key] = merge_configs(base_config[key], value)
        else:
            merged[key] = value
    return merged


def load_config(base_config_path: str) -> dict:
    """
    Load and merge two YAML configuration files, where specialized config overwrites base config.

    Args:
        base_config_path (str): Path to the base configuration YAML file

    Returns:
        dict: Merged configuration dictionary
    """
    base_config = load_yaml_config(base_config_path)
    specialized_config = load_yaml_config(
        f"cnc_genai/conf/v1_custom/{base_config['scenario_name']}.yaml"
    )

    return merge_configs(base_config, specialized_config)


def process_v1_sp_config(
    df_sub_prog, ban_n_dict, ban_row_dict, hyper_params_dict
) -> dict:

    # 確保ban_n_dict和ban_row_dict的鍵是字符串類型
    ban_n_dict = {str(k): v for k, v in ban_n_dict.items()}
    ban_row_dict = {str(k): v for k, v in ban_row_dict.items()}

    # 生成dict
    specialized_config = {}
    df_sub_prog["sub_program"] = df_sub_prog["sub_program"].astype(str).str.zfill(4)
    specialized_config["sub_programs"] = df_sub_prog.set_index("sub_program").to_dict(
        orient="index"
    )

    for sub_program in specialized_config["sub_programs"].keys():
        if sub_program in ban_n_dict.keys():
            specialized_config["sub_programs"][sub_program]["ban_n"] = ban_n_dict[
                sub_program
            ]
        else:
            specialized_config["sub_programs"][sub_program]["ban_n"] = []

        if sub_program in ban_row_dict.keys():
            specialized_config["sub_programs"][sub_program]["ban_row"] = ban_row_dict[
                sub_program
            ]
        else:
            specialized_config["sub_programs"][sub_program]["ban_row"] = []

    specialized_config["hyper_params"] = hyper_params_dict

    return specialized_config


def load_config_v1(base_config_path: str) -> dict:
    """
    Load and merge two YAML configuration files, where specialized config overwrites base config.

    Args:
        base_config_path (str): Path to the base configuration YAML file

    Returns:
        dict: Merged configuration dictionary
    """
    base_config = load_yaml_config(base_config_path)
    default_config_excel_path = f"cnc_genai/conf/v1_custom/demo.xlsx"
    config_excel_path = f"cnc_genai/conf/v1_custom/{base_config['clamping_name']}/{base_config['scenario_name']}.xlsx"
    if not os.path.exists(config_excel_path):
        config_excel_path = default_config_excel_path
    specialized_config = process_v1_sp_config(
        df_sub_prog=pd.read_excel(
            config_excel_path,
            sheet_name="sub_program",
            skiprows=1,
        ),
        ban_n_dict=(
            pd.read_excel(
                config_excel_path,
                sheet_name="ban_n",
                skiprows=1,
            )
            .assign(sub_program=lambda x: x["sub_program"].astype(str))
            .groupby("sub_program")["ban_n"]
            .apply(list)
            .to_dict()
        ),
        ban_row_dict=(
            pd.read_excel(
                config_excel_path,
                sheet_name="ban_row",
                skiprows=1,
            )
            .assign(sub_program=lambda x: x["sub_program"].astype(str))
            .groupby("sub_program")["row_id"]
            .apply(list)
            .to_dict()
        ),
        hyper_params_dict=pd.read_excel(
            config_excel_path,
            sheet_name="hyper_params",
            skiprows=1,
        ).to_dict(orient="records")[0],
    )

    return merge_configs(base_config, specialized_config)


def line_intersects_bbox(line, bbox):
    # 判斷直線是否經過bbox的边框函數
    x1, y1, x2, y2 = line
    bx1, by1, bx2, by2 = bbox

    def point_in_bbox(x, y, bbox):
        # 這個函數檢查點是否在bbox內
        bx1, by1, bx2, by2 = bbox
        return bx1 <= x <= bx2 and by1 <= y <= by2

    def on_segment(px, py, qx, qy, rx, ry):
        return min(px, qx) <= rx <= max(px, qx) and min(py, qy) <= ry <= max(py, qy)

    def orientation(px, py, qx, qy, rx, ry):
        val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # clockwise or counterclockwise

    def segments_intersect(p1, q1, p2, q2):
        o1 = orientation(*p1, *q1, *p2)
        o2 = orientation(*p1, *q1, *q2)
        o3 = orientation(*p2, *q2, *p1)
        o4 = orientation(*p2, *q2, *q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(*p1, *q1, *p2):
            return True
        if o2 == 0 and on_segment(*p1, *q1, *q2):
            return True
        if o3 == 0 and on_segment(*p2, *q2, *p1):
            return True
        if o4 == 0 and on_segment(*p2, *q2, *q1):
            return True

        return False

    if point_in_bbox(x1, y1, bbox) and point_in_bbox(x2, y2, bbox):
        return True

    rect_lines = [
        ((bx1, by1), (bx2, by1)),
        ((bx2, by1), (bx2, by2)),
        ((bx2, by2), (bx1, by2)),
        ((bx1, by2), (bx1, by1)),
    ]

    for rect_line in rect_lines:
        if segments_intersect((x1, y1), (x2, y2), *rect_line):
            return True

    return False


# 檢查每條線是否擊中bboxes
def check_intersections(row, bboxes, coord_label="Z"):
    if row["move_code"] in ["G01", "G02", "G03"]:
        if coord_label == "Z":
            line = (
                row["X_prev_pixel"],
                row["Y_prev_pixel"],
                row["X_pixel"],
                row["Y_pixel"],
            )

            bboxes = [
                (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]) for bbox in bboxes
            ]
        elif coord_label == "Y":
            line = (
                row["X_prev_pixel"],
                row["Z_prev_pixel"],
                row["X_pixel"],
                row["Z_pixel"],
            )
            bboxes = [
                (bbox["x1"], bbox["z1"], bbox["x2"], bbox["z2"]) for bbox in bboxes
            ]
        elif coord_label == "X":
            line = (
                row["Y_prev_pixel"],
                row["Z_prev_pixel"],
                row["Y_pixel"],
                row["Z_pixel"],
            )
            bboxes = [
                (bbox["y1"], bbox["z1"], bbox["y2"], bbox["z2"]) for bbox in bboxes
            ]
        else:
            raise ValueError(f"Invalid coord_label: {coord_label}")
        hit_bboxes = [
            i for i, bbox in enumerate(bboxes) if line_intersects_bbox(line, bbox)
        ]
        return hit_bboxes if hit_bboxes else []
    else:
        return []


def read_gcode_from_json(json_path):
    """
    從 json_path 的 JSON 數據中讀取 G 代碼並將其轉換為文本。

    Args:
        json_path (str): JSON 文件的路徑

    Returns:
        dict: 程序編號到 G 代碼文本的映射
    """
    # 讀取 JSON 文件
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # 如果輸入是字符串，嘗試解析為 JSON
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            raise ValueError("無法解析 JSON 字符串")

    # 確保 json_data 是字典
    if not isinstance(json_data, dict):
        raise ValueError("JSON 數據必須是字典格式")

    # 創建程序編號到 G 代碼文本的映射
    gcode_map = {}

    # 遍歷 JSON 中的每個鍵值對
    for program_number, gcode_text in json_data.items():
        gcode_map[program_number] = gcode_text.replace("\r\n", "\n")

    return gcode_map


def load_raw_gcodes(conf, verbose=True):
    """
    读取原始Gcode，优先读取gcode_from_json，其次读取g_code_directory
    """
    if verbose:
        print(
            f"[INFO] Starting load_raw_gcodes - 夾位名稱: {conf.get('clamping_name', '未設置')} - {conf['path']['dir_machine_folder']}"
        )

    if conf["path"]["dir_machine_folder"] != "製工標準":
        conf["path"]["dir_machine_data"] = conf["path"]["dir_machine_data"].format(
            department=conf["department"], folder=conf["path"]["dir_machine_folder"]
        )
        try:
            gcodes_dict = read_gcode_from_json(conf["path"]["dir_machine_data"])
            gcodes_dict = normalize_program_keys(gcodes_dict)
            if verbose:
                print(
                    f"[INFO] 非製工標准，load gcodes from json {conf['path']['dir_machine_data']} successed"
                )
            return gcodes_dict
        except:
            if verbose:
                print(
                    f"[INFO] 非製工標准，load gcodes from json {conf['path']['dir_machine_data']} failed"
                )
            pass
    else:
        conf["path"]["dir_app"] = conf["path"]["dir_app"].format(
            department=conf["department"]
        )

    g_code_directory = (
        f"{conf['path']['dir_app']}/{conf['clamping_name']}/{conf['path']['dir_gcode']}"
    )
    if verbose:
        print(f"[INFO] G代碼目錄路徑: {g_code_directory}")

    gcodes_dict = {}
    for each in os.listdir(g_code_directory):
        sub_program = each.replace(".txt", "").replace("O", "")
        try:
            # 嘗試使用 UTF-8 編碼讀取
            with open(f"{g_code_directory}/{each}", "r", encoding="utf-8") as f:
                gcodes_dict[f"O{sub_program}"] = f.read()
        except UnicodeDecodeError:
            # 如果 UTF-8 失敗，嘗試使用 Big5 編碼（常用於繁體中文）
            try:
                with open(f"{g_code_directory}/{each}", "r", encoding="big5") as f:
                    gcodes_dict[f"O{sub_program}"] = f.read()
            except UnicodeDecodeError:
                # 如果 Big5 也失敗，嘗試使用 GB2312 編碼（常用於簡體中文）
                try:
                    with open(
                        f"{g_code_directory}/{each}", "r", encoding="gb2312"
                    ) as f:
                        gcodes_dict[f"O{sub_program}"] = f.read()
                except UnicodeDecodeError:
                    # 如果所有嘗試都失敗，使用二進制模式讀取並忽略錯誤
                    with open(
                        f"{g_code_directory}/{each}",
                        "r",
                        encoding="utf-8",
                        errors="ignore",
                    ) as f:
                        gcodes_dict[f"O{sub_program}"] = f.read()
                    # print(
                    #     f"警告：文件 {each} 使用了不支持的編碼，某些字符可能無法正確顯示"
                    # )

    if verbose:
        print(f"[INFO] load gcodes from txt in {g_code_directory}")
    return gcodes_dict
    # except:
    #     print("沒有讀取到Gcode，請檢查")
    #     return {}


def normalize_program_keys(input_dict):
    """
    將字典中的鍵名格式化為 'Oxxxx'，其中 xxxx 為四位數字。
    如果鍵已經是 'Oxxxx' 格式，則保持不變。

    Args:
        input_dict (dict): 輸入的字典

    Returns:
        dict: 格式化鍵名後的新字典
    """
    result_dict = {}

    for key, value in input_dict.items():
        # 檢查鍵是否已經是 'Oxxxx' 格式
        if isinstance(key, str) and key.startswith("O") and key[1:].isdigit():
            # 已經是正確格式，直接保留
            result_dict[key] = value
        # 檢查鍵是否為四位數字字串
        elif isinstance(key, str) and key.isdigit():
            # 轉換為 'Oxxxx' 格式
            result_dict[f"O{key}"] = value
        else:
            # 其他格式的鍵保持不變
            result_dict[key] = value

    return result_dict
