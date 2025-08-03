# -*- coding: utf-8 -*-
import pandas as pd
import os
import json
import argparse
from cnc_genai.src.utils.utils import normalize_program_keys


def convert_txt_to_json(input_folder=None, output_folder=None):
    parser = argparse.ArgumentParser(description="將文字檔案轉換為 JSON 格式")
    parser.add_argument("-i", "--input", required=True, help="輸入資料夾路徑")
    parser.add_argument("-o", "--output", required=True, help="輸出資料夾路徑")
    args = parser.parse_args()
    if input_folder is None:
        input_folder = args.input
    if output_folder is None:
        output_folder = args.output

    output = {}
    for each in os.listdir(input_folder):
        try:
            # 首先嘗試 utf-8 編碼
            with open(os.path.join(input_folder, each), "r", encoding="utf-8") as f:
                content = f.read().replace("\n", "\r\n")
        except UnicodeDecodeError:
            # 如果 utf-8 失敗，嘗試 gbk 編碼
            with open(os.path.join(input_folder, each), "r", encoding="gbk") as f:
                content = f.read().replace("\n", "\r\n")

        each_name = each.replace(".txt", "")
        each_name = f"O{each_name}" if each_name[0] != "O" else each_name
        output[each_name] = content

    output = normalize_program_keys(output)

    os.makedirs(output_folder, exist_ok=True)
    # 寫入文件時指定編碼為 utf-8
    with open(os.path.join(output_folder, "programs.txt"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    convert_txt_to_json()
