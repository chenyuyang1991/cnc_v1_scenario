from cnc_genai.src.utils.utils import read_gcode_from_json, normalize_program_keys
from cnc_genai.src.code_parsing.code_parsing import structure_cnc_code
from cnc_genai.src.v1_algo.generate_nc_code import validate_codes
import re
import pandas as pd


def decrease_feed_rate(
    src_code_path: str,
    main_fucntion: str,
    decrease_multiplier: float,
):
    """
    Decrease the feed rate of the main function in the source code.

    Args:
        src_code_path: The path to the source code file.
        main_fucntion: The name of the main function.
        decrease_multiplier: The multiplier to decrease the feed rate.
    """

    # 1. load gcodes
    old_codes_dict = read_gcode_from_json(src_code_path)
    old_codes_dict = normalize_program_keys(old_codes_dict)
    print(f"[INFO] load gcodes from json {src_code_path} successed")

    # 2. parse gcodes
    main_gcode = old_codes_dict[main_fucntion]
    df, _, _ = structure_cnc_code(
        main_gcode,
        gcodes_dict=old_codes_dict,
        macro={},
        inspect_sub_program=True,
        init_rotation=[],
        main_program_id=main_fucntion,
        verbose=False,
        output_df=True,
    )
    df = df.drop_duplicates(["sub_program", "row_id", "src"], keep="last")

    # 3. decrease feed rate
    def decrease_row(row):
        if row["src"] in ["G01", "G02", "G03"]:
            row["F_adjusted"] = row["F"] * decrease_multiplier
        return row

    df = df.apply(decrease_row, axis=1)

    # 4. generate new codes
    new_codes_dict = {}
    for _, sub_program in enumerate(df["sub_program"].unique(), start=1):
        print(f"[INFO] 生成新代碼 {sub_program}")
        code_lines = []
        df_sub_program = df[
            df["sub_program"].astype(str).str.zfill(4) == str(sub_program).zfill(4)
        ].reset_index(drop=True)
        df_sub_program = df_sub_program.assign(
            move_code_prev=df_sub_program["move_code"].shift(1)
        )
        num_rows_M98 = 0
        for i, row in enumerate(old_codes_dict[sub_program].split("\r\n")):
            if row not in df_sub_program["src"].values:
                code_lines.append(row)
                num_rows_M98 += 1
            else:
                df_row = df_sub_program.iloc[i + num_rows_M98]
                assert (
                    row == df_row["src"]
                ), f"[ERROR] row {row} != row_src {df_row['src']}"
                src_code = df_row["src"]
                if (
                    re.match(
                        r"^N\d+$", src_code
                    )  # 形如Nxx（x為數字）的純標籤行不改速度
                    or src_code.startswith("#")
                    or not len(src_code)
                ):
                    pass
                elif row["move_code"] in [
                    "G01",
                    "G02",
                    "G03",
                ]:  # , "G81", "G82", "G83"]:
                    if not pd.isna(row["F_adjusted"]):
                        if "F" in src_code:
                            # 如果F_adjusted和F_adjusted_prev相同，且上一行不是G00，则删除Fxx
                            if (
                                row["F_adjusted"] == row["F_adjusted_prev"]
                                and row["move_code_prev"] != "G00"
                            ):
                                src_code = re.sub(r"F\d+(\.)?(\d+)?", "", src_code)
                            # 如果F_adjusted和F_adjusted_prev不相同，则替换Fxx
                            else:
                                src_code = re.sub(
                                    r"F\d+(\.)?(\d+)?",
                                    f"F{int(row['F_adjusted'])}",
                                    src_code,
                                )
                        else:
                            # 新增條件：如果前一條是G00則強制添加F參數
                            if (
                                row["F_adjusted"] != row["F_adjusted_prev"]
                                or row["move_code_prev"] == "G00"
                            ):
                                src_code = src_code + f"F{int(row['F_adjusted'])}"
                            # 如果F_adjusted和F_adjusted_prev相同，则不做任何操作
                            else:
                                pass
                    else:
                        # 如果代碼有設置F，则删除Fxx，如果前一條是G00則保留F參數
                        if "F" in src_code and row["move_code_prev"] != "G00":
                            src_code = re.sub(r"F\d+(\.)?(\d+)?", "", src_code)

                code_lines.append(src_code)
        new_codes_dict[sub_program] = "\n".join(code_lines)
        # 驗證新代碼是否與舊代碼一致
        assert validate_codes(sub_program, new_codes_dict, old_codes_dict)

    return new_codes_dict, old_codes_dict


if __name__ == "__main__":
    decrease_feed_rate(
        src_code_path="../cnc_data/nc_code/B05-2F#Diamond-Cell#CNC5#M19#2025-02-26/programs.txt",
        main_fucntion="O5000",
        decrease_multiplier=0.5,
    )
