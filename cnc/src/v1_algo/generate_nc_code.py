import os
import re
import pandas as pd
from cnc_genai.src.v1_algo.adjust_feed_rate import run_adjust_feed_rate
from cnc_genai.src.utils.utils import load_raw_gcodes


def _validate_df(sub_program, new_code_df, old_code_df):

    for col in new_code_df.columns:
        if not any(x in col for x in ["F", "src", "command"]):
            try:
                if any(new_code_df[col].astype(str) != old_code_df[col].astype(str)):
                    # print(f"Validating: Column {col} not same")
                    # print(f"new {col}: {list(new_code_df[col])}")
                    # print(f"old {col}: {list(old_code_df[col])}")
                    return False
            except:
                print(sub_program, col)
                # print("new_code_df", new_code_df.index)
                # print("old_code_df", old_code_df.index)
    return True


def validate_codes(sub_program, new_codes, old_codes):

    new_lines = new_codes[sub_program].split("\n")
    old_lines = old_codes[sub_program].split("\n")

    assert len(new_lines) == len(
        old_lines
    ), f"Length not same: {len(new_lines)} vs {len(old_lines)}"

    for i, (new_line, old_line) in enumerate(zip(new_lines, old_lines)):
        # 刪除new_line中的F參數
        new_line_wo_F = re.sub(r"F\d+\.?\d*", "", new_line)
        old_line_wo_F = re.sub(r"F\d+\.?\d*", "", old_line)
        if new_line_wo_F != old_line_wo_F:
            if i != 0:
                print(f"Line {i} not same")
                print(f"new: {new_line}")
                print(f"old: {old_line}")
                return False
    return True


def run_generate_nc_code(conf, validate=True):

    out_df = run_adjust_feed_rate(conf)
    print(f"scenario_name, {conf['scenario_name']}")
    print(f"time_physical, {round(out_df.time_physical.sum(), 6)}")
    print(
        f'{round(out_df.time_physical_improved.sum(), 6)}, {round(out_df.drop_duplicates("sub_program")["real_ct"].sum(), 2)}'
    )
    print(
        f"uplift, {out_df.time_physical_improved.sum() / out_df.drop_duplicates('sub_program')['real_ct'].sum() * 100:.2f}%"
    )

    out_df["F_adjusted_prev"] = out_df["F_adjusted"].shift(1)
    out_df["sub_program"] = out_df["sub_program"].astype(str).str.zfill(4)

    # 舊代码
    new_codes = {}
    old_codes = load_raw_gcodes(conf)

    for idx, sub_program in enumerate(conf["sub_programs"].keys(), start=1):
        # 新代码
        print(f"[INFO] 生成新代碼 {sub_program}")
        code_lines = []
        df_sub_program = out_df[
            out_df["sub_program"].astype(str).str.zfill(4) == str(sub_program).zfill(4)
        ]

        df_sub_program = df_sub_program.assign(
            move_code_prev=df_sub_program["move_code"].shift(1)
        )

        for i, row in df_sub_program.reset_index().iterrows():

            src_code = row["src"]
            if (
                re.match(r"^N\d+$", src_code)  # 形如Nxx（x為數字）的純標籤行不改速度
                or src_code.startswith("#")
                or not len(src_code)
            ):
                pass
            elif row["move_code"] in ["G01", "G02", "G03"]:  # , "G81", "G82", "G83"]:
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
        new_codes[f"O{sub_program}"] = "\n".join(code_lines)

        # 與舊代碼進行驗證
        if validate:
            assert validate_codes(f"O{sub_program}", new_codes, old_codes)

    return new_codes, old_codes, out_df


if __name__ == "__main__":

    import warnings

    warnings.filterwarnings("ignore")
    from cnc_genai.src.utils import utils

    conf = utils.load_config_v1(base_config_path="cnc_genai/conf/v1_config.yaml")

    new_codes, _, _ = run_generate_nc_code(conf)

    out_dir = os.path.join(
        f'{conf["path"]["dir_intermediate"]}/{conf["clamping_name"]}/{conf["output_path"]["generate_code"]}',
        conf["scenario_name"],
    )
    os.makedirs(out_dir, exist_ok=True)
    all_codes_list = []
    for sub_program in conf["sub_programs"]:
        with open(f"{out_dir}/{sub_program}", "w") as f:
            f.write(new_codes[f"O{sub_program}"])
        all_codes_list.append(new_codes[f"O{sub_program}"])
        all_codes_list.append("\n")

    all_codes = "".join(all_codes_list)

    with open(f"{out_dir}/X2867_CNC2", "w") as f:
        f.write(all_codes)
