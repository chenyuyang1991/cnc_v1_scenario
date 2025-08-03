import pandas as pd
import yaml


def load_sensor(conf):

    df_sensor_dict = {}
    df_sensor_dict["df_sensor_opt"] = proc_sensor(
        pd.read_json(conf["dir"] + conf["machine_experiment_sensor_opt"], lines=True)
    )
    df_sensor_dict["df_sensor"] = proc_sensor(
        pd.read_json(conf["dir"] + conf["machine_experiment_sensor"], lines=True)
    )
    return df_sensor_dict


def proc_sensor(df_sensor):
    # Updated regular expression to handle integers and floats
    df_sensor[["X", "Y", "Z"]] = df_sensor["path_position"].str.extract(
        r"X:(-?\d+\.?\d*)/Y:(-?\d+\.?\d*)/Z:(-?\d+\.?\d*)"
    )
    # Convert to numeric types if needed
    df_sensor[["X", "Y", "Z"]] = df_sensor[["X", "Y", "Z"]].apply(pd.to_numeric)

    df_sensor["processing_code"] = df_sensor["processing_code"].fillna(0).astype(int)
    df_sensor["processing_code"] = df_sensor["processing_code"].astype(str)

    return df_sensor


def add_seq_id(df, group_col=["processing_code"], sort_col=["datetime"]):
    df = df.sort_values(by=group_col + sort_col).reset_index(drop=True)
    df["seq_id"] = df.groupby(group_col).cumcount() + 1
    return df


def proc_sensors(df_sensor_dict):

    for df_name in df_sensor_dict.keys():
        df_sensor_dict[df_name] = add_seq_id(df_sensor_dict[df_name])


if __name__ == "__main__":

    with open("cnc_genai/conf/data_path.yaml", "r") as file:
        conf = yaml.safe_load(file)

    df_sensor_dict = load_sensor(conf)
    for df_name in df_sensor_dict.keys():
        df_sensor_dict[df_name].to_excel(
            f"{conf['dir_intermediate']}/sensor_exp/{df_name}.xlsx", index=False
        )
    print("done")
