import pandas as pd
import numpy as np
import yaml


def load_ct_log(conf):

    df_ct = pd.read_excel(conf["dir"] + conf["ct_log"])

    return df_ct


if __name__ == "__main__":

    with open("conf/data_path.yaml", "r") as file:
        conf = yaml.safe_load(file)

    df_ct = load_ct_log(conf)

    print("loaded")
