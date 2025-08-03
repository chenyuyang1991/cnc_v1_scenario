import warnings

warnings.filterwarnings("ignore")
from cnc_genai.src.v1_algo.generate_nc_code import run_generate_nc_code
from cnc_genai.src.utils import utils


if __name__ == "__main__":

    conf = utils.load_config_v1(base_config_path="cnc_genai/conf/v1_config.yaml")

    print(f"scenario config {conf['scenario_name']} has loaded")

    new_codes, old_codes, out_df = run_generate_nc_code(conf)

    print("done")
