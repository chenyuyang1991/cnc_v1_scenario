import argparse
import numpy as np
from cnc_genai.src.simulation.utils import save_to_zst, load_from_zst

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

product, origin = load_from_zst(args.input)

product = product.transpose(2, 1, 0, 3)
product = np.flip(product, axis=1)
product = np.flip(product, axis=2)

save_to_zst(product, args.output)
