import numpy as np
from cnc_genai.src.simulation.colors import *
from cnc_genai.src.simulation.utils import save_to_zst


SIZE_X = 173  # 173
SIZE_Y = 85  # 85
SIZE_Z = 9.2  # 9.2
precision = 4

# 注意xy
size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
image[:] = MATERIAL_COLOR

stock = image.copy()
save_to_zst(stock, f"../app/mac3/simulation_master/FX2319-CNC4/stock.zst")
