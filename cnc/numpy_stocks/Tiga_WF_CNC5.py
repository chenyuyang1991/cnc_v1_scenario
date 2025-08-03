import numpy as np
from cnc_genai.src.simulation.colors import *


SIZE_X = 247.64
SIZE_Y = 178.52
SIZE_Z = 6.32
precision = 4

# 注意xy
size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
image[:] = MATERIAL_COLOR

stock = image.copy()
