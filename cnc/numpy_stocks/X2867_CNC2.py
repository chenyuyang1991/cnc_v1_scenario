import numpy as np
from cnc_genai.src.simulation.colors import *


THICKNESS = 2.3
EDGE_W = 8.2
EDGE_H = 10.2
SIZE_X = 548.63  # mm
SIZE_Y = 376.32
SIZE_Z = 12.46
precision = 4

# 注意xy
size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

thickness = int(THICKNESS * 10 ** (precision - 3))
edge_w = int(EDGE_W * 10 ** (precision - 3))
edge_h = int(EDGE_H * 10 ** (precision - 3))

image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
image[:] = MATERIAL_COLOR
image[thickness:-thickness, thickness:-thickness, thickness:-thickness] = EMPTY_COLOR
image[edge_w:-edge_w, edge_h:-edge_h, thickness:] = EMPTY_COLOR

stock = image.copy()
