import numpy as np
from cnc_genai.src.simulation.colors import *


THICKNESS = 2.2
EDGE_TOP = 7
EDGE_BOTTOM = 8
EDGE_LEFT = 7
EDGE_RIGHT = 7
SIZE_X = 625
SIZE_Y = 323
SIZE_Z = 20
precision = 4

# 注意xy
size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

thickness = int(THICKNESS * 10 ** (precision - 3))
edge_top = int(EDGE_TOP * 10 ** (precision - 3))
edge_bottom = int(EDGE_BOTTOM * 10 ** (precision - 3))
edge_left = int(EDGE_LEFT * 10 ** (precision - 3))
edge_right = int(EDGE_RIGHT * 10 ** (precision - 3))

image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
image[:] = MATERIAL_COLOR
image[thickness:-thickness, thickness:-thickness, thickness:-thickness] = EMPTY_COLOR
image[edge_left:-edge_right, edge_bottom:-edge_top, thickness:] = EMPTY_COLOR

stock = image.copy()
