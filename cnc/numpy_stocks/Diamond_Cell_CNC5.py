import numpy as np
from cnc_genai.src.simulation.colors import *


THICKNESS = 2.94
EDGE_TOP = 6.35
EDGE_BOTTOM = 6.7
EDGE_LEFT = 4
EDGE_RIGHT = 5

EDGE_LEFT_ = 6.8
EDGE_RIGHT_MID = 7.3

EDGE_LEFT_POS_START = 70.5 + 6.35
EDGE_LEFT_POS_END = 74.5 + 6.7
EDGE_LEFT_SIZE = 6.8

EDGE_RIGHT_POS_START = 53.5 + 6.35
EDGE_RIGHT_POS_END = 78.5 + 6.7
EDGE_RIGHT_SIZE = 7.3

CYLINDER_LEFT_POS_X = 39 + 4
CYLINDER_LEFT_POS_Y = 109 + 6.35
CYLINDER_LEFT_SIZE_X = 25
CYLINDER_LEFT_SIZE_Y = 31

CYLINDER_RIGHT_POS_X = 30 + 5
CYLINDER_RIGHT_POS_Y = 26.7 + 6.35
CYLINDER_RIGHT_SIZE_X = 36
CYLINDER_RIGHT_SIZE_Y = 24

SIZE_X = 248.628  # mm
SIZE_Y = 180.74
SIZE_Z = 7.28

# 注意xy
precision = 4
size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

thickness = int(THICKNESS * 10 ** (precision - 3))
edge_top = int(EDGE_TOP * 10 ** (precision - 3))
edge_bottom = int(EDGE_BOTTOM * 10 ** (precision - 3))
edge_left = int(EDGE_LEFT * 10 ** (precision - 3))
edge_right = int(EDGE_RIGHT * 10 ** (precision - 3))

edge_left_pos_start = int(EDGE_LEFT_POS_START * 10 ** (precision - 3))
edge_left_pos_end = int(EDGE_LEFT_POS_END * 10 ** (precision - 3))
edge_left_size = int(EDGE_LEFT_SIZE * 10 ** (precision - 3))
edge_right_pos_start = int(EDGE_RIGHT_POS_START * 10 ** (precision - 3))
edge_right_pos_end = int(EDGE_RIGHT_POS_END * 10 ** (precision - 3))
edge_right_size = int(EDGE_RIGHT_SIZE * 10 ** (precision - 3))

cylinder_left_pos_x = int(CYLINDER_LEFT_POS_X * 10 ** (precision - 3))
cylinder_left_pos_y = int(CYLINDER_LEFT_POS_Y * 10 ** (precision - 3))
cylinder_left_size_x = int(CYLINDER_LEFT_SIZE_X * 10 ** (precision - 3))
cylinder_left_size_y = int(CYLINDER_LEFT_SIZE_Y * 10 ** (precision - 3))
cylinder_right_pos_x = int(CYLINDER_RIGHT_POS_X * 10 ** (precision - 3))
cylinder_right_pos_y = int(CYLINDER_RIGHT_POS_X * 10 ** (precision - 3))
cylinder_right_size_x = int(CYLINDER_RIGHT_SIZE_X * 10 ** (precision - 3))
cylinder_right_size_y = int(CYLINDER_RIGHT_SIZE_X * 10 ** (precision - 3))

image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
image[:] = MATERIAL_COLOR
image[edge_top:-edge_bottom, edge_left:-edge_right, thickness:] = EMPTY_COLOR

image[edge_left_pos_start:-edge_left_pos_end, :edge_left_size, :] = MATERIAL_COLOR
image[edge_right_pos_start:-edge_right_pos_end, -edge_right_size:, :] = MATERIAL_COLOR

image[
    cylinder_left_pos_y : cylinder_left_pos_y + cylinder_left_size_y,
    cylinder_left_pos_x : cylinder_left_pos_x + cylinder_left_size_x,
    :,
] = MATERIAL_COLOR
image[
    cylinder_right_pos_y : cylinder_right_pos_y + cylinder_right_size_y,
    -cylinder_right_pos_x - cylinder_right_size_x : -cylinder_right_pos_x,
    :,
] = MATERIAL_COLOR

stock = image.copy()
