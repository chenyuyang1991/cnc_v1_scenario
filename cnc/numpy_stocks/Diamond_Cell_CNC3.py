import numpy as np
from cnc_genai.src.simulation.colors import *

DEEP_CHANNEL = 5.1
POS_CHANNEL_LEFT = 10.75
POS_CHANNEL_LEFT_TOP = 11.5
POS_CHANNEL_LEFT_BOTTOM = 11
POS_CHANNEL_RIGHT = 10.5
POS_CHANNEL_RIGHT_TOP = 10.8
POS_CHANNEL_RIGHT_BOTTOM = 10.8
W_CHANNEL = 12.5
SIZE_X = 255.60
SIZE_Y = 192.50
SIZE_Z = 8.05
precision = 4

# 注意xy
size = np.array([SIZE_X, SIZE_Y, SIZE_Z])
pixel_size = np.round(size * 10 ** (precision - 3)).astype(int)

depth_channel = int(DEEP_CHANNEL * 10 ** (precision - 3))
width_channel = int(W_CHANNEL * 10 ** (precision - 3))
pos_channel_left = int(POS_CHANNEL_LEFT * 10 ** (precision - 3))
pos_channel_left_top = int(POS_CHANNEL_LEFT_TOP * 10 ** (precision - 3))
pos_channel_left_bottom = int(POS_CHANNEL_LEFT_BOTTOM * 10 ** (precision - 3))
pos_channel_right = int(POS_CHANNEL_RIGHT * 10 ** (precision - 3))
pos_channel_right_top = int(POS_CHANNEL_RIGHT_TOP * 10 ** (precision - 3))
pos_channel_right_bottom = int(POS_CHANNEL_RIGHT_BOTTOM * 10 ** (precision - 3))

image = np.zeros((pixel_size[1], pixel_size[0], pixel_size[2], 3), np.uint8)
image[:] = MATERIAL_COLOR

image[
    pos_channel_left : pos_channel_left + width_channel,
    pos_channel_left_top:-pos_channel_left_bottom,
    depth_channel:,
] = EMPTY_COLOR

image[
    (pixel_size[2] - pos_channel_right - width_channel) : (
        pixel_size[2] - pos_channel_right
    ),
    pos_channel_right_top:-pos_channel_right_bottom,
    depth_channel:,
] = EMPTY_COLOR

stock = image.copy()
