# for grayscale images
MATERIAL_MASK_COLOR = 255
EMPTY_MASK_COLOR = 0
CUTTING_MASK_COLOR = 1  # 无材料的路径
PATH_MASK_COLOR = 2  # 被切掉的工件

# for RGB images
MATERIAL_COLOR = (0, 255, 0)
EMPTY_COLOR = (0, 0, 0)
IGNORE_COLOR = (255, 255, 255)
CUTTING_COLOR = (0, 0, 255)  # 无材料的路径


# step color for tracing
def get_step_color(idx):
    """
    根据索引生成一个颜色元组。

    这个函数的目的是为了生成一个颜色，这个颜色的RGB值根据索引值变化。
    颜色的红色和绿色分量是通过索引值进行计算得到的，而蓝色分量固定为128。

    参数:
    idx (int): 用于计算颜色的索引值。

    返回值:
    tuple: 包含三个整数的元组，分别代表红色、绿色和蓝色分量。

    颜色生成的逻辑:
    - 红色分量: 索引值对256取模，确保值在0到255之间。
    - 绿色分量: 索引值除以256后的商再对256取模，确保值在0到255之间。
    - 蓝色分量: 固定值128，为颜色添加一定的饱和度。
    """
    return (
        (idx % 256),  # 计算红色分量
        (idx // 256) % 256,  # 计算绿色分量
        128,  # 固定蓝色分量
    )


def get_step_from_color(rgb):
    assert rgb[2] == 128, "Color is not step color"
    step = rgb[0] + rgb[1] * 256
    return step
