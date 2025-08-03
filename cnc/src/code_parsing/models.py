from pydantic import BaseModel, Field
from typing import Optional


class CurrentStatus(BaseModel):

    sub_program: Optional[str] = None

    # 使用 alias 映射外部字段 "coordinates_abs/rel"
    coordinates_abs_rel: str = "absolute"

    coordinates_sys: str = "G54"
    unit: str = "公制單位"
    precision_mode: bool = False
    move_code: Optional[str] = None
    panel_selected: str = "XY"
    call_func: Optional[str] = None

    # 带点号的字段通过 alias 与外部名称对应
    G04_time: Optional[float] = None
    G10_L: Optional[int] = None
    G54p1_P: Optional[int] = None
    G54p1_X: Optional[float] = None
    G54p1_Y: Optional[float] = None

    # 工件旋轉相關
    G68_X: Optional[float] = None
    G68_Y: Optional[float] = None
    G68_R: Optional[float] = None

    # G81鑽孔循環相關
    G81_X: Optional[float] = None  # 鑽孔X位置
    G81_Y: Optional[float] = None  # 鑽孔Y位置
    G81_Z: Optional[float] = None  # 鑽孔深度

    # G83深孔鑽削循環相關
    G83_X: Optional[float] = None  # 深孔X位置
    G83_Y: Optional[float] = None  # 深孔Y位置
    G83_Z: Optional[float] = None  # 深孔最終深度
    G83_Q: Optional[float] = None  # 每次切入深度

    # G98/G99返回平面相關
    G98_Z: Optional[float] = None  # G98初始平面（循環開始前的Z軸位置）
    G99_R: Optional[float] = None  # G99快速進給平面（R平面）

    # 基本運動參數
    O: Optional[str] = None
    N: Optional[str] = None
    G: Optional[str] = None
    M: Optional[str] = None
    T: Optional[str] = None
    X: Optional[float] = None
    Y: Optional[float] = None
    Z: Optional[float] = None
    S: Optional[float] = None
    F: Optional[float] = None
    H: Optional[float] = None
    D: Optional[float] = None
    I: Optional[float] = None
    J: Optional[float] = None
    K: Optional[float] = None
    A: Optional[float] = None
    B: Optional[float] = None
    C: Optional[float] = None

    # 記錄上一個位置
    X_prev: Optional[float] = None
    Y_prev: Optional[float] = None
    Z_prev: Optional[float] = None

    class Config:
        extra = "allow"
