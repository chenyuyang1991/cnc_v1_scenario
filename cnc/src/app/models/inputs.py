from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
from datetime import datetime
# from model.agent.langchain.src.utils.utils import TIMEZONE
import pytz
TIMEZONE = pytz.timezone("Asia/Shanghai")

class ReturnModel(BaseModel):
    session_id: str


class CaseModel(BaseModel):
    case_name: Optional[str] = None

    @validator("case_name", pre=True, always=True)
    def set_case_name(cls, v):
        if v is None:
            return datetime.now(tz=TIMEZONE).strftime("%H:%M:%S")
        if isinstance(v, str):
            return v
        raise ValueError("case_name must be a string or None")


class SessionModel(BaseModel):
    session_id: str
    machine: Dict
    material: Dict
    mold: Dict
    approach: Optional[str] = "老師傅"


class IssueModel(BaseModel):
    type: str
    data: Dict


class ChatModel(BaseModel):
    session_id: str
    query: str


class ConfigureModel(BaseModel):
    mold_id: str
    machine_id: str


class GraphModel(BaseModel):
    射速_mm_s: int = Field(alias="射速")
    填充時間_s: str = Field(alias="填充時間(s)")
    注射峰值壓力: str
    剪切率: float
    強化率: float
    有效黏度: float
    有效黏度變化值: Optional[float] = None
    是否出現外觀缺陷: str
    是否壓力受限: str


class InjectionGraphModel(BaseModel):
    session_id: str
    data: List[GraphModel]


class SetpointModel(BaseModel):
    母模模溫_c: Optional[float] = Field(None, alias="母模模溫(°C)")
    公模模溫_c: Optional[float] = Field(None, alias="公模模溫(°C)")
    滑塊模溫_c: Optional[float] = Field(None, alias="滑塊模溫(°C)")
    冷卻時間_s: Optional[float] = Field(None, alias="冷却时间(s)")
    背壓: Optional[float] = Field(None, alias="背壓")
    熔膠速度: Optional[float] = Field(None, alias="熔膠速度")
    料筒溫度_第1段_c: Optional[float] = Field(None, alias="料筒溫度-第1段(°C)")
    料筒溫度_第2段_c: Optional[float] = Field(None, alias="料筒溫度-第2段(°C)")
    料筒溫度_第3段_c: Optional[float] = Field(None, alias="料筒溫度-第3段(°C)")
    料筒溫度_第4段_c: Optional[float] = Field(None, alias="料筒溫度-第4段(°C)")
    料筒溫度_第5段_c: Optional[float] = Field(None, alias="料筒溫度-第5段(°C)")
    計量行程_mm: Optional[float] = Field(None, alias="計量行程(mm)")
    後鬆退距離_mm: Optional[float] = Field(None, alias="後鬆退距離(mm)")
    保壓切換位置_mm: Optional[float] = Field(None, alias="保壓切換位置(mm)")
    射出壓力_第1段: Optional[float] = Field(None, alias="射出壓力-第1段")
    射出壓力_第2段: Optional[float] = Field(None, alias="射出壓力-第2段")
    射出壓力_第3段: Optional[float] = Field(None, alias="射出壓力-第3段")
    射出壓力_第4段: Optional[float] = Field(None, alias="射出壓力-第4段")
    射出壓力_第5段: Optional[float] = Field(None, alias="射出壓力-第5段")
    射出終點位置_第1段_mm: Optional[float] = Field(None, alias="射出終點位置-第1段(mm)")
    射出終點位置_第2段_mm: Optional[float] = Field(None, alias="射出終點位置-第2段(mm)")
    射出終點位置_第3段_mm: Optional[float] = Field(None, alias="射出終點位置-第3段(mm)")
    射出終點位置_第4段_mm: Optional[float] = Field(None, alias="射出終點位置-第4段(mm)")
    射出終點位置_第5段_mm: Optional[float] = Field(None, alias="射出終點位置-第5段(mm)")
    射出速度_第1段: Optional[float] = Field(None, alias="射出速度-第1段")
    射出速度_第2段: Optional[float] = Field(None, alias="射出速度-第2段")
    射出速度_第3段: Optional[float] = Field(None, alias="射出速度-第3段")
    射出速度_第4段: Optional[float] = Field(None, alias="射出速度-第4段")
    射出速度_第5段: Optional[float] = Field(None, alias="射出速度-第5段")
    保壓壓力_第1段: Optional[float] = Field(None, alias="保壓壓力-第1段")
    保壓壓力_第2段: Optional[float] = Field(None, alias="保壓壓力-第2段")
    保壓壓力_第3段: Optional[float] = Field(None, alias="保壓壓力-第3段")
    保壓速度_第1段: Optional[float] = Field(None, alias="保壓速度-第1段")
    保壓速度_第2段: Optional[float] = Field(None, alias="保壓速度-第2段")
    保壓速度_第3段: Optional[float] = Field(None, alias="保壓速度-第3段")
    保壓時間_第1段_s: Optional[float] = Field(None, alias="保壓時間-第1段(s)")
    保壓時間_第2段_s: Optional[float] = Field(None, alias="保壓時間-第2段(s)")
    保壓時間_第3段_s: Optional[float] = Field(None, alias="保壓時間-第3段(s)")
    延遲時間_1號澆口_s: Optional[float] = Field(None, alias="延遲時間-1號澆口(s)")
    延遲時間_2號澆口_s: Optional[float] = Field(None, alias="延遲時間-2號澆口(s)")
    延遲時間_3號澆口_s: Optional[float] = Field(None, alias="延遲時間-3號澆口(s)")
    延遲時間_4號澆口_s: Optional[float] = Field(None, alias="延遲時間-4號澆口(s)")
    延遲時間_5號澆口_s: Optional[float] = Field(None, alias="延遲時間-5號澆口(s)")
    延遲時間_6號澆口_s: Optional[float] = Field(None, alias="延遲時間-6號澆口(s)")

    class Config:
        populate_by_name = True


class UpdateModel(BaseModel):
    session_id: str
    setpoint: SetpointModel


class ImageUploadModel(BaseModel):
    image_name: str
    image: str


class HistoricalSessionInputModel(BaseModel):
    machine_id: str
    mold_id: str
    session_start_date: str
    session_end_date: str
