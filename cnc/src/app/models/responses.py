from typing import Dict, Optional, List, Union, Any
from pydantic import BaseModel


class CreateCaseResponse(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    event: str
    data: str


class EndResponse(BaseModel):
    event: str
    data: str


class DataFrameResponse(BaseModel):
    event: str
    data: List[Dict[str, Optional[Union[str, float, int]]]]


class UserDataFrameResponse(BaseModel):
    event: str
    data: List[Dict[str, Optional[Union[str, float, int]]]]


class InjectionResponse(BaseModel):
    response_graph: str
    response_dataframe: List[Dict[str, Optional[Union[str, float, int]]]]


class InjectionDiagnosisResponse(BaseModel):
    response_string: str
    response_dataframe: List[Dict[str, Optional[Union[str, float, int]]]]
    response_current_node: str


class ChatSessionResponse(BaseModel):
    text: Optional[str]
    setpoint: Optional[Dict]
    graph: Optional[List[str]]
    image: Optional[List[str]]
    dataframe: Optional[List[Dict]]
    dataframe_schema: Optional[List[Dict]]
    tree: Optional[str]
    current_node: Optional[str]
    nodes_path: Optional[List[str]]
    next_step: int
    buttons: Optional[List[str]]
    faq: Optional[List[str]]
    is_chat: bool
    pages: Optional[Dict]


class UpdateResponse(BaseModel):
    response: str = "Session Initialised!"


class DataValidationResponse(BaseModel):
    is_valid: bool
    file_id: str
    errors: Optional[List[str]]
    warnings: Optional[List[str]]
    errors_text: Optional[str]
    warnings_text: Optional[str]
