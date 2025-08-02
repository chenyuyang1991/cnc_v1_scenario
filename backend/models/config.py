from pydantic import BaseModel
from typing import List, Dict, Any

class MachineConfig(BaseModel):
    machines: List[Dict[str, Any]]

class MaterialConfig(BaseModel):
    materials: List[Dict[str, Any]]

class ToolingConfig(BaseModel):
    tools: List[Dict[str, Any]]

class ConfigSaveRequest(BaseModel):
    config_data: Dict[str, Any]
