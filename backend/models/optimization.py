from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class OptimizationRequest(BaseModel):
    project_id: str
    config: Dict[str, Any]
    files: List[str]

class OptimizationResults(BaseModel):
    optimization_id: str
    status: str
    summary: Dict[str, Any]
    charts: Dict[str, Any]
    code_diff: Dict[str, Any]
    simulation: Dict[str, Any]
    validation: Dict[str, Any]
    created: str

class OptimizationConfig(BaseModel):
    machine: Dict[str, Any]
    material: Dict[str, Any]
    tooling: Dict[str, Any]
    optimization: Dict[str, Any]
    safety: Dict[str, Any]
