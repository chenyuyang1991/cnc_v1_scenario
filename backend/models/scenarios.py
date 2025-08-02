from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Scenario(BaseModel):
    id: str
    name: str
    project: str
    date: str
    type: str
    status: str
    version: str
    completion: str

class ScenarioCreate(BaseModel):
    name: str
    project_id: str
    machine_id: str

class ScenarioIteration(BaseModel):
    iteration_type: str
    changes: Dict[str, Any]

class ScenarioIterationResponse(BaseModel):
    new_scenario_id: str
    parent_scenario_id: str
    iteration_type: str
    status: str

class ScenariosResponse(BaseModel):
    scenarios: List[Scenario]
