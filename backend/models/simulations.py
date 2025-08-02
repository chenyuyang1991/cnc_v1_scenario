from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Simulation(BaseModel):
    id: str
    name: str
    project: str
    status: str
    created: str
    config: Optional[Dict[str, Any]] = None

class SimulationCreate(BaseModel):
    name: str
    project_id: str
    config: Dict[str, Any]

class SimulationResults(BaseModel):
    simulation_id: str
    status: str
    results: Dict[str, Any]
    created: str

class SimulationsResponse(BaseModel):
    simulations: List[Simulation]
