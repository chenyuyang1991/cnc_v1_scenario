from models.simulations import Simulation, SimulationCreate, SimulationResults
from datetime import datetime
from typing import List, Optional
import uuid

mock_simulations = [
    {
        "id": "SIM-001",
        "name": "批次處理 A",
        "project": "X1111-CNC2",
        "status": "completed",
        "created": "2024-01-15",
        "config": {"batch_size": 100, "quality_check": True}
    },
    {
        "id": "SIM-002",
        "name": "品質測試 B",
        "project": "DM-CNC5",
        "status": "running",
        "created": "2024-01-14",
        "config": {"test_type": "quality", "samples": 50}
    },
    {
        "id": "SIM-003",
        "name": "效能測試 C",
        "project": "TG-CNC6",
        "status": "failed",
        "created": "2024-01-13",
        "config": {"test_type": "performance", "duration": 3600}
    },
    {
        "id": "SIM-004",
        "name": "負載測試 D",
        "project": "X1111-CNC2",
        "status": "pending",
        "created": "2024-01-12",
        "config": {"test_type": "load", "concurrent_users": 100}
    }
]

def get_all_simulations() -> List[Simulation]:
    return [Simulation(**simulation) for simulation in mock_simulations]

def get_simulation_by_id(simulation_id: str) -> Optional[Simulation]:
    for simulation in mock_simulations:
        if simulation["id"] == simulation_id:
            return Simulation(**simulation)
    return None

def create_new_simulation(simulation_data: SimulationCreate) -> Simulation:
    new_simulation = {
        "id": f"SIM-{str(uuid.uuid4())[:8].upper()}",
        "name": simulation_data.name,
        "project": simulation_data.project_id,
        "status": "created",
        "created": datetime.utcnow().strftime("%Y-%m-%d"),
        "config": simulation_data.config
    }
    mock_simulations.append(new_simulation)
    return Simulation(**new_simulation)

def run_simulation(simulation_id: str) -> Optional[dict]:
    for i, simulation in enumerate(mock_simulations):
        if simulation["id"] == simulation_id:
            simulation["status"] = "running"
            mock_simulations[i] = simulation
            return {"message": "Simulation started", "simulation_id": simulation_id}
    return None

def get_simulation_results(simulation_id: str) -> Optional[SimulationResults]:
    simulation = get_simulation_by_id(simulation_id)
    if not simulation:
        return None
    
    mock_results = {
        "simulation_id": simulation_id,
        "status": simulation.status,
        "results": {
            "execution_time": "45.2 seconds",
            "success_rate": "98.5%",
            "errors": [],
            "performance_metrics": {
                "throughput": "150 parts/hour",
                "efficiency": "92%",
                "quality_score": "98.5%"
            }
        },
        "created": simulation.created
    }
    
    return SimulationResults(**mock_results)
