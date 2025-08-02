from models.scenarios import Scenario, ScenarioCreate, ScenarioIteration, ScenarioIterationResponse
from datetime import datetime
from typing import List, Optional
import uuid

mock_scenarios = [
    {
        "id": "SCN-001",
        "name": "汽車零件加工專案",
        "project": "PRJ-001",
        "date": "2024-01-15",
        "type": "專案",
        "status": "completed",
        "version": "1.2",
        "completion": "92"
    },
    {
        "id": "SCN-002",
        "name": "航空零件製造專案", 
        "project": "PRJ-002",
        "date": "2024-01-14",
        "type": "專案",
        "status": "running",
        "version": "2.1",
        "completion": "88"
    },
    {
        "id": "SCN-003",
        "name": "精密模具專案",
        "project": "PRJ-001", 
        "date": "2024-01-13",
        "type": "專案",
        "status": "completed",
        "version": "1.0",
        "completion": "95"
    }
]

def get_all_scenarios() -> List[Scenario]:
    return [Scenario(**scenario) for scenario in mock_scenarios]

def get_scenario_by_id(scenario_id: str) -> Optional[Scenario]:
    for scenario in mock_scenarios:
        if scenario["id"] == scenario_id:
            return Scenario(**scenario)
    return None

def create_new_scenario(scenario_data: ScenarioCreate) -> Scenario:
    new_scenario = {
        "id": f"SCN-{str(uuid.uuid4())[:8].upper()}",
        "name": scenario_data.name,
        "project": scenario_data.project_id,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "type": "專案",
        "status": "created",
        "version": "1.0",
        "completion": "0"
    }
    mock_scenarios.append(new_scenario)
    return Scenario(**new_scenario)

def update_scenario_by_id(scenario_id: str, scenario_data: dict) -> Optional[Scenario]:
    for i, scenario in enumerate(mock_scenarios):
        if scenario["id"] == scenario_id:
            scenario.update(scenario_data)
            mock_scenarios[i] = scenario
            return Scenario(**scenario)
    return None

def iterate_scenario(scenario_id: str, iteration_data: ScenarioIteration) -> Optional[ScenarioIterationResponse]:
    parent_scenario = get_scenario_by_id(scenario_id)
    if not parent_scenario:
        return None
    
    new_scenario_id = f"SCN-{str(uuid.uuid4())[:8].upper()}"
    
    new_scenario = {
        "id": new_scenario_id,
        "name": f"{parent_scenario.name} - 迭代",
        "project": parent_scenario.project,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "type": "迭代",
        "status": "created",
        "version": f"{float(parent_scenario.version) + 0.1:.1f}",
        "completion": "0"
    }
    mock_scenarios.append(new_scenario)
    
    return ScenarioIterationResponse(
        new_scenario_id=new_scenario_id,
        parent_scenario_id=scenario_id,
        iteration_type=iteration_data.iteration_type,
        status="created"
    )
