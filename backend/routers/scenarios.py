from fastapi import APIRouter, HTTPException, status
from models.scenarios import Scenario, ScenarioCreate, ScenarioIteration, ScenarioIterationResponse, ScenariosResponse
from services.scenarios_service import get_all_scenarios, get_scenario_by_id, create_new_scenario, update_scenario_by_id, iterate_scenario

router = APIRouter()

@router.get("/", response_model=ScenariosResponse)
async def get_scenarios():
    scenarios = get_all_scenarios()
    return ScenariosResponse(scenarios=scenarios)

@router.get("/{scenario_id}", response_model=Scenario)
async def get_scenario(scenario_id: str):
    scenario = get_scenario_by_id(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.post("/", response_model=Scenario)
async def create_scenario(scenario_data: ScenarioCreate):
    scenario = create_new_scenario(scenario_data)
    return scenario

@router.put("/{scenario_id}", response_model=Scenario)
async def update_scenario(scenario_id: str, scenario_data: dict):
    scenario = update_scenario_by_id(scenario_id, scenario_data)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario

@router.post("/{scenario_id}/iterate", response_model=ScenarioIterationResponse)
async def iterate_scenario_endpoint(scenario_id: str, iteration_data: ScenarioIteration):
    result = iterate_scenario(scenario_id, iteration_data)
    if not result:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return result
