from fastapi import APIRouter, HTTPException, status
from models.simulations import Simulation, SimulationCreate, SimulationResults, SimulationsResponse
from services.simulations_service import get_all_simulations, get_simulation_by_id, create_new_simulation, run_simulation, get_simulation_results

router = APIRouter()

@router.get("/", response_model=SimulationsResponse)
async def get_simulations():
    simulations = get_all_simulations()
    return SimulationsResponse(simulations=simulations)

@router.get("/{simulation_id}", response_model=Simulation)
async def get_simulation(simulation_id: str):
    simulation = get_simulation_by_id(simulation_id)
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulation

@router.post("/", response_model=Simulation)
async def create_simulation(simulation_data: SimulationCreate):
    simulation = create_new_simulation(simulation_data)
    return simulation

@router.post("/{simulation_id}/run")
async def run_simulation_endpoint(simulation_id: str):
    result = run_simulation(simulation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return result

@router.get("/{simulation_id}/results", response_model=SimulationResults)
async def get_simulation_results_endpoint(simulation_id: str):
    results = get_simulation_results(simulation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Simulation results not found")
    return results
