from fastapi import APIRouter, HTTPException, status
from models.optimization import OptimizationRequest, OptimizationResults, OptimizationConfig
from services.optimization_service import run_optimization, get_optimization_results, get_optimization_config, get_optimization_status
import asyncio

router = APIRouter()

@router.post("/run")
async def run_optimization_endpoint(optimization_data: OptimizationRequest):
    result = await run_optimization(optimization_data)
    return result

@router.get("/{optimization_id}/status")
async def get_optimization_status_endpoint(optimization_id: str):
    """Get optimization status"""
    status_info = get_optimization_status(optimization_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Optimization not found")
    return status_info

@router.get("/{optimization_id}/results", response_model=OptimizationResults)
async def get_optimization_results_endpoint(optimization_id: str):
    results = get_optimization_results(optimization_id)
    if not results:
        raise HTTPException(status_code=404, detail="Optimization results not found")
    return results

@router.get("/config", response_model=OptimizationConfig)
async def get_optimization_config_endpoint():
    config = get_optimization_config()
    return config
