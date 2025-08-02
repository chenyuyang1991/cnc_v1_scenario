from fastapi import APIRouter, HTTPException, status
from models.config import MachineConfig, MaterialConfig, ToolingConfig, ConfigSaveRequest
from services.config_service import get_machine_config, get_material_config, get_tooling_config, save_config

router = APIRouter()

@router.get("/machine", response_model=MachineConfig)
async def get_machine_config_endpoint():
    config = get_machine_config()
    return config

@router.get("/materials", response_model=MaterialConfig)
async def get_material_config_endpoint():
    config = get_material_config()
    return config

@router.get("/tooling", response_model=ToolingConfig)
async def get_tooling_config_endpoint():
    config = get_tooling_config()
    return config

@router.post("/{config_type}")
async def save_config_endpoint(config_type: str, config_data: ConfigSaveRequest):
    result = save_config(config_type, config_data.config_data)
    return result
