from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db

router = APIRouter()


@router.get("/machine")
async def get_machine_config(db: Session = Depends(get_db)):
    """獲取機台配置"""
    return {
        "machines": [
            {
                "id": "cnc-001",
                "name": "Haas VF-2",
                "type": "milling",
                "specs": {
                    "max_spindle_speed": 8000,
                    "max_feed_rate": 1000,
                    "work_area": "762x406x508"
                }
            }
        ]
    }


@router.get("/materials")
async def get_material_config(db: Session = Depends(get_db)):
    """獲取材料配置"""
    return {
        "materials": [
            {
                "id": "aluminum-6061",
                "name": "鋁合金 6061-T6",
                "density": 2.7,
                "hardness": 95,
                "cutting_speed": 300,
                "feed_rate": 0.1
            }
        ]
    }


@router.get("/tooling")
async def get_tooling_config(db: Session = Depends(get_db)):
    """獲取刀具配置"""
    return {
        "tools": [
            {
                "id": "endmill-6mm",
                "name": "端銑刀 6mm",
                "type": "end_mill",
                "diameter": 6,
                "flutes": 4,
                "material": "carbide"
            }
        ]
    } 