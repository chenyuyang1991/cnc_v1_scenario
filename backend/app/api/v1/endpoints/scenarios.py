from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db

router = APIRouter()


@router.get("/")
async def get_scenarios(db: Session = Depends(get_db)):
    """獲取場景列表"""
    return {
        "scenarios": [
            {
                "id": "SCN-001",
                "name": "汽車零件加工專案",
                "project": "PRJ-001",
                "date": "2024-01-15",
                "status": "completed",
                "version": "1.2",
                "completion": 92
            }
        ]
    } 