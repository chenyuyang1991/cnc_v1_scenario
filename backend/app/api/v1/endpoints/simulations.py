from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db

router = APIRouter()


@router.get("/")
async def get_simulations(db: Session = Depends(get_db)):
    """獲取模擬列表"""
    return {
        "simulations": [
            {
                "id": "SIM-001",
                "name": "批次處理 A",
                "project": "PRJ-001",
                "status": "completed",
                "created": "2024-01-15T10:30:00Z",
                "progress": 100
            }
        ]
    } 