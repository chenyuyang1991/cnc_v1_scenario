from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.auth_service import get_current_user

router = APIRouter()


@router.get("/")
async def get_projects(db: Session = Depends(get_db)):
    """獲取專案列表"""
    return {
        "projects": [
            {"id": "X1111-CNC2", "name": "航太零件 A", "description": "航太零件加工專案"},
            {"id": "DM-CNC5", "name": "汽車零件 B", "description": "汽車零件製造專案"},
            {"id": "TG-CNC6", "name": "醫療器材 C", "description": "醫療器材精密加工"}
        ]
    }


@router.get("/{project_id}")
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """獲取單個專案"""
    return {
        "id": project_id,
        "name": f"專案 {project_id}",
        "description": "專案描述"
    } 