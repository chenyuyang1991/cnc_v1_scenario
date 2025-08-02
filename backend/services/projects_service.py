from models.projects import Project, ProjectCreate, ProjectUpdate
from datetime import datetime
from typing import List, Optional
import uuid

mock_projects = [
    {
        "id": "PRJ-001",
        "name": "航太零件 A",
        "description": "高精度航太零件加工專案",
        "created": "2024-01-15T10:30:00Z",
        "status": "active"
    },
    {
        "id": "PRJ-002", 
        "name": "汽車零件 B",
        "description": "汽車引擎零件製造專案",
        "created": "2024-01-14T09:15:00Z",
        "status": "active"
    },
    {
        "id": "PRJ-003",
        "name": "醫療器材 C", 
        "description": "醫療設備精密零件專案",
        "created": "2024-01-13T14:20:00Z",
        "status": "active"
    }
]

def get_all_projects() -> List[Project]:
    return [Project(**project) for project in mock_projects]

def get_project_by_id(project_id: str) -> Optional[Project]:
    for project in mock_projects:
        if project["id"] == project_id:
            return Project(**project)
    return None

def create_new_project(project_data: ProjectCreate) -> Project:
    new_project = {
        "id": f"PRJ-{str(uuid.uuid4())[:8].upper()}",
        "name": project_data.name,
        "description": project_data.description or "",
        "created": datetime.utcnow().isoformat() + "Z",
        "status": "active"
    }
    mock_projects.append(new_project)
    return Project(**new_project)

def update_project_by_id(project_id: str, project_data: ProjectUpdate) -> Optional[Project]:
    for i, project in enumerate(mock_projects):
        if project["id"] == project_id:
            if project_data.name is not None:
                project["name"] = project_data.name
            if project_data.description is not None:
                project["description"] = project_data.description
            if project_data.status is not None:
                project["status"] = project_data.status
            mock_projects[i] = project
            return Project(**project)
    return None

def delete_project_by_id(project_id: str) -> bool:
    for i, project in enumerate(mock_projects):
        if project["id"] == project_id:
            del mock_projects[i]
            return True
    return False
