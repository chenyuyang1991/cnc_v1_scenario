from fastapi import APIRouter, HTTPException, status
from models.projects import Project, ProjectCreate, ProjectUpdate, ProjectsResponse
from services.projects_service import get_all_projects, get_project_by_id, create_new_project, update_project_by_id, delete_project_by_id

router = APIRouter()

@router.get("/", response_model=ProjectsResponse)
async def get_projects():
    projects = get_all_projects()
    return ProjectsResponse(projects=projects)

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    project = get_project_by_id(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.post("/", response_model=Project)
async def create_project(project_data: ProjectCreate):
    project = create_new_project(project_data)
    return project

@router.put("/{project_id}", response_model=Project)
async def update_project(project_id: str, project_data: ProjectUpdate):
    project = update_project_by_id(project_id, project_data)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.delete("/{project_id}")
async def delete_project(project_id: str):
    success = delete_project_by_id(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully"}
