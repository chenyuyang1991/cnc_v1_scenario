from pydantic import BaseModel
from typing import List, Optional

class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created: str
    status: str = "active"

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

class ProjectsResponse(BaseModel):
    projects: List[Project]
