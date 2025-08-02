from pydantic import BaseModel
from typing import List, Optional

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    type: str
    status: str
    uploaded_at: str

class FileValidationResponse(BaseModel):
    file_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: dict
