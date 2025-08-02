from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from models.files import FileUploadResponse, FileValidationResponse
from services.files_service import save_uploaded_file, validate_file, get_template_file
from typing import List
import os

router = APIRouter()

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), type: str = "cnc"):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    result = await save_uploaded_file(file, type)
    return result

@router.post("/{file_id}/validate", response_model=FileValidationResponse)
async def validate_file_endpoint(file_id: str):
    result = validate_file(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="File not found")
    return result

@router.get("/templates/{template_type}")
async def download_template(template_type: str):
    template_path = get_template_file(template_type)
    if not template_path or not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template not found")
    
    return FileResponse(
        path=template_path,
        filename=f"cnc_template.{template_type}",
        media_type="application/octet-stream"
    )
