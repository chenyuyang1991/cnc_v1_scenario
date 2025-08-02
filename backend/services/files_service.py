from fastapi import UploadFile
from models.files import FileUploadResponse, FileValidationResponse
from datetime import datetime
from typing import Optional
import uuid
import os
import aiofiles

UPLOAD_DIR = "uploads"
TEMPLATES_DIR = "static/templates"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

uploaded_files = {}

async def save_uploaded_file(file: UploadFile, file_type: str) -> FileUploadResponse:
    file_id = f"FILE-{str(uuid.uuid4())[:8].upper()}"
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    file_info = {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "type": file_type,
        "status": "uploaded",
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "path": file_path
    }
    
    uploaded_files[file_id] = file_info
    
    return FileUploadResponse(**file_info)

def validate_file(file_id: str) -> Optional[FileValidationResponse]:
    if file_id not in uploaded_files:
        return None
    
    file_info = uploaded_files[file_id]
    
    errors = []
    warnings = []
    
    if file_info["size"] > 10 * 1024 * 1024:
        warnings.append("檔案大小超過 10MB，處理時間可能較長")
    
    if not file_info["filename"].lower().endswith(('.nc', '.cnc', '.gcode', '.txt', '.csv', '.xlsx')):
        errors.append("不支援的檔案格式")
    
    is_valid = len(errors) == 0
    
    metadata = {
        "file_type": "G-Code" if file_info["filename"].lower().endswith(('.nc', '.cnc', '.gcode')) else "Data",
        "estimated_lines": 1500 if is_valid else 0,
        "estimated_processing_time": "2-3 分鐘" if is_valid else "N/A"
    }
    
    return FileValidationResponse(
        file_id=file_id,
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        metadata=metadata
    )

def get_template_file(template_type: str) -> Optional[str]:
    template_files = {
        "excel": os.path.join(TEMPLATES_DIR, "cnc_template.xlsx"),
        "csv": os.path.join(TEMPLATES_DIR, "cnc_template.csv")
    }
    
    template_path = template_files.get(template_type)
    if template_path and not os.path.exists(template_path):
        create_template_file(template_path, template_type)
    
    return template_path

def create_template_file(file_path: str, template_type: str):
    if template_type == "csv":
        content = "Tool,Diameter,Speed,Feed,Depth\nEndmill,6,3000,500,2\nDrill,3,2000,200,5\n"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    elif template_type == "excel":
        content = "Tool,Diameter,Speed,Feed,Depth\nEndmill,6,3000,500,2\nDrill,3,2000,200,5\n"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
