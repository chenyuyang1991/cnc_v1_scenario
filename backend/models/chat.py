from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    id: str
    content: str
    type: str
    showFileUpload: bool = False
    showConfig: bool = False
    showResults: bool = False
    timestamp: str

class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatResponse]
