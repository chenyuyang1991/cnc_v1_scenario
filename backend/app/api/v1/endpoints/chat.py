from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db

router = APIRouter()


@router.post("/send")
async def send_message(message: dict, db: Session = Depends(get_db)):
    """發送聊天消息"""
    user_message = message.get("message", "").lower()
    
    # 簡單的 AI 回應邏輯
    if "你好" in user_message or "hello" in user_message:
        response = "您好！我是 CNC AI 優化助手，很高興為您服務。"
    elif "優化" in user_message:
        response = "我可以協助您進行 CNC 程式優化。請上傳您的檔案或選擇配置。"
    elif "模擬" in user_message:
        response = "模擬功能可以幫助您預測加工結果。請配置您的參數。"
    else:
        response = "我了解您的需求。請告訴我您想要進行什麼操作？"
    
    return {
        "id": "msg_001",
        "content": response,
        "type": "assistant",
        "timestamp": "2024-01-15T10:30:00Z",
        "showFileUpload": "檔案" in user_message,
        "showConfig": "配置" in user_message or "參數" in user_message,
        "showResults": False,
        "metadata": {
            "suggestions": ["上傳檔案", "配置參數", "運行優化"],
            "actions": ["upload", "configure", "optimize"]
        }
    }


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str, db: Session = Depends(get_db)):
    """獲取聊天歷史"""
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": "msg_001",
                "type": "assistant",
                "content": "您好！我是 CNC AI 優化助手。",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        ]
    } 