from fastapi import APIRouter, HTTPException, status
from services.chat_service import process_chat_message, get_chat_history
import asyncio

router = APIRouter()

@router.post("/send")
async def send_message(message_data: dict):
    await asyncio.sleep(1)
    message = message_data.get("message", "")
    context = message_data.get("context", {})
    response = process_chat_message(message, context)
    return response

@router.get("/history/{session_id}")
async def get_chat_history_endpoint(session_id: str):
    history = get_chat_history(session_id)
    return history
