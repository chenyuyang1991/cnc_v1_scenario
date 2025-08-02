from fastapi import APIRouter, HTTPException, status
from models.chat import ChatMessage, ChatResponse, ChatHistory
from services.chat_service import process_chat_message, get_chat_history
import asyncio

router = APIRouter()

@router.post("/send", response_model=ChatResponse)
async def send_message(message_data: ChatMessage):
    await asyncio.sleep(1)
    response = process_chat_message(message_data.message, message_data.context)
    return response

@router.get("/history/{session_id}", response_model=ChatHistory)
async def get_chat_history_endpoint(session_id: str):
    history = get_chat_history(session_id)
    return history
