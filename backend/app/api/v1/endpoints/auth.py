from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from app.core.database import get_db
from app.core.config import settings
from app.schemas.auth import Token, UserCreate, User
from app.services.auth_service import authenticate_user, create_access_token, get_current_user

router = APIRouter()


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """用戶登入"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用戶名或密碼錯誤",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "success": True,
        "token": access_token,
        "user": {
            "id": user.id,
            "username": user.username,
            "role": "admin" if user.is_superuser else "user",
            "permissions": ["read", "write", "admin"] if user.is_superuser else ["read", "write"]
        },
        "platform": "v1"  # 默認平台
    }


@router.post("/logout")
async def logout():
    """用戶登出"""
    return {"message": "登出成功"}


@router.get("/verify")
async def verify_token(current_user: User = Depends(get_current_user)):
    """驗證 token"""
    return {
        "valid": True,
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": "admin" if current_user.is_superuser else "user"
        }
    } 