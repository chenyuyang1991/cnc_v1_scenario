from fastapi import APIRouter, HTTPException, status
from models.auth import LoginRequest, LoginResponse, User
from services.auth_service import authenticate_user, create_access_token, verify_token
from datetime import datetime

router = APIRouter()

@router.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=user
    )

@router.post("/logout")
async def logout():
    return {"message": "Successfully logged out"}

@router.get("/verify")
async def verify():
    return {"status": "valid", "message": "Token is valid"}
