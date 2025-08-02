from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class User(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: str

class TokenData(BaseModel):
    username: Optional[str] = None
