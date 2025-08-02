from pydantic import BaseModel
from typing import Optional, List


class Token(BaseModel):
    success: bool
    token: str
    user: dict
    platform: str


class UserBase(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    is_superuser: bool

    class Config:
        from_attributes = True 