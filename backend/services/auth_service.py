from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

mock_users = {
    "admin": {
        "id": "user-001",
        "username": "admin",
        "email": "admin@cnc-optimizer.com",
        "role": "administrator",
        "created_at": "2024-01-01T00:00:00Z",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p02T/oXyuSrXiSWBdFHm.9q2"
    },
    "operator": {
        "id": "user-002", 
        "username": "operator",
        "email": "operator@cnc-optimizer.com",
        "role": "operator",
        "created_at": "2024-01-01T00:00:00Z",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p02T/oXyuSrXiSWBdFHm.9q2"
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = mock_users.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    
    user_data = user.copy()
    del user_data["hashed_password"]
    return user_data

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None
