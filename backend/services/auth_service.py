from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import yaml
import os
import hashlib

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def load_user_credentials(config_path="../cnc_data/cnc_cred/auth_users.yaml") -> Dict[str, Any]:
    """Load user credentials from YAML file"""
    try:
        if not os.path.exists(config_path):
            config_path = "cnc_data/cnc_cred/auth_users.yaml"
        
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        credentials = {}
        acc_data = data.get("account_name", {})
        for k in acc_data:
            username = k
            hashed_password = acc_data[k].get("password", None)
            access_level = acc_data[k].get("access_level", [])
            credentials[username] = {"password": hashed_password, "access_level": access_level}
        return credentials
    except Exception as e:
        print(f"Error loading user credentials: {str(e)}")
        return {}

def verify_password(plain_password, hashed_password):
    """Verify password - supports both plain text and hashed passwords"""
    if hashed_password == plain_password:
        return True
    
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except:
        try:
            computed_hash = hashlib.sha256(plain_password.encode()).hexdigest()
            return computed_hash == hashed_password
        except:
            return False

def authenticate_user(username: str, password: str):
    """Authenticate user against YAML credentials"""
    valid_users = load_user_credentials()
    
    if username not in valid_users:
        return False
    
    try:
        check_password_res = verify_password(password, valid_users[username]["password"])
        if check_password_res:
            return {
                "id": f"user-{username}",
                "username": username,
                "email": f"{username}@cnc.local",
                "role": "admin" if "admin" in valid_users[username]["access_level"] else "operator",
                "access_level": valid_users[username]["access_level"],
                "created_at": "2024-01-01T00:00:00Z"
            }
        return False
    except Exception as e:
        print(f"Authentication error: {e}")
        return False

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
        username = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None
