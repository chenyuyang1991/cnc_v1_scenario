
import getpass
import bcrypt
import base64

def save_password(plain_password: str, file_path="hashed_pw.txt"):
    """Generate bcrypt hash and save as base64 string"""
    password_bytes = plain_password.encode('utf-8')
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

    # Convert to base64 for safe storage
    hashed_b64 = base64.b64encode(hashed).decode('ascii')

    with open(file_path, "w", encoding='utf-8') as f:
        f.write(hashed_b64)

# This will be replaced with the actual base64-encoded hash
# Base64 format is safe for text processing across all platforms
encrypted_passwd_b64 = "<password>"

def check_password(input_password: str, file_path=None):
    """Check password against stored bcrypt hash"""
    try:
        if file_path:
            # Load from file
            with open(file_path, "r", encoding='utf-8') as f:
                stored_hash_b64 = f.read().strip()
        else:
            # Use embedded hash
            stored_hash_b64 = encrypted_passwd_b64
        
        # Decode base64 to get original bcrypt hash
        stored_hash = base64.b64decode(stored_hash_b64.encode('ascii'))
        
        # Verify password using bcrypt
        return bcrypt.checkpw(input_password.encode('utf-8'), stored_hash)
    
    except Exception as e:
        print(f"Password verification error: {e}")
        return False

def input_password_and_check():
    """Prompt for password and verify"""
    user_input = getpass.getpass("請輸入密碼：")
    if check_password(user_input):
        print("✅ 密碼正確，驗證通過。")
        return True
    print("❌ 密碼錯誤，驗證失敗。")
    return False