import sys
from auth_template import save_password, check_password

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python save_password.py <password>")
        sys.exit(1)

    password = sys.argv[1]
    save_password(password)
    print("密碼已加密儲存至 hashed_pw.txt")

    if len(sys.argv) >= 3 and sys.argv[2] == "check":
        user_input = input("請輸入密碼：")
        if check_password(user_input):
            print("✅ 密碼正確，驗證通過。")
        else:
            print("❌ 密碼錯誤，驗證失敗。")
