#!/bin/bash

set -e

# generate hashed_pw.txt
python save_password.py $1

# 讀取 bcrypt hash（去除換行）
# Read base64 encoded hash (safe for text processing)
replacement=$(<hashed_pw.txt)

# Since it's base64, no special character escaping needed
# Base64 only contains A-Z, a-z, 0-9, +, /, = which are safe for sed
sed "s/<password>/${replacement}/g" auth_template.py > auth.py

echo "Authentication file generated: auth.py"