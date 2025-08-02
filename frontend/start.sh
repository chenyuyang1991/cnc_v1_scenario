#!/bin/bash

echo "🚀 啟動 CNC AI 優化器前端服務..."

# 檢查是否安裝了 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安裝，請先安裝 Node.js"
    exit 1
fi

# 檢查是否安裝了 npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安裝，請先安裝 npm"
    exit 1
fi

# 安裝依賴
echo "📦 安裝依賴..."
npm install

# 啟動開發服務器
echo "🌐 啟動開發服務器..."
npm run dev 