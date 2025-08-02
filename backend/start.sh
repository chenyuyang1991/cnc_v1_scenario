#!/bin/bash

echo "🚀 启动 CNC AI 优化器后端服务..."

# 检查是否安装了 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安装，请先安装 Python 3.9 或更高版本"
    exit 1
fi

# 检查 Python 版本
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 版本过低，需要 3.9 或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python 版本检查通过: $PYTHON_VERSION"

# 检查是否安装了 uv
if command -v uv &> /dev/null; then
    echo "📦 使用 uv 管理依赖..."
    
    # 同步依赖
    echo "🔄 同步依赖..."
    uv sync
    
    # 检查环境变量文件
    if [ ! -f ".env" ]; then
        echo "📝 创建环境变量文件..."
        cp env.example .env
        echo "⚠️  请编辑 .env 文件配置您的环境变量"
    fi
    
    # 启动服务器
    echo "🌐 启动开发服务器..."
    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
    
else
    echo "📦 使用 pip 管理依赖..."
    
    # 检查虚拟环境
    if [ ! -d ".venv" ]; then
        echo "🔧 创建虚拟环境..."
        python3 -m venv .venv
    fi
    
    # 激活虚拟环境
    echo "🔧 激活虚拟环境..."
    source .venv/bin/activate
    
    # 安装依赖
    echo "📦 安装依赖..."
    pip install -r requirements.txt
    
    # 检查环境变量文件
    if [ ! -f ".env" ]; then
        echo "📝 创建环境变量文件..."
        cp env.example .env
        echo "⚠️  请编辑 .env 文件配置您的环境变量"
    fi
    
    # 启动服务器
    echo "🌐 启动开发服务器..."
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
fi 