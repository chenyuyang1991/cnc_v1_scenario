#!/bin/bash

echo "🚀 CNC AI 优化器 - 一键启动脚本"
echo "=================================="

# 检查是否在正确的目录
if [ ! -d "frontend" ] || [ ! -d "backend" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 函数：启动后端
start_backend() {
    echo "🔧 启动后端服务..."
    cd backend
    
    # 检查是否有启动脚本
    if [ -f "start.sh" ]; then
        chmod +x start.sh
        ./start.sh &
    else
        echo "❌ 后端启动脚本不存在"
        exit 1
    fi
    
    cd ..
    echo "✅ 后端服务启动中..."
}

# 函数：启动前端
start_frontend() {
    echo "🎨 启动前端服务..."
    cd frontend
    
    # 检查是否有启动脚本
    if [ -f "start.sh" ]; then
        chmod +x start.sh
        ./start.sh &
    else
        echo "📦 安装前端依赖..."
        npm install
        
        echo "🌐 启动前端开发服务器..."
        npm run dev &
    fi
    
    cd ..
    echo "✅ 前端服务启动中..."
}

# 函数：检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  端口 $port 已被占用"
        return 1
    fi
    return 0
}

# 检查端口
echo "🔍 检查端口占用情况..."
if ! check_port 8000; then
    echo "❌ 后端端口 8000 被占用，请先停止占用该端口的服务"
    exit 1
fi

if ! check_port 5173; then
    echo "❌ 前端端口 5173 被占用，请先停止占用该端口的服务"
    exit 1
fi

echo "✅ 端口检查通过"

# 启动服务
start_backend
sleep 3  # 等待后端启动
start_frontend

echo ""
echo "🎉 服务启动完成！"
echo "=================================="
echo "📱 前端地址: http://localhost:5173"
echo "🔧 后端地址: http://localhost:8000"
echo "📚 API 文档: http://localhost:8000/docs"
echo "💚 健康检查: http://localhost:8000/health"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo ""

# 等待用户中断
wait 