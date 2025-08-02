#!/bin/bash

echo "🛑 CNC AI 优化器 - 停止服务脚本"
echo "=================================="

# 函数：停止指定端口的服务
stop_service() {
    local port=$1
    local service_name=$2
    
    echo "🔍 查找 $service_name 服务 (端口: $port)..."
    
    # 查找占用端口的进程
    local pids=$(lsof -ti :$port)
    
    if [ -n "$pids" ]; then
        echo "🛑 停止 $service_name 服务..."
        echo "$pids" | xargs kill -9
        echo "✅ $service_name 服务已停止"
    else
        echo "ℹ️  $service_name 服务未运行"
    fi
}

# 停止后端服务 (端口 8000)
stop_service 8000 "后端"

# 停止前端服务 (端口 5173)
stop_service 5173 "前端"

# 停止前端预览服务 (端口 4173)
stop_service 4173 "前端预览"

echo ""
echo "🎉 所有服务已停止！"
echo "==================================" 