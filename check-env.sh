#!/bin/bash

echo "🔍 CNC AI 优化器 - 环境检查脚本"
echo "=================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查结果
PASS=0
FAIL=0

# 函数：检查命令是否存在
check_command() {
    local cmd=$1
    local name=$2
    local required=$3
    
    if command -v $cmd &> /dev/null; then
        local version=$($cmd --version 2>/dev/null | head -n1)
        echo -e "${GREEN}✅ $name${NC}: $version"
        ((PASS++))
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}❌ $name${NC}: 未安装 (必需)"
            ((FAIL++))
        else
            echo -e "${YELLOW}⚠️  $name${NC}: 未安装 (可选)"
        fi
    fi
}

# 函数：检查 Python 版本
check_python_version() {
    if command -v python3 &> /dev/null; then
        local version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local major=$(echo $version | cut -d. -f1)
        local minor=$(echo $version | cut -d. -f2)
        
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            echo -e "${GREEN}✅ Python 3${NC}: $version"
            ((PASS++))
        else
            echo -e "${RED}❌ Python 3${NC}: $version (需要 3.9 或更高版本)"
            ((FAIL++))
        fi
    else
        echo -e "${RED}❌ Python 3${NC}: 未安装"
        ((FAIL++))
    fi
}

# 函数：检查 Node.js 版本
check_node_version() {
    if command -v node &> /dev/null; then
        local version=$(node --version)
        local major=$(echo $version | cut -c2- | cut -d. -f1)
        
        if [ "$major" -ge 18 ]; then
            echo -e "${GREEN}✅ Node.js${NC}: $version"
            ((PASS++))
        else
            echo -e "${RED}❌ Node.js${NC}: $version (需要 18 或更高版本)"
            ((FAIL++))
        fi
    else
        echo -e "${RED}❌ Node.js${NC}: 未安装"
        ((FAIL++))
    fi
}

# 函数：检查目录结构
check_project_structure() {
    echo -e "\n${BLUE}📁 项目结构检查${NC}"
    
    local dirs=("frontend" "backend" "frontend/src" "backend/app" "backend/routers")
    local files=("frontend/package.json" "backend/pyproject.toml" "backend/main.py" "README.md")
    
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            echo -e "${GREEN}✅ 目录${NC}: $dir"
            ((PASS++))
        else
            echo -e "${RED}❌ 目录${NC}: $dir"
            ((FAIL++))
        fi
    done
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            echo -e "${GREEN}✅ 文件${NC}: $file"
            ((PASS++))
        else
            echo -e "${RED}❌ 文件${NC}: $file"
            ((FAIL++))
        fi
    done
}

# 函数：检查端口占用
check_ports() {
    echo -e "\n${BLUE}🔌 端口占用检查${NC}"
    
    local ports=("8000" "5173" "4173")
    local services=("后端" "前端" "前端预览")
    
    for i in "${!ports[@]}"; do
        local port=${ports[$i]}
        local service=${services[$i]}
        
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo -e "${YELLOW}⚠️  端口${NC}: $port ($service) 已被占用"
        else
            echo -e "${GREEN}✅ 端口${NC}: $port ($service) 可用"
            ((PASS++))
        fi
    done
}

# 函数：检查依赖安装
check_dependencies() {
    echo -e "\n${BLUE}📦 依赖检查${NC}"
    
    # 检查前端依赖
    if [ -d "frontend/node_modules" ]; then
        echo -e "${GREEN}✅ 前端依赖${NC}: 已安装"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠️  前端依赖${NC}: 未安装 (运行 'cd frontend && npm install')"
    fi
    
    # 检查后端依赖
    if [ -d "backend/.venv" ]; then
        echo -e "${GREEN}✅ 后端虚拟环境${NC}: 已创建"
        ((PASS++))
    else
        echo -e "${YELLOW}⚠️  后端虚拟环境${NC}: 未创建 (运行 'cd backend && uv sync')"
    fi
}

# 开始检查
echo -e "${BLUE}🔧 系统环境检查${NC}"
check_command "git" "Git" "true"
check_python_version
check_command "uv" "uv (Python 包管理器)" "false"
check_command "pip" "pip" "false"
check_node_version
check_command "npm" "npm" "true"

# 检查项目结构
check_project_structure

# 检查端口占用
check_ports

# 检查依赖安装
check_dependencies

# 显示结果
echo -e "\n${BLUE}📊 检查结果${NC}"
echo -e "${GREEN}✅ 通过: $PASS${NC}"
echo -e "${RED}❌ 失败: $FAIL${NC}"

if [ $FAIL -eq 0 ]; then
    echo -e "\n${GREEN}🎉 环境检查通过！可以开始开发了。${NC}"
    echo -e "${BLUE}💡 运行 './start.sh' 启动所有服务${NC}"
else
    echo -e "\n${RED}⚠️  请先解决上述问题再继续。${NC}"
    echo -e "${BLUE}💡 查看 README.md 了解详细的安装说明${NC}"
fi

echo "==================================" 