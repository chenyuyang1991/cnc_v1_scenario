# CNC AI 优化器 - 完整设置指南

## 🚀 快速开始

### 一键启动
```bash
# 检查环境
./check-env.sh

# 启动所有服务
./start.sh

# 停止所有服务
./stop.sh
```

### 常用命令
```bash
# 前端开发
cd frontend
npm run dev          # 启动开发服务器
npm run build        # 构建生产版本
npm run preview      # 预览生产版本

# 后端开发
cd backend
./start.sh           # 启动后端服务
uv run uvicorn main:app --reload  # 手动启动

# 环境检查
./check-env.sh       # 检查开发环境
```

---

## 📋 项目概述

CNC AI 优化器是一个基于 Vue.js 前端和 FastAPI 后端的全栈应用程序，用于 CNC 加工优化。

### 技术栈
- **前端**: Vue.js 3 + Vite + Tailwind CSS
- **后端**: FastAPI + SQLAlchemy + Python 3.9+
- **数据库**: SQLite (开发) / PostgreSQL (生产)
- **包管理**: npm (前端) / uv (后端)

## 🛠️ 环境要求

### 系统要求
- **操作系统**: macOS, Linux, Windows
- **Node.js**: 18.0.0 或更高版本
- **Python**: 3.9 或更高版本
- **内存**: 至少 4GB RAM
- **磁盘空间**: 至少 2GB 可用空间

### 必需软件
1. **Node.js & npm**
   - 下载地址: https://nodejs.org/
   - 验证安装: `node --version` 和 `npm --version`

2. **Python 3.9+**
   - 下载地址: https://www.python.org/downloads/
   - 验证安装: `python --version`

3. **uv (Python 包管理器)** - 推荐
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## 🔧 详细设置

### 1. 克隆项目
```bash
git clone <repository-url>
cd cnc_v1_scenario
```

### 2. 后端设置

#### 方法一：使用 uv (推荐)
```bash
cd backend

# 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 设置环境变量
cp env.example .env
# 编辑 .env 文件，根据需要修改配置
```

#### 方法二：使用 pip
```bash
cd backend

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
cp env.example .env
# 编辑 .env 文件，根据需要修改配置
```

#### 环境变量配置
编辑 `backend/.env` 文件：
```env
# 数据库配置 (开发环境使用 SQLite)
DATABASE_URL=sqlite:///./cnc_optimizer.db

# 安全配置
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 应用配置
DEBUG=true
ENVIRONMENT=development
API_V1_STR=/api/v1
PROJECT_NAME=CNC AI 优化器

# 文件上传配置
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760

# CORS 配置
BACKEND_CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"]

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### 3. 前端设置
```bash
cd frontend

# 安装依赖
npm install

# 可选：检查依赖更新
npm audit fix
```

## 🏃‍♂️ 运行应用

### 启动后端服务
```bash
cd backend

# 激活虚拟环境 (如果使用 pip)
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 启动开发服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

后端服务将在以下地址运行：
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **API 根路径**: http://localhost:8000/

### 启动前端服务
```bash
cd frontend

# 方法一：使用 npm
npm run dev

# 方法二：使用提供的脚本
chmod +x start.sh
./start.sh
```

前端服务将在以下地址运行：
- **开发服务器**: http://localhost:5173
- **预览模式**: http://localhost:4173 (构建后)

## 📁 项目结构

```
cnc_v1_scenario/
├── backend/                 # 后端代码
│   ├── app/                # 应用核心代码
│   ├── routers/            # API 路由
│   ├── models/             # 数据模型
│   ├── services/           # 业务逻辑
│   ├── static/             # 静态文件
│   ├── uploads/            # 上传文件目录
│   ├── logs/               # 日志文件
│   ├── main.py             # 应用入口
│   ├── pyproject.toml      # Python 项目配置
│   ├── requirements.txt    # Python 依赖
│   └── env.example         # 环境变量示例
├── frontend/               # 前端代码
│   ├── src/                # 源代码
│   ├── public/             # 公共资源
│   ├── package.json        # Node.js 项目配置
│   ├── vite.config.js      # Vite 配置
│   ├── tailwind.config.js  # Tailwind CSS 配置
│   └── start.sh            # 启动脚本
└── README.md               # 项目文档
```

## 🔧 开发工具

### 后端开发
```bash
cd backend

# 代码格式化
uv run black .
uv run isort .

# 代码检查
uv run flake8 .
uv run mypy .

# 运行测试
uv run pytest
```

### 前端开发
```bash
cd frontend

# 代码格式化
npm run format

# 代码检查
npm run lint

# 构建生产版本
npm run build

# 预览生产版本
npm run preview
```

## 🌐 API 端点

### 认证相关
- `POST /auth/login` - 用户登录
- `POST /auth/register` - 用户注册
- `POST /auth/refresh` - 刷新令牌

### 项目管理
- `GET /projects` - 获取项目列表
- `POST /projects` - 创建新项目
- `GET /projects/{id}` - 获取项目详情
- `PUT /projects/{id}` - 更新项目
- `DELETE /projects/{id}` - 删除项目

### 场景管理
- `GET /scenarios` - 获取场景列表
- `POST /scenarios` - 创建新场景
- `GET /scenarios/{id}` - 获取场景详情

### 模拟和优化
- `POST /simulations` - 运行模拟
- `POST /optimization` - 运行优化
- `GET /simulations/{id}` - 获取模拟结果

### 文件管理
- `POST /files/upload` - 上传文件
- `GET /files/{filename}` - 下载文件

### 聊天功能
- `POST /chat/message` - 发送聊天消息
- `GET /chat/history` - 获取聊天历史

## 🐛 故障排除

### 常见问题

#### 1. 端口被占用
```bash
# 查找占用端口的进程
lsof -i :8000  # 后端端口
lsof -i :5173  # 前端端口

# 杀死进程
kill -9 <PID>
```

#### 2. 依赖安装失败
```bash
# 清除缓存
npm cache clean --force  # 前端
uv cache clean          # 后端

# 重新安装
rm -rf node_modules package-lock.json
npm install
```

#### 3. 数据库连接问题
- 检查 `.env` 文件中的 `DATABASE_URL`
- 确保数据库服务正在运行
- 检查网络连接

#### 4. CORS 错误
- 确保前端和后端的端口配置正确
- 检查 `BACKEND_CORS_ORIGINS` 设置

### 日志查看
```bash
# 后端日志
tail -f backend/logs/app.log

# 前端开发服务器日志
# 在终端中查看 npm run dev 的输出
```

## 📦 部署

### 生产环境部署

#### 后端部署
```bash
cd backend

# 安装生产依赖
uv sync --production

# 设置生产环境变量
export ENVIRONMENT=production
export DEBUG=false

# 启动生产服务器
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 前端部署
```bash
cd frontend

# 构建生产版本
npm run build

# 部署 dist/ 目录到 Web 服务器
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 支持

如果您遇到问题或有疑问，请：
1. 查看本文档的故障排除部分
2. 检查项目的 Issues 页面
3. 联系开发团队

---

**祝您使用愉快！** 🎉
