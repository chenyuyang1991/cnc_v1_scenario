from fastapi import APIRouter
from app.api.v1.endpoints import auth, projects, scenarios, simulations, optimization, files, chat, config

api_router = APIRouter()

# 包含所有端點路由
api_router.include_router(auth.router, prefix="/auth", tags=["認證"])
api_router.include_router(projects.router, prefix="/projects", tags=["專案"])
api_router.include_router(scenarios.router, prefix="/scenarios", tags=["場景"])
api_router.include_router(simulations.router, prefix="/simulations", tags=["模擬"])
api_router.include_router(optimization.router, prefix="/optimization", tags=["優化"])
api_router.include_router(files.router, prefix="/files", tags=["檔案"])
api_router.include_router(chat.router, prefix="/chat", tags=["聊天"])
api_router.include_router(config.router, prefix="/config", tags=["配置"]) 