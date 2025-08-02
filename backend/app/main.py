from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.database import engine
from app.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時執行
    print("🚀 啟動 CNC AI 優化器後端服務...")
    
    # 創建數據庫表
    Base.metadata.create_all(bind=engine)
    print("✅ 數據庫表創建完成")
    
    yield
    
    # 關閉時執行
    print("🛑 關閉 CNC AI 優化器後端服務...")


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description="CNC AI 優化器後端 API 服務",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # 設置 CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 包含 API 路由
    application.include_router(api_router, prefix=settings.API_V1_STR)

    # 靜態檔案服務
    application.mount("/static", StaticFiles(directory="static"), name="static")

    return application


app = create_application()


@app.get("/")
async def root():
    return {
        "message": "CNC AI 優化器後端服務",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cnc-ai-optimizer-backend"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    ) 