from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routers import auth, projects, scenarios, simulations, optimization, files, chat, config

app = FastAPI(
    title="CNC AI Optimizer API",
    description="FastAPI backend for CNC AI Optimizer application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(projects.router, prefix="/projects", tags=["Projects"])
app.include_router(scenarios.router, prefix="/scenarios", tags=["Scenarios"])
app.include_router(simulations.router, prefix="/simulations", tags=["Simulations"])
app.include_router(optimization.router, prefix="/optimization", tags=["Optimization"])
app.include_router(files.router, prefix="/files", tags=["Files"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(config.router, prefix="/config", tags=["Configuration"])

@app.get("/")
async def root():
    return {"message": "CNC AI Optimizer API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
