"""FastAPI backend for the EcholoKernel Frontend Dashboard."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import runs, experiments, launch
from .ws.handler import router as ws_router

app = FastAPI(
    title="EcholoKernel API",
    description="Backend API for the Triton Kernel Optimizer dashboard",
    version="1.0.0",
)

# CORS - allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(runs.router)
app.include_router(experiments.router)
app.include_router(launch.router)
app.include_router(ws_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
