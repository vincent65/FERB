"""WebSocket handler for real-time run updates."""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.file_watcher import RunFileWatcher

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections per run."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, run_id: str, websocket: WebSocket):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/ws/runs/{run_id}")
async def websocket_run(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for streaming run updates."""
    await manager.connect(run_id, websocket)

    watcher = RunFileWatcher(run_id)

    try:
        async for event in watcher.watch():
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        watcher.stop()
        manager.disconnect(run_id, websocket)
