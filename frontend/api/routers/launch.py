"""REST endpoints for launching and stopping runs."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.run_launcher import RunLauncher

router = APIRouter(prefix="/api/runs", tags=["launch"])


class LaunchRequest(BaseModel):
    config: str  # e.g. "problem50.yaml"


@router.post("/launch")
async def api_launch_run(request: LaunchRequest):
    """Launch a new optimization run."""
    try:
        result = RunLauncher.launch(request.config)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{run_id}/stop")
async def api_stop_run(run_id: str):
    """Stop a running optimization."""
    result = RunLauncher.stop(run_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result
