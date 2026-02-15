"""REST endpoints for experiment configs."""

from fastapi import APIRouter

from ..services.run_reader import list_experiments

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("")
async def api_list_experiments():
    """List available experiment YAML configs."""
    return list_experiments()
