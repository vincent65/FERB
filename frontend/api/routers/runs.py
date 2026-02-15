"""REST endpoints for runs, iterations, snapshots, and LLM outputs."""

from fastapi import APIRouter, HTTPException

from ..services.run_reader import (
    get_iterations,
    get_llm_output,
    get_reference_code,
    get_run_detail,
    get_snapshot_code,
    get_snapshots,
    get_trace,
    list_runs,
    is_demo_run,
)
from ..services.demo_runner import DemoRunner

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("")
async def api_list_runs():
    """List all optimization runs."""
    return list_runs()


@router.get("/{run_id}")
async def api_get_run(run_id: str):
    """Get full run detail."""
    detail = get_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@router.get("/{run_id}/iterations")
async def api_get_iterations(run_id: str):
    """Get iteration history with scores."""
    return get_iterations(run_id)


@router.get("/{run_id}/problems/{problem_id}/snapshots")
async def api_get_snapshots(run_id: str, problem_id: int):
    """List all code snapshots for a problem."""
    return get_snapshots(run_id, problem_id)


@router.get("/{run_id}/problems/{problem_id}/snapshots/{iteration}")
async def api_get_snapshot_code(run_id: str, problem_id: int, iteration: int):
    """Get the code for a specific snapshot."""
    code = get_snapshot_code(run_id, problem_id, iteration)
    if code is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {"iteration": iteration, "problem_id": problem_id, "code": code}


@router.get("/{run_id}/problems/{problem_id}/llm-output/{iteration}")
async def api_get_llm_output(run_id: str, problem_id: int, iteration: int):
    """Get LLM reasoning output for an iteration."""
    output = get_llm_output(run_id, problem_id, iteration)
    if output is None:
        raise HTTPException(status_code=404, detail="LLM output not found")
    return output


@router.get("/{run_id}/problems/{problem_id}/reference")
async def api_get_reference_code(problem_id: int, run_id: str = ""):
    """Get the reference implementation code."""
    code = get_reference_code(problem_id)
    if code is None:
        raise HTTPException(status_code=404, detail="Reference code not found")
    return {"problem_id": problem_id, "code": code}


@router.get("/{run_id}/problems/{problem_id}/traces/{iteration}/{rank}")
async def api_get_trace(run_id: str, problem_id: int, iteration: int, rank: int, backend: str = "agent"):
    """Get Chrome trace JSON."""
    trace = get_trace(run_id, problem_id, iteration, rank, backend)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@router.get("/{run_id}/is-demo")
async def api_is_demo(run_id: str):
    """Check if a run is a demo simulation."""
    return {"is_demo": is_demo_run(run_id) or DemoRunner.is_demo_active(run_id)}


@router.post("/{run_id}/demo/advance")
async def api_demo_advance(run_id: str):
    """Advance demo simulation by one iteration."""
    if not DemoRunner.is_demo_active(run_id):
        # Try to restore from disk (e.g. after server restart)
        if not DemoRunner.try_restore_demo(run_id):
            return {"status": "complete", "run_id": run_id, "is_complete": True}
    result = DemoRunner.get_next_iteration_data(run_id)
    if result is None:
        return {"status": "complete", "run_id": run_id, "is_complete": True}
    return result
