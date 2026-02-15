"""Service for reading run data from the filesystem."""

import json
import os
import glob
from pathlib import Path
from typing import Any

# Project root - two levels up from this file (frontend/api/services -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_runs_dir() -> Path:
    return PROJECT_ROOT / "runs"


def get_experiments_dir() -> Path:
    return PROJECT_ROOT / "experiments"


def get_reference_dir() -> Path:
    return PROJECT_ROOT / "reference"


def get_logs_dir() -> Path:
    return PROJECT_ROOT / "logs"


def list_runs() -> list[dict[str, Any]]:
    """List all runs with metadata."""
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []

    runs = []
    for entry in sorted(runs_dir.iterdir(), reverse=True):
        if not entry.is_dir():
            continue

        run_id = entry.name
        run_info: dict[str, Any] = {
            "id": run_id,
            "name": run_id,
            "created_at": entry.stat().st_ctime,
            "status": "complete",
            "iterations": 0,
            "best_speedup": 0.0,
            "problems": [],
        }

        # Read summary.json if it exists
        summary_path = entry / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                run_info["name"] = summary.get("name", run_id)
                run_info["iterations"] = summary.get("iterations", 0)
            except (json.JSONDecodeError, IOError):
                pass

        # Check if run_state.json exists (for active runs)
        state_path = entry / "run_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                run_info["status"] = state.get("status", "complete")
                run_info["current_iteration"] = state.get("current_iteration", 0)
            except (json.JSONDecodeError, IOError):
                pass

        # Parse iterations.jsonl for scores and problem info
        iterations_path = entry / "iterations.jsonl"
        if iterations_path.exists():
            try:
                best_speedup = 0.0
                problem_ids = set()
                max_iter = 0
                with open(iterations_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "problem_id" in event:
                            problem_ids.add(event["problem_id"])

                        if "iteration" in event and isinstance(event.get("iteration"), (int, float)):
                            max_iter = max(max_iter, int(event["iteration"]))

                        score = event.get("score")
                        if score is not None and isinstance(score, (int, float)):
                            best_speedup = max(best_speedup, float(score))

                run_info["best_speedup"] = round(best_speedup, 4)
                run_info["problems"] = sorted(problem_ids)
                if max_iter > 0:
                    run_info["iterations"] = max_iter
            except IOError:
                pass

        runs.append(run_info)

    return runs


def get_run_detail(run_id: str) -> dict[str, Any] | None:
    """Get full run detail including all iterations."""
    run_dir = get_runs_dir() / run_id
    if not run_dir.exists():
        return None

    detail: dict[str, Any] = {
        "id": run_id,
        "name": run_id,
        "iterations": [],
        "events": [],
        "problems": [],
        "status": "complete",
    }

    # Read summary
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            detail["name"] = summary.get("name", run_id)
        except (json.JSONDecodeError, IOError):
            pass

    # Read run state
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
            detail["status"] = state.get("status", "complete")
            detail["current_iteration"] = state.get("current_iteration", 0)
            detail["pid"] = state.get("pid")
        except (json.JSONDecodeError, IOError):
            pass

    # Parse iterations.jsonl
    iterations_path = run_dir / "iterations.jsonl"
    if iterations_path.exists():
        try:
            problem_ids = set()
            with open(iterations_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    detail["events"].append(event)
                    if "problem_id" in event:
                        problem_ids.add(event["problem_id"])
            detail["problems"] = sorted(problem_ids)
        except IOError:
            pass

    # For demo runs with no iterations yet, infer problems from snapshot dirs
    if not detail["problems"]:
        snapshots_dir = run_dir / "snapshots"
        if snapshots_dir.exists():
            for d in snapshots_dir.iterdir():
                if d.is_dir() and d.name.startswith("problem_"):
                    try:
                        pid = int(d.name.replace("problem_", ""))
                        detail["problems"].append(pid)
                    except ValueError:
                        pass
            detail["problems"] = sorted(detail["problems"])

    return detail


def get_iterations(run_id: str) -> list[dict[str, Any]]:
    """Get iteration history with scores for a run."""
    run_dir = get_runs_dir() / run_id
    iterations_path = run_dir / "iterations.jsonl"
    if not iterations_path.exists():
        return []

    iterations: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    with open(iterations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            events.append(event)
            iteration = event.get("iteration")
            if iteration is None:
                continue

            if iteration not in iterations:
                iterations[iteration] = {
                    "iteration": iteration,
                    "problems": {},
                    "events": [],
                }

            pid = event.get("problem_id")
            iter_data = iterations[iteration]
            iter_data["events"].append(event)

            if pid is not None and "score" in event:
                iter_data["problems"][pid] = {
                    "problem_id": pid,
                    "score": event.get("score", 0),
                    "eval_failed": event.get("eval_failed", False),
                    "eval_failure_kind": event.get("eval_failure_kind"),
                    "eval_error": event.get("eval_error"),
                    "reference_summary": event.get("reference_summary"),
                    "candidate_summary": event.get("candidate_summary"),
                    "reference_feedback": event.get("reference_feedback"),
                    "candidate_feedback": event.get("candidate_feedback"),
                }

    result = []
    for iter_num in sorted(iterations.keys()):
        it = iterations[iter_num]
        # Compute aggregate score across problems
        scores = [p["score"] for p in it["problems"].values() if p["score"] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Determine status
        any_failed = any(p.get("eval_failed") for p in it["problems"].values())
        has_rollback = any(e.get("event") == "rollback_after_eval_failure" for e in it["events"])
        has_correctness_fix = any(e.get("event") == "correctness_fix_cycle" for e in it["events"])

        status = "success"
        if has_rollback:
            status = "rollback"
        elif any_failed:
            status = "failed"
        elif has_correctness_fix:
            status = "correctness_fix"

        result.append({
            "iteration": iter_num,
            "score": round(avg_score, 4),
            "status": status,
            "problems": it["problems"],
        })

    return result


def get_snapshots(run_id: str, problem_id: int) -> list[dict[str, Any]]:
    """List all code snapshots for a problem in a run."""
    run_dir = get_runs_dir() / run_id
    snapshots_dir = run_dir / "snapshots" / f"problem_{problem_id}"
    if not snapshots_dir.exists():
        return []

    snapshots = []
    for f in sorted(snapshots_dir.iterdir()):
        if not f.name.endswith(".py"):
            continue
        name = f.name
        # Parse iteration number from filename
        iter_num = 0
        if "bootstrap" in name:
            iter_num = 0
        else:
            try:
                iter_num = int(name.replace("iter_", "").replace(".py", ""))
            except ValueError:
                pass

        snapshots.append({
            "filename": name,
            "iteration": iter_num,
            "is_bootstrap": "bootstrap" in name,
            "path": str(f),
        })

    return snapshots


def get_snapshot_code(run_id: str, problem_id: int, iteration: int) -> str | None:
    """Get the code for a specific snapshot."""
    run_dir = get_runs_dir() / run_id
    snapshots_dir = run_dir / "snapshots" / f"problem_{problem_id}"
    if not snapshots_dir.exists():
        return None

    # Try exact iteration file
    candidates = [
        snapshots_dir / f"iter_{iteration}.py",
        snapshots_dir / f"iter_{iteration}_bootstrap.py",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text()
    return None


def get_llm_output(run_id: str, problem_id: int, iteration: int) -> dict[str, Any] | None:
    """Get LLM reasoning output for a specific iteration."""
    run_dir = get_runs_dir() / run_id
    llm_path = run_dir / "llm_outputs" / f"iter_{iteration}_problem_{problem_id}.json"
    if not llm_path.exists():
        return None

    try:
        with open(llm_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_reference_code(problem_id: int) -> str | None:
    """Get the reference implementation code for a problem."""
    ref_path = get_reference_dir() / f"{problem_id}.py"
    if not ref_path.exists():
        return None
    return ref_path.read_text()


def get_trace(run_id: str, problem_id: int, iteration: int, rank: int, backend: str = "agent") -> dict | None:
    """Get Chrome trace JSON for a specific run/iteration/rank."""
    candidates = [
        get_runs_dir() / run_id / "traces" / f"problem_{problem_id}" / f"iter_{iteration}" / backend / f"trace_candidate_rank{rank}.json",
        get_runs_dir() / run_id / "traces" / f"problem_{problem_id}" / f"iter_{iteration}" / backend / f"trace_reference_rank{rank}.json",
        get_logs_dir() / f"problem_{problem_id}" / backend / f"trace_candidate_rank{rank}.json",
        get_logs_dir() / f"problem_{problem_id}" / backend / f"trace_reference_rank{rank}.json",
    ]
    for trace_path in candidates:
        if trace_path.exists():
            try:
                with open(trace_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    return None


def list_experiments() -> list[dict[str, Any]]:
    """List available experiment YAML configs."""
    import yaml

    exp_dir = get_experiments_dir()
    if not exp_dir.exists():
        return []

    experiments = []
    for f in sorted(exp_dir.iterdir()):
        if not f.name.endswith(".yaml") and not f.name.endswith(".yml"):
            continue

        exp_info: dict[str, Any] = {
            "filename": f.name,
            "path": str(f),
        }

        try:
            with open(f) as fh:
                config = yaml.safe_load(fh)
            exp_info["name"] = config.get("name", f.stem)
            exp_info["max_iterations"] = config.get("max_iterations")
            exp_info["problems"] = config.get("problems", [])
            exp_info["early_stop_speedup"] = config.get("early_stop_speedup")
        except Exception:
            exp_info["name"] = f.stem

        experiments.append(exp_info)

    return experiments


def get_experiment_detail(filename: str) -> dict[str, Any] | None:
    """Get full experiment config as a dictionary."""
    import yaml

    exp_dir = get_experiments_dir()
    filepath = exp_dir / filename
    if not filepath.exists():
        return None

    try:
        with open(filepath) as f:
            config = yaml.safe_load(f)
        config["filename"] = filename
        config["path"] = str(filepath)
        return config
    except Exception:
        return None


def is_demo_run(run_id: str) -> bool:
    """Check if a run is a demo run."""
    run_dir = get_runs_dir() / run_id
    manifest_path = run_dir / "demo_manifest.json"
    if manifest_path.exists():
        return True
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            return summary.get("is_demo", False)
        except (json.JSONDecodeError, IOError):
            pass
    state_path = run_dir / "run_state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
            return state.get("is_demo", False)
        except (json.JSONDecodeError, IOError):
            pass
    return False
