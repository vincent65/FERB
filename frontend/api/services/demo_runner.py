"""Service for running demo simulations using pre-generated data."""

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .run_reader import get_runs_dir, PROJECT_ROOT


DEMO_SOURCE_DIR = get_runs_dir() / "demo_problem4_optimization"


class DemoRunner:
    """Manages demo simulation runs that replay pre-generated iteration data."""

    _active_demos: dict[str, dict] = {}

    @classmethod
    def start_demo(cls, experiment_name: str) -> dict[str, Any]:
        """
        Start a demo simulation by creating a new run directory that starts empty
        and progressively reveals pre-generated iterations.
        """
        if not DEMO_SOURCE_DIR.exists():
            raise FileNotFoundError("Demo source data not found")

        # Read manifest
        manifest_path = DEMO_SOURCE_DIR / "demo_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Demo manifest not found")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Create a new timestamped run directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = experiment_name.replace(" ", "_").lower()
        run_id = f"{timestamp}_{safe_name}"
        run_dir = get_runs_dir() / run_id

        # Create the directory structure
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "snapshots" / "problem_4").mkdir(parents=True, exist_ok=True)
        (run_dir / "llm_outputs").mkdir(parents=True, exist_ok=True)

        # Write initial summary (running state)
        summary = {
            "name": experiment_name,
            "run_dir": str(run_dir),
            "iterations": 0,
            "is_demo": True,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Write run_state.json as running
        state = {
            "status": "running",
            "current_iteration": 0,
            "pid": os.getpid(),
            "is_demo": True,
        }
        with open(run_dir / "run_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Create empty iterations.jsonl
        (run_dir / "iterations.jsonl").touch()

        # Copy bootstrap snapshot
        bootstrap_src = DEMO_SOURCE_DIR / "snapshots" / "problem_4" / "iter_0_bootstrap.py"
        if bootstrap_src.exists():
            shutil.copy2(bootstrap_src, run_dir / "snapshots" / "problem_4" / "iter_0_bootstrap.py")

        # Store demo state
        cls._active_demos[run_id] = {
            "run_dir": str(run_dir),
            "source_dir": str(DEMO_SOURCE_DIR),
            "manifest": manifest,
            "current_iteration": 0,
            "total_iterations": manifest["total_iterations"],
            "started_at": time.time(),
        }

        return {
            "run_id": run_id,
            "status": "demo_started",
            "total_iterations": manifest["total_iterations"],
            "delay_ms": manifest["delay_per_iteration_ms"],
        }

    @classmethod
    def get_next_iteration_data(cls, run_id: str) -> dict[str, Any] | None:
        """Get the next iteration's data to be revealed."""
        demo = cls._active_demos.get(run_id)
        if not demo:
            return None

        current = demo["current_iteration"] + 1
        if current > demo["total_iterations"]:
            return None

        source_dir = Path(demo["source_dir"])
        run_dir = Path(demo["run_dir"])

        # Read the iteration lines from source
        source_iterations = source_dir / "iterations.jsonl"
        all_lines = []
        with open(source_iterations) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("iteration") == current:
                        all_lines.append(line)
                except json.JSONDecodeError:
                    pass

        # Copy snapshot
        snap_src = source_dir / "snapshots" / "problem_4" / f"iter_{current}.py"
        if snap_src.exists():
            shutil.copy2(snap_src, run_dir / "snapshots" / "problem_4" / f"iter_{current}.py")

        # Copy LLM output
        llm_src = source_dir / "llm_outputs" / f"iter_{current}_problem_4.json"
        if llm_src.exists():
            shutil.copy2(llm_src, run_dir / "llm_outputs" / f"iter_{current}_problem_4.json")

        # Append iteration lines
        with open(run_dir / "iterations.jsonl", "a") as f:
            for line in all_lines:
                f.write(line + "\n")

        # Update state
        demo["current_iteration"] = current
        state = {
            "status": "running" if current < demo["total_iterations"] else "complete",
            "current_iteration": current,
            "pid": os.getpid(),
            "is_demo": True,
        }
        with open(run_dir / "run_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Update summary
        summary = {
            "name": demo.get("name", "Demo Run"),
            "run_dir": str(run_dir),
            "iterations": current,
            "is_demo": True,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        is_complete = current >= demo["total_iterations"]
        if is_complete:
            del cls._active_demos[run_id]

        return {
            "iteration": current,
            "total": demo["total_iterations"],
            "is_complete": is_complete,
            "events": all_lines,
        }

    @classmethod
    def is_demo_active(cls, run_id: str) -> bool:
        return run_id in cls._active_demos

    @classmethod
    def try_restore_demo(cls, run_id: str) -> bool:
        """
        Try to restore demo state from disk if the in-memory state was lost
        (e.g. due to server restart / reload).
        """
        if run_id in cls._active_demos:
            return True

        run_dir = get_runs_dir() / run_id
        state_path = run_dir / "run_state.json"
        if not state_path.exists():
            return False

        try:
            with open(state_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        if not state.get("is_demo"):
            return False

        # It's a demo run – restore state from disk
        if not DEMO_SOURCE_DIR.exists():
            return False

        manifest_path = DEMO_SOURCE_DIR / "demo_manifest.json"
        if not manifest_path.exists():
            return False

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        current_iteration = state.get("current_iteration", 0)
        total_iterations = manifest.get("total_iterations", 10)

        # If already complete, no need to restore
        if current_iteration >= total_iterations:
            return False

        cls._active_demos[run_id] = {
            "run_dir": str(run_dir),
            "source_dir": str(DEMO_SOURCE_DIR),
            "manifest": manifest,
            "current_iteration": current_iteration,
            "total_iterations": total_iterations,
            "started_at": time.time(),
        }
        return True
