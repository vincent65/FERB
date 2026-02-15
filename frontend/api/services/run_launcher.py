"""Service for launching and managing optimization runs."""

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

from .run_reader import PROJECT_ROOT, get_runs_dir


class RunLauncher:
    """Manages launching and stopping optimization runs."""

    _active_processes: dict[str, subprocess.Popen] = {}

    @classmethod
    def launch(cls, config_path: str) -> dict[str, Any]:
        """Launch a new optimization run with the given experiment config."""
        full_config_path = PROJECT_ROOT / "experiments" / config_path
        if not full_config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Launch the optimizer as a subprocess
        cmd = [
            "python", str(PROJECT_ROOT / "run_experiment.py"),
            "--config", str(full_config_path),
        ]

        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # We'll track the process. The run_id will be determined by the optimizer
        # (it creates a timestamped directory). We return the PID for now.
        return {
            "pid": process.pid,
            "config": config_path,
            "status": "launched",
        }

    @classmethod
    def stop(cls, run_id: str) -> dict[str, Any]:
        """Stop a running optimization by its run_id."""
        # Check for run_state.json which has the PID
        run_dir = get_runs_dir() / run_id
        state_path = run_dir / "run_state.json"

        pid = None
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                pid = state.get("pid")
            except (json.JSONDecodeError, IOError):
                pass

        if pid is None:
            return {"status": "error", "message": "No active process found for this run"}

        try:
            os.kill(pid, signal.SIGTERM)
            return {"status": "stopped", "run_id": run_id, "pid": pid}
        except ProcessLookupError:
            return {"status": "already_stopped", "run_id": run_id}
        except PermissionError:
            return {"status": "error", "message": f"Permission denied to stop process {pid}"}
