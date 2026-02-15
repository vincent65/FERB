"""File watcher service for real-time WebSocket updates."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .run_reader import get_runs_dir


class RunFileWatcher:
    """Watches a run directory for changes and yields new events."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run_dir = get_runs_dir() / run_id
        self._last_iterations_pos = 0
        self._last_llm_files: set[str] = set()
        self._last_snapshot_files: set[str] = set()
        self._running = False

    async def watch(self):
        """Async generator that yields new events as they appear."""
        self._running = True

        # Initialize positions
        iterations_path = self.run_dir / "iterations.jsonl"
        if iterations_path.exists():
            self._last_iterations_pos = iterations_path.stat().st_size

        llm_dir = self.run_dir / "llm_outputs"
        if llm_dir.exists():
            self._last_llm_files = {f.name for f in llm_dir.iterdir() if f.name.endswith(".json")}

        snapshots_dir = self.run_dir / "snapshots"
        if snapshots_dir.exists():
            self._last_snapshot_files = set()
            for problem_dir in snapshots_dir.iterdir():
                if problem_dir.is_dir():
                    for f in problem_dir.iterdir():
                        self._last_snapshot_files.add(str(f.relative_to(snapshots_dir)))

        while self._running:
            try:
                # Check for new lines in iterations.jsonl
                if iterations_path.exists():
                    current_size = iterations_path.stat().st_size
                    if current_size > self._last_iterations_pos:
                        with open(iterations_path) as f:
                            f.seek(self._last_iterations_pos)
                            new_content = f.read()
                        self._last_iterations_pos = current_size

                        for line in new_content.strip().split("\n"):
                            if line.strip():
                                try:
                                    event = json.loads(line)
                                    yield {"type": "iteration_event", "data": event}
                                except json.JSONDecodeError:
                                    pass

                # Check for new LLM output files
                if llm_dir.exists():
                    current_llm_files = {f.name for f in llm_dir.iterdir() if f.name.endswith(".json")}
                    new_llm_files = current_llm_files - self._last_llm_files
                    for fname in sorted(new_llm_files):
                        try:
                            with open(llm_dir / fname) as f:
                                llm_data = json.load(f)
                            yield {"type": "llm_proposal", "filename": fname, "data": llm_data}
                        except (json.JSONDecodeError, IOError):
                            pass
                    self._last_llm_files = current_llm_files

                # Check for new snapshot files
                if snapshots_dir.exists():
                    current_snapshot_files = set()
                    for problem_dir in snapshots_dir.iterdir():
                        if problem_dir.is_dir():
                            for f in problem_dir.iterdir():
                                current_snapshot_files.add(str(f.relative_to(snapshots_dir)))

                    new_snapshots = current_snapshot_files - self._last_snapshot_files
                    for snap_path in sorted(new_snapshots):
                        full_path = snapshots_dir / snap_path
                        try:
                            code = full_path.read_text()
                            yield {
                                "type": "snapshot",
                                "path": snap_path,
                                "code": code,
                            }
                        except IOError:
                            pass
                    self._last_snapshot_files = current_snapshot_files

                # Check run state
                state_path = self.run_dir / "run_state.json"
                if state_path.exists():
                    try:
                        with open(state_path) as f:
                            state = json.load(f)
                        if state.get("status") == "complete":
                            yield {"type": "run_complete", "data": state}
                            self._running = False
                    except (json.JSONDecodeError, IOError):
                        pass

            except Exception as e:
                yield {"type": "error", "message": str(e)}

            await asyncio.sleep(1.0)  # Poll every 1 second

    def stop(self):
        self._running = False
