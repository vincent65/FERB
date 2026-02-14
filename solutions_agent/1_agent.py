from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_seed_solution():
    path = Path(__file__).resolve().parents[1] / "solutions_triton" / "1_triton.py"
    spec = importlib.util.spec_from_file_location("seed_solution_1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load seed solution from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solution


solution = _load_seed_solution()
