#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modular Triton optimization experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/default.yaml",
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    from agent.config import ExperimentConfig
    from agent.core.optimizer import Optimizer

    repo_root = Path(__file__).resolve().parent
    cfg = ExperimentConfig.from_yaml(repo_root / args.config)
    optimizer = Optimizer(repo_root=repo_root, cfg=cfg)
    run_dir = optimizer.run()
    print(f"Experiment complete: {run_dir}")


if __name__ == "__main__":
    main()
