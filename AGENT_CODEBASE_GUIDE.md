# AGENT Codebase Guide

This file is a handoff/reference doc for future agents and conversations.
It summarizes what has been implemented, how the system is wired, and how to run it.

## Project Goal

Build an iterative OpenAI-powered kernel optimizer that improves Triton solutions against reference PyTorch distributed kernels using Modal-based H100 evaluation, correctness checks, timing, and profiling feedback.

## What Was Added In This Implementation Pass

### 1) Distributed worker harness

- Added `scripts/worker.py`
- This is the runtime worker launched by `torchrun` from `run_modal.py`.
- Responsibilities:
  - backend init/finalize (reference/triton/agent/numba_cuda)
  - dynamic module loading (reference and candidate solution)
  - standardized input generation
  - correctness check
  - timing benchmark (CUDA events + barriers)
  - optional `torch.profiler` traces
  - metrics artifact writing (`rank_*.json`, `summary_rank0.json`)

### 2) Input/output standardization + extension hook

- Reused `utils/input_output_tensors.py` as the default input path.
- Updated `create_input_tensor(...)` to support optional override functions.
- Added `utils/problem_specs.py` as an override registry for edge-case problems.

### 3) Agent backend directory

- Added `solutions_agent/` for editable candidate kernels.
- Added seed wrappers:
  - `solutions_agent/1_agent.py`
  - `solutions_agent/2_agent.py`
  - `solutions_agent/10_agent.py`
- These currently import corresponding Triton seeds from `solutions_triton/`.

### 4) Modular optimizer framework

- Added `agent/` package with pluggable components.
- Core orchestrator:
  - `agent/core/optimizer.py`
- Bootstrap stage for unseen problems:
  - `agent/core/bootstrap.py`
- Modal evaluation:
  - `agent/eval/modal_evaluator.py`
- OpenAI integration:
  - `agent/openai_client.py`
- Run logging and rollback:
  - `agent/core/run_log.py`
  - `agent/core/rollback.py`

### 5) Strategy architecture

- Added strategy folders for easy variant testing:
  - `agent/strategies/proposers/`
  - `agent/strategies/memory/`
  - `agent/strategies/scorers/`
- Added a registry factory:
  - `agent/strategies/registry.py`

### 6) Experiment runner + config

- Added `run_experiment.py`
- Added default config:
  - `experiments/default.yaml`
- Added dependencies:
  - `requirements.txt`

## High-Level Architecture

### Single evaluation run

1. `modal run run_modal.py ...`
2. Modal launches 8x H100 and runs `torchrun`
3. `scripts/worker.py` evaluates reference + candidate
4. Artifacts are written to logs and downloaded locally

### Iterative optimization run

1. `python3 run_experiment.py --config experiments/default.yaml`
2. Bootstrap stage ensures a candidate exists in `solutions_agent/`:
   - copy seed from solved backend if available
   - otherwise generate initial candidate from reference + solved examples
3. Evaluator precomputes reference once and caches outputs/timing metadata
4. Evaluator runs candidate jobs and returns structured feedback (correctness, aggregate timings, trace paths, cache flags)
5. Optimizer routes proposal mode:
   - `correctness_fix` when mismatches are detected
   - `perf_opt` when correctness is clean
6. Proposer calls OpenAI to generate patch
7. Patch is applied, logged, and rollback is used on failure
8. Loop repeats for configured iterations

## Important File Map

### Launch and execution

- `run_modal.py` - Modal launcher and local download path handling
- `scripts/worker.py` - distributed worker runtime

### Utilities

- `utils/init_and_finalize_backends.py` - backend setup/teardown
- `utils/input_output_tensors.py` - default input generation + save helpers
- `utils/problem_specs.py` - optional problem-specific input overrides
- `utils/performance.py` - timing/bandwidth utilities

### Problems and kernels

- `reference/` - baseline reference kernels by problem id
- `solutions_triton/` - existing Triton implementations
- `solutions_agent/` - candidate kernels for agent iteration

### Agent framework

- `agent/config.py` - typed experiment config loader
- `agent/openai_client.py` - OpenAI wrapper
- `agent/eval/modal_evaluator.py` - Modal-backed evaluator
- `agent/core/optimizer.py` - optimization loop
- `agent/core/run_log.py` - JSONL run logs
- `agent/core/rollback.py` - backup/rollback
- `agent/strategies/...` - modular proposer/memory/scorer implementations

### Experiment configs

- `experiments/default.yaml` - default run plan

### Prompt and model tuning

- Edit `experiments/default.yaml`:
  - `openai.model` and `openai.temperature` control model selection/sampling.
  - `prompts.patch_system_prompt` controls iterative patching system instruction.
  - `prompts.patch_schema_hint` controls strict patch output-format instructions.
  - `prompts.bootstrap_system_prompt` controls unseen-problem kernel generation behavior.
  - `prompts.performance_patch_template` controls perf optimization prompt.
  - `prompts.correctness_patch_template` controls correctness-fix prompt.
  - `prompts.bootstrap_template` controls bootstrap prompt and supports:
    - `{problem_id}`
    - `{candidate_file}`
    - `{reference_code}`
    - `{context_examples}`
  - Iterative templates support:
    - `{problem_id}`
    - `{candidate_file}`
    - `{memory_summary}`
    - `{eval_feedback}`
    - `{current_code}`

## Commands

### Single backend smoke test (Problem 1 + agent backend)

```bash
modal run run_modal.py --problem 1 --solution agent --m 1024 --n 1024 --dtype float32 --download
```

### Reference outputs then candidate-only generation runs

Generate and persist reference `.pt` outputs once:

```bash
modal run run_modal.py --problem 1 --solution reference --m 1024 --n 1024 --dtype float32 --download
```

Run candidate backend using cached reference outputs for correctness checks:

```bash
modal run run_modal.py --problem 1 --solution agent --m 1024 --n 1024 --dtype float32 --download --use-cached-reference
```

### Full iterative loop

```bash
python3 run_experiment.py --config experiments/default.yaml
```

### Install local dependencies (if needed)

```bash
pip install -r requirements.txt
```

## Artifact Paths

Per-problem backend artifacts are downloaded under:

- `logs/problem_<id>/<backend>/`

Typical files:

- `rank_<rank>_perf.json` (per-rank payload)
- `summary_rank0.json` (aggregate summary)
- `trace_reference_rank<rank>.json` (profiler trace)
- `trace_candidate_rank<rank>.json` (profiler trace)

Iterative optimizer run logs are written under:

- `runs/<timestamp>_<experiment_name>/`

## Extension Points For Future Agents

1. Add/replace strategies in:
   - `agent/strategies/proposers/`
   - `agent/strategies/memory/`
   - `agent/strategies/scorers/`
2. Register strategy names in `agent/strategies/registry.py`
3. Add new experiment config files under `experiments/`
4. Add problem-specific input overrides in `utils/problem_specs.py`
5. Replace seed candidate generation policy in `agent/core/optimizer.py`

## Known Operational Notes

- `run_modal.py` expects worker at `/workspace/scripts/worker.py`; keep that path stable.
- `agent` backend is treated as Triton/NVSHMEM backend for initialization.
- Local environments without `torch`/CUDA packages cannot execute worker locally; worker is designed for Modal runtime.
- Profiling is enabled in worker by default and can be disabled with `--no-profile`.
- `.pt` outputs are saved by default for all runs (all ranks), under `reference_outputs/` and `candidate_outputs/`.
- Optimizer runs precompute reference once per problem and then evaluate candidate with cached reference outputs.
- Run logs include phase/cycle events (`bootstrap_seed_copy`, `bootstrap_generated`, `correctness_fix_cycle`, `perf_opt_cycle`).

## Suggested Next Improvements

1. Add richer scorer options (multi-shape weighted, p95 rank-aware objective).
2. Add robust patch application (fallback to direct file rewrite if git apply fails).
3. Add explicit correctness adapters for tricky collectives (reduce-to-dst, reorder-sensitive ops).
4. Add prompt templates/few-shot examples per problem family.
5. Add lightweight CI checks for strategy/config loading.

