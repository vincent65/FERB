# FERB Agentic API

This API now supports:

- `POST /chat` - GPT-backed kernel optimization chat
- `POST /optimize` - iterative agent loop for solution improvement
- `POST /optimize/stream` - live event stream of agent iterations

## Setup

```bash
cd /Users/rohk/FERB/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
# If you plan to run evaluator on Modal (recommended for Triton backend):
modal token new
# Optional but recommended: point to Python with torch installed for evaluator runs
export FERB_EVAL_PYTHON="/absolute/path/to/python-with-torch"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## `POST /chat`

Request:

```json
{
  "message": "How do I optimize problem 10 for fewer all-to-all rounds?",
  "model": "gpt-4o-mini"
}
```

## `POST /optimize`

Runs a generate -> (optional evaluate) -> select-best loop.

Request:

```json
{
  "objective": "Improve throughput while keeping correctness for problem 1 all-reduce",
  "problem_id": 1,
  "iterations": 3,
  "model": "gpt-4o-mini",
  "topology_json_path": "/Users/rohk/FERB/utils/example_topologies/nccl_topology_parsed.json",
  "evaluator_command": "python /path/to/evaluator.py --candidate {candidate_path}",
  "evaluator_timeout_s": 240
}
```

Notes:

- `evaluator_command` is optional. If omitted, the API still iterates and stores candidates.
- If used, your evaluator should print either:
  - JSON: `{"score": 123.4}`, or
  - text line: `score=123.4`
- Higher score is treated as better.

Outputs are saved in:

- `/Users/rohk/FERB/.agent_runs/<run_id>/candidate_iter_<n>.py`

### Real speedup evaluator (recommended)

Use the provided distributed benchmark as evaluator:

```bash
torchrun --nproc-per-node 8 /Users/rohk/FERB/scripts/benchmark_candidate.py \
  --problem 1 \
  --candidate {candidate_path} \
  --rows 1024 --cols 1024 --dtype float32 \
  --warmup 3 --iters 10 --score-only
```

For API usage, pass that whole string as `evaluator_command`.

## `POST /optimize/stream` (live thinking/run events)

This endpoint streams JSON events using Server-Sent Events (SSE), so you can watch:

- run start
- iteration start
- candidate generated
- evaluation start/completed (if evaluator is configured)
- best update
- run completed

Example:

```bash
curl -N -X POST "http://127.0.0.1:8000/optimize/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "Improve throughput for problem 1 while preserving correctness",
    "problem_id": 1,
    "iterations": 3,
    "model": "gpt-4o-mini"
  }'
```

You will receive events like:

```text
data: {"type":"run_started", ...}
data: {"type":"iteration_started","iteration":1}
data: {"type":"candidate_generated","iteration":1,...}
data: {"type":"best_updated","iteration":1,...}
data: {"type":"run_completed","result":{...}}
```

