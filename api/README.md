# FERB Agentic API

This API now supports:

- `POST /chat` - GPT-backed kernel optimization chat
- `POST /optimize` - iterative agent loop for solution improvement

## Setup

```bash
cd /Users/rohk/FERB/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
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

