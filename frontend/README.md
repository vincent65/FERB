# EcholoKernel Frontend Dashboard

Real-time Triton Kernel Optimizer dashboard with TreeHacks-inspired dark cyberpunk design.

## Quick Start

```bash
# From the project root
./frontend/start.sh
```

Or start manually:

```bash
# Terminal 1: Backend API (from project root)
uvicorn frontend.api.main:app --port 8000

# Terminal 2: Next.js frontend
cd frontend/app
npm run dev
```

Open http://localhost:3000

## Architecture

- **Backend**: FastAPI (port 8000) — reads runs, iterations, snapshots, LLM outputs directly from filesystem
- **Frontend**: Next.js 16 + TypeScript + Tailwind CSS (port 3000)
- **Real-time**: WebSocket streaming via file watcher on iterations.jsonl
- **No database** — all state from `runs/`, `logs/`, `experiments/` directories

## Features

- Run dashboard with launch capability
- Side-by-side code viewer (Reference PyTorch vs Candidate Triton)
- Iteration slider with diff toggle
- LLM reasoning panel (diagnosis, hypotheses, expectations)
- Profiler output with per-rank timing breakdown
- Correctness checks per rank
- Timing comparison charts (speedup over iterations, per-rank bar chart)
- Modal evaluation logs
- WebSocket live updates for active runs

## Dependencies

Backend: `pip install fastapi uvicorn pyyaml watchfiles`
Frontend: `cd frontend/app && npm install`
