# Agent Optimization Loop: Step-by-Step Example (Problem 5)

This document walks through exactly how the agent architecture works and what commands run when you optimize **problem 5**.

---

## Prerequisites

**Problem 5 must have:**
- `reference/5.py` — reference PyTorch implementation
- `solutions_triton/5_triton.py` — Triton seed (or another seed backend)

Currently, the repo has Triton seeds only for problems 1, 2, and 10. To run problem 5, you would need to add `solutions_triton/5_triton.py` and configure it in your experiment YAML.

---

## 1. Create an Experiment Config for Problem 5

Create `experiments/problem5.yaml`:

```yaml
name: problem5_optimization
max_iterations: 3
early_stop_speedup: 1.5
seed_from_backend: triton
candidate_dir: solutions_agent
run_root: runs

problems:
  - problem_id: 5
    rows: 1024
    cols: 1024
    dtype: float32

eval:
  reference_backend: reference
  candidate_backend: agent
  warmup_iters: 3
  measure_iters: 10
  profile: true
  download: true

openai:
  model: gpt-5
  temperature: 0.2

prompts:
  system_prompt: >
    You are an expert Triton/NVSHMEM performance engineer.
    Suggest concrete code patches grounded in profiling and timing data.
  schema_hint: >
    Return ONLY valid JSON with keys: diagnosis (string[]), hypotheses (string[]),
    proposed_patch (string), test_expectations (string[]).
  single_shot_template: |
    Problem: {problem_id}
    Candidate file: {candidate_file}
    Recent memory summary:
    {memory_summary}
    Latest metrics:
    {latest_metrics}
    Current candidate code:
    ```python
    {current_code}
    ```
    Return a JSON object with diagnosis/hypotheses/proposed_patch/test_expectations.
    The proposed_patch should be a unified diff patch against the candidate file.

strategies:
  proposer: single_shot
  memory: best_so_far
  scorer: speedup_mean
```

---

## 2. Run the Experiment

```bash
python3 run_experiment.py --config experiments/problem5.yaml
```

This kicks off the full optimization loop. Below is what happens step by step.

---

## 3. Optimization Loop: Step-by-Step

### Phase 0: Initialization (before loop)

**What happens:**
1. `Optimizer` loads config from `experiments/problem5.yaml`
2. Creates run directory: `runs/YYYYMMDD_HHMMSS_problem5_optimization/`
3. Seeds candidate: copies `solutions_triton/5_triton.py` → `solutions_agent/5_agent.py`

**Equivalent manual command:**
```bash
cp solutions_triton/5_triton.py solutions_agent/5_agent.py
```

---

### Phase 1: Iteration 1 — Evaluate

**Step 1a: Evaluate reference backend**

`ModalEvaluator.evaluate_pair()` runs the **reference** kernel on Modal (8× H100).

**Equivalent command:**
```bash
modal run run_modal.py \
  --problem 5 \
  --solution reference \
  --m 1024 --n 1024 \
  --dtype float32 \
  --download
```

**What happens on Modal:**
1. Modal spins up 8× H100 GPUs
2. `run_modal.py` invokes `torchrun` with `scripts/worker.py`
3. Worker loads `reference/5.py`, initializes reference backend, creates inputs via `utils/input_output_tensors.py`
4. Worker runs reference `solution()` for correctness + timing (warmup 3, measure 10 iters)
5. Worker writes `rank_*_perf.json` and `summary_rank0.json` to `/logs/problem_5/reference/`
6. If `download: true`, files are pulled to `logs/problem_5/reference/`

**Step 1b: Evaluate candidate (agent) backend**

**Equivalent command:**
```bash
modal run run_modal.py \
  --problem 5 \
  --solution agent \
  --m 1024 --n 1024 \
  --dtype float32 \
  --download
```

**What happens:**
- Same flow, but worker loads `solutions_agent/5_agent.py` instead of reference
- Outputs go to `logs/problem_5/agent/`
- Worker compares reference vs candidate outputs for correctness
- `summary_rank0.json` contains `aggregate.candidate_mean_ms`, `aggregate.reference_mean_ms`, `aggregate.speedup_vs_ref_mean`

**Step 1c: Score**

`SpeedupMeanScorer` computes:
```
score = reference_mean_ms / candidate_mean_ms
```
(e.g. 1.2 = candidate 20% faster than reference)

**Step 1d: Log**

Appends to `runs/.../run.jsonl`:
```json
{"iteration": 1, "problem_id": 5, "reference_summary": {...}, "candidate_summary": {...}, "score": 1.15}
```

---

### Phase 2: Iteration 1 — Propose & Apply

**Step 2a: Early stop check**

If `score >= early_stop_speedup` (e.g. 1.5), loop exits. Otherwise continue.

**Step 2b: Propose patch (per problem)**

For problem 5:

1. **Memory** (`BestSoFarMemory`): builds a short summary from history, e.g.  
   `"Best iteration: 1 score=1.1500. Latest iteration: 1 score=1.1500. Avoid regressions while improving speedup."`

2. **Proposer** (`SingleShotProposer`): builds prompt from template:
   - `{problem_id}` → 5
   - `{candidate_file}` → `.../solutions_agent/5_agent.py`
   - `{memory_summary}` → above string
   - `{latest_metrics}` → JSON of latest eval (reference/candidate times, speedup)
   - `{current_code}` → full contents of `solutions_agent/5_agent.py`

3. **OpenAI** (`OpenAIPatchClient`): sends system + user prompt to GPT, expects JSON:
   ```json
   {
     "diagnosis": ["..."],
     "hypotheses": ["..."],
     "proposed_patch": "--- a/solutions_agent/5_agent.py\n+++ b/...\n...",
     "test_expectations": ["..."]
   }
   ```

**Step 2c: Rollback snapshot**

```bash
cp solutions_agent/5_agent.py solutions_agent/5_agent.py.bak
```

**Step 2d: Apply patch**

```bash
# Patch written to runs/.../candidate.patch
git apply --recount --reject --whitespace=nowarn runs/.../candidate.patch
```

- Success → log `proposal_applied`
- Failure → `cp solutions_agent/5_agent.py.bak solutions_agent/5_agent.py`, log `proposal_failed`

---

### Phase 3: Iteration 2, 3, …

Same pattern:

1. **Evaluate** reference + candidate (2× `modal run` per iteration)
2. **Score** and append to history
3. **Early stop** if score ≥ threshold
4. **Propose** (with updated history and metrics)
5. **Snapshot → Apply → Rollback on failure**

---

## 4. Command Summary for Problem 5

| Step | Command |
|------|---------|
| **Start experiment** | `python3 run_experiment.py --config experiments/problem5.yaml` |
| **Seed candidate** | `cp solutions_triton/5_triton.py solutions_agent/5_agent.py` |
| **Eval reference** | `modal run run_modal.py --problem 5 --solution reference --m 1024 --n 1024 --dtype float32 --download` |
| **Eval candidate** | `modal run run_modal.py --problem 5 --solution agent --m 1024 --n 1024 --dtype float32 --download` |
| **Apply patch** | `git apply --recount --reject --whitespace=nowarn candidate.patch` |
| **Rollback** | `cp solutions_agent/5_agent.py.bak solutions_agent/5_agent.py` |

---

## 5. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  run_experiment.py                                                           │
│  └─ Optimizer(cfg).run()                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: Seed                                                               │
│  solutions_triton/5_triton.py → solutions_agent/5_agent.py                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ITERATION LOOP (1..max_iterations)                                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. EVALUATE                                                          │   │
│  │    ModalEvaluator.evaluate_pair(problem_5)                            │   │
│  │    ├─ modal run ... --solution reference  → logs/problem_5/reference │   │
│  │    └─ modal run ... --solution agent       → logs/problem_5/agent    │   │
│  │    Scorer: speedup = ref_mean / cand_mean                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. EARLY STOP?                                                       │   │
│  │    if score >= 1.5 → break                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. PROPOSE & APPLY (per problem)                                     │   │
│  │    Memory.summarize(history) → "Best: iter 1 score=1.15..."          │   │
│  │    Proposer.propose(ctx) → OpenAI API → {proposed_patch, ...}         │   │
│  │    RollbackManager.snapshot() → 5_agent.py.bak                       │   │
│  │    git apply candidate.patch                                         │   │
│  │    (on failure: rollback)                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    └──────────────► next iteration            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Files

| File | Role |
|------|------|
| `run_experiment.py` | Entry point; loads config, creates Optimizer, runs loop |
| `agent/core/optimizer.py` | Main loop: seed → evaluate → score → propose → apply |
| `agent/eval/modal_evaluator.py` | Runs `modal run` for reference + candidate |
| `agent/strategies/proposers/single_shot.py` | Builds prompt, calls OpenAI |
| `agent/strategies/memory/best_so_far.py` | Summarizes history for prompt |
| `agent/strategies/scorers/speedup_mean.py` | score = ref_mean / cand_mean |
| `run_modal.py` | Modal app; launches 8× H100, runs worker |
| `scripts/worker.py` | Loads kernel, runs correctness + timing, writes JSON |
| `solutions_agent/5_agent.py` | Editable candidate (gets patched) |
| `reference/5.py` | Baseline reference kernel |

---

## 7. Running Problem 5 Today

Problem 5 has `reference/5.py` but **no** `solutions_triton/5_triton.py`. To run the optimizer on problem 5 you need to:

1. Implement `solutions_triton/5_triton.py` (or another seed)
2. Add problem 5 to your experiment YAML
3. Run `python3 run_experiment.py --config experiments/your_config.yaml`

You can still run a **single** evaluation (no optimization loop) for problem 5 if you have a candidate:

```bash
# Reference only
modal run run_modal.py --problem 5 --solution reference --m 1024 --n 1024 --dtype float32 --download

# Candidate (requires solutions_agent/5_agent.py to exist)
modal run run_modal.py --problem 5 --solution agent --m 1024 --n 1024 --dtype float32 --download
```
