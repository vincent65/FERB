"""REST endpoints for experiment configs."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.run_reader import list_experiments, get_experiments_dir, get_experiment_detail
from ..services.demo_runner import DemoRunner

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("")
async def api_list_experiments():
    """List available experiment YAML configs."""
    return list_experiments()


@router.get("/{filename}")
async def api_get_experiment_detail(filename: str):
    """Get full experiment config details."""
    detail = get_experiment_detail(filename)
    if detail is None:
        raise HTTPException(status_code=404, detail="Experiment config not found")
    return detail


class SaveExperimentRequest(BaseModel):
    name: str
    max_iterations: int = 10
    early_stop_speedup: float = 1.5
    seed_from_backend: str = "triton"
    problems: list[dict] = []
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-4-6"
    llm_temperature: float = 0.2
    eval_timeout_s: int = 300
    warmup_iters: int = 1
    measure_iters: int = 2
    profile: bool = True
    proposer: str = "single_shot"
    memory: str = "best_so_far"
    scorer: str = "speedup_mean"
    retrieval_enabled: bool = True
    retrieval_top_k: int = 2


@router.post("/save")
async def api_save_experiment(request: SaveExperimentRequest):
    """Save a new experiment config YAML."""
    import yaml

    exp_dir = get_experiments_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Build YAML structure
    config = {
        "name": request.name,
        "max_iterations": request.max_iterations,
        "early_stop_speedup": request.early_stop_speedup,
        "seed_from_backend": request.seed_from_backend,
        "candidate_dir": "solutions_agent",
        "run_root": "runs",
        "problems": request.problems,
        "eval": {
            "reference_backend": "reference",
            "candidate_backend": "agent",
            "warmup_iters": request.warmup_iters,
            "measure_iters": request.measure_iters,
            "profile": request.profile,
            "download": True,
            "eval_timeout_s": request.eval_timeout_s,
            "worker_timeout_s": request.eval_timeout_s - 60,
        },
        "llm": {
            "provider": request.llm_provider,
            "model": request.llm_model,
            "temperature": request.llm_temperature,
        },
        "prompts": {
            "patch_system_prompt": "You are an expert Triton/NVSHMEM performance engineer. Suggest concrete code patches grounded in profiling and timing data.",
            "patch_schema_hint": 'Return ONLY valid JSON with keys: diagnosis (string[]), hypotheses (string[]), candidate_code (string), test_expectations (string[]). candidate_code must be the COMPLETE rewritten Python source file (not a diff/patch). It must define a top-level `solution` function.',
            "performance_patch_template": "Problem: {problem_id}\nCandidate file: {candidate_file}\nMode: performance_optimization\nYou are iteratively improving this kernel. Study the prior kernel history appended below to understand what approaches have been tried and their results.\n\nRecent memory summary:\n{memory_summary}\n\nRelevant NVSHMEM documentation:\n{retrieved_docs}\n\nLatest evaluation feedback:\n{eval_feedback}\n\nCurrent candidate code (this is what was just evaluated):\n```python\n{current_code}\n```\n\nReturn a JSON object with keys: diagnosis (string[]), hypotheses (string[]), candidate_code (string), test_expectations (string[]).\ncandidate_code must be the COMPLETE rewritten Python file (not a diff/patch).\nIt must define a top-level `solution` function.\n",
            "correctness_patch_template": "Problem: {problem_id}\nCandidate file: {candidate_file}\nMode: correctness_fix\n\nThe latest evaluation indicates correctness issues or errors.\nPrioritize making candidate outputs match reference outputs.\n\nRecent memory summary:\n{memory_summary}\n\nRelevant NVSHMEM documentation:\n{retrieved_docs}\n\nLatest evaluation feedback:\n{eval_feedback}\n\nCurrent candidate code (this is what was just evaluated):\n```python\n{current_code}\n```\n\nReturn a JSON object with keys: diagnosis (string[]), hypotheses (string[]), candidate_code (string), test_expectations (string[]).\ncandidate_code must be the COMPLETE rewritten Python file (not a diff/patch).\nIt must define a top-level `solution` function.\n",
            "bootstrap_system_prompt": "You are an expert Triton/NVSHMEM kernel engineer. Generate a correct optimization-ready Triton candidate from a reference implementation.",
            "bootstrap_template": "You are bootstrapping a brand-new problem.\nProblem: {problem_id}\nOutput file: {candidate_file}\n\nReference implementation:\n```python\n{reference_code}\n```\n\nRelevant NVSHMEM documentation:\n{retrieved_docs}\n\nSolved Triton examples to use as style/context:\n{context_examples}\n\nReturn ONLY valid JSON with key: candidate_code (string).\ncandidate_code must be a complete Python file that defines `solution`.\n",
        },
        "strategies": {
            "proposer": request.proposer,
            "memory": request.memory,
            "scorer": request.scorer,
        },
        "retrieval": {
            "enabled": request.retrieval_enabled,
            "top_k": request.retrieval_top_k,
            "docs_dir": "scraped_docs",
            "retrieval_model": "claude-haiku-4-5-20251001",
            "rlm_max_iterations": 25,
            "rlm_max_depth": 1,
        },
    }

    # Sanitize filename
    safe_name = request.name.replace(" ", "_").lower()
    filename = f"{safe_name}.yaml"
    filepath = exp_dir / filename

    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return {"filename": filename, "path": str(filepath), "status": "saved"}


class LaunchDemoRequest(BaseModel):
    name: str


@router.post("/launch-demo")
async def api_launch_demo(request: LaunchDemoRequest):
    """Launch a demo simulation run using pre-generated data."""
    result = DemoRunner.start_demo(request.name)
    return result
