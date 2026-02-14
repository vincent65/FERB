"""
FERB API:
- Chat endpoint for kernel optimization guidance
- Agentic optimization endpoint for iterative GPT-driven solution improvement
"""

import os
import json
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic import gpt_chat_reply
from agentic import run_agentic_optimization
from agentic import stream_agentic_optimization_events


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o-mini"


class ChatResponse(BaseModel):
    reply: str
    ok: bool = True


class OptimizeRequest(BaseModel):
    objective: str = Field(..., min_length=8)
    problem_id: int = Field(..., ge=1)
    iterations: int = Field(default=3, ge=1, le=10)
    model: str = "gpt-4o-mini"
    target_backend: str = "triton"
    topology_json_path: str | None = None
    evaluator_command: str | None = None
    evaluator_timeout_s: int = Field(default=240, ge=10, le=3600)
    evaluator_python: str | None = None
    include_full_code: bool = False
    include_trace_output: bool = False


class OptimizeResponse(BaseModel):
    ok: bool = True
    result: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="FERB Kernel Optimization Chat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    # GPT-backed chat; if no API key, return setup guidance.
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return ChatResponse(
            reply=(
                "OPENAI_API_KEY is not set. Export it first, then retry. "
                "Example: `export OPENAI_API_KEY=...`"
            ),
            ok=False,
        )
    try:
        reply = gpt_chat_reply(body.message, model=body.model)
        return ChatResponse(reply=reply, ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(body: OptimizeRequest) -> OptimizeResponse:
    """
    Agentic optimization loop:
    - Generate candidate code with GPT
    - Optionally evaluate candidate via command
    - Keep best candidate
    """
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for /optimize",
        )
    try:
        result = run_agentic_optimization(
            objective=body.objective,
            problem_id=body.problem_id,
            iterations=body.iterations,
            model=body.model,
            target_backend=body.target_backend,
            topology_json_path=body.topology_json_path,
            evaluator_command=body.evaluator_command,
            evaluator_timeout_s=body.evaluator_timeout_s,
            evaluator_python=body.evaluator_python,
            include_full_code=body.include_full_code,
            include_trace_output=body.include_trace_output,
        )
        return OptimizeResponse(ok=True, result=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"optimization failed: {exc}") from exc


@app.post("/optimize/stream")
def optimize_stream(body: OptimizeRequest) -> StreamingResponse:
    """
    Stream live agent iteration events via Server-Sent Events (SSE).
    """
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is required for /optimize/stream",
        )

    def _event_stream():
        try:
            for event in stream_agentic_optimization_events(
                objective=body.objective,
                problem_id=body.problem_id,
                iterations=body.iterations,
                model=body.model,
                target_backend=body.target_backend,
                topology_json_path=body.topology_json_path,
                evaluator_command=body.evaluator_command,
                evaluator_timeout_s=body.evaluator_timeout_s,
                evaluator_python=body.evaluator_python,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            err_event = {"type": "error", "detail": str(exc)}
            yield f"data: {json.dumps(err_event)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# Serve frontend (whitespace chatbot) when running from repo root
_frontend = Path(__file__).resolve().parent.parent / "frontend"
if _frontend.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Dev server (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
