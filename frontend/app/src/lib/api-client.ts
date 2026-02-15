const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ── API functions ──────────────────────────────────────────────

import type {
  RunSummary,
  RunDetail,
  IterationData,
  SnapshotInfo,
  SnapshotCode,
  LLMOutput,
  ExperimentConfig,
} from "./types";

export const api = {
  // Runs
  listRuns: () => fetchJSON<RunSummary[]>("/api/runs"),
  getRun: (runId: string) => fetchJSON<RunDetail>(`/api/runs/${runId}`),
  getIterations: (runId: string) =>
    fetchJSON<IterationData[]>(`/api/runs/${runId}/iterations`),

  // Snapshots
  getSnapshots: (runId: string, problemId: number) =>
    fetchJSON<SnapshotInfo[]>(
      `/api/runs/${runId}/problems/${problemId}/snapshots`
    ),
  getSnapshotCode: (runId: string, problemId: number, iteration: number) =>
    fetchJSON<SnapshotCode>(
      `/api/runs/${runId}/problems/${problemId}/snapshots/${iteration}`
    ),

  // Reference code
  getReferenceCode: (runId: string, problemId: number) =>
    fetchJSON<{ problem_id: number; code: string }>(
      `/api/runs/${runId}/problems/${problemId}/reference`
    ),

  // LLM outputs
  getLLMOutput: (runId: string, problemId: number, iteration: number) =>
    fetchJSON<LLMOutput>(
      `/api/runs/${runId}/problems/${problemId}/llm-output/${iteration}`
    ),

  // Traces
  getTrace: (
    runId: string,
    problemId: number,
    iteration: number,
    rank: number,
    backend = "agent"
  ) =>
    fetchJSON<unknown>(
      `/api/runs/${runId}/problems/${problemId}/traces/${iteration}/${rank}?backend=${backend}`
    ),

  // Experiments
  listExperiments: () => fetchJSON<ExperimentConfig[]>("/api/experiments"),
  getExperimentDetail: (filename: string) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    fetchJSON<any>(`/api/experiments/${filename}`),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  saveExperiment: (data: any) =>
    postJSON<{ filename: string; path: string; status: string }>(
      "/api/experiments/save",
      data
    ),

  // Launch/stop
  launchRun: (config: string) =>
    postJSON<{ pid: number; config: string; status: string }>(
      "/api/runs/launch",
      { config }
    ),
  stopRun: (runId: string) =>
    postJSON<{ status: string; run_id: string }>(`/api/runs/${runId}/stop`),

  // Demo
  launchDemo: (name: string) =>
    postJSON<{ run_id: string; status: string; total_iterations: number; delay_ms: number }>(
      "/api/experiments/launch-demo",
      { name }
    ),
  advanceDemo: (runId: string) =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    postJSON<any>(`/api/runs/${runId}/demo/advance`),
  isDemoRun: (runId: string) =>
    fetchJSON<{ is_demo: boolean }>(`/api/runs/${runId}/is-demo`),

  // WebSocket URL
  getWSUrl: (runId: string) => {
    const wsBase = API_BASE.replace("http", "ws");
    return `${wsBase}/ws/runs/${runId}`;
  },
};
