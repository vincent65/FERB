// ── Run types ──────────────────────────────────────────────────

export interface RunSummary {
  id: string;
  name: string;
  created_at: number;
  status: "running" | "complete" | "failed";
  iterations: number;
  best_speedup: number;
  problems: number[];
  current_iteration?: number;
}

export interface RunDetail {
  id: string;
  name: string;
  iterations: unknown[];
  events: RunEvent[];
  problems: number[];
  status: string;
  current_iteration?: number;
  pid?: number;
}

export interface RunEvent {
  iteration?: number;
  event?: string;
  problem_id?: number;
  score?: number;
  eval_failed?: boolean;
  eval_failure_kind?: string;
  eval_error?: string;
  reference_summary?: EvalSummary;
  candidate_summary?: EvalSummary;
  reference_feedback?: EvalFeedback;
  candidate_feedback?: EvalFeedback;
}

// ── Iteration types ────────────────────────────────────────────

export interface IterationData {
  iteration: number;
  score: number;
  status: "success" | "rollback" | "failed" | "correctness_fix";
  problems: Record<string, ProblemIterationData>;
}

export interface ProblemIterationData {
  problem_id: number;
  score: number;
  eval_failed: boolean;
  eval_failure_kind?: string;
  eval_error?: string;
  reference_summary?: EvalSummary;
  candidate_summary?: EvalSummary;
  reference_feedback?: EvalFeedback;
  candidate_feedback?: EvalFeedback;
}

// ── Evaluation types ───────────────────────────────────────────

export interface EvalSummary {
  problem_id: number | null;
  backend: string;
  status: string;
  n_ranks: number;
  ranks: RankResult[];
  aggregate: AggregateResult;
  error?: string;
}

export interface RankResult {
  rank: number;
  world_size: number;
  problem_id: number;
  backend: string;
  status: string;
  correctness_ok: boolean;
  correctness_note?: string;
  tolerances?: { rtol: number; atol: number };
  reference: TimingResult;
  candidate: TimingResult;
  speedup_vs_ref: number;
  trace_candidate?: string;
  trace_reference?: string;
  error?: string;
}

export interface TimingResult {
  wall_time_ms: number;
  wall_time_std_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  iterations: number;
}

export interface AggregateResult {
  candidate_mean_ms?: number;
  candidate_p95_ms?: number;
  reference_mean_ms?: number;
  reference_p95_ms?: number;
  speedup_vs_ref_mean?: number;
}

export interface EvalFeedback {
  status: string;
  aggregate: AggregateResult;
  correctness: {
    all_ok: boolean;
    failed_ranks: number[];
  };
  traces: {
    candidate: string[];
    reference: string[];
  };
  cache: {
    used_cached_reference_outputs: boolean;
    reference_timing_skipped: boolean;
  };
  error?: string;
  is_timeout?: boolean;
}

// ── LLM Output types ──────────────────────────────────────────

export interface LLMOutput {
  event: string;
  iteration: number;
  problem_id: number;
  candidate_file: string;
  proposal_mode: "perf_opt" | "correctness_fix";
  model: string;
  diagnosis: string[];
  hypotheses: string[];
  test_expectations: string[];
  candidate_code: string;
}

// ── Snapshot types ─────────────────────────────────────────────

export interface SnapshotInfo {
  filename: string;
  iteration: number;
  is_bootstrap: boolean;
  path: string;
}

export interface SnapshotCode {
  iteration: number;
  problem_id: number;
  code: string;
}

// ── Experiment types ───────────────────────────────────────────

export interface ExperimentConfig {
  filename: string;
  path: string;
  name: string;
  max_iterations?: number;
  problems: { problem_id: number; rows: number; cols: number; dtype: string }[];
  early_stop_speedup?: number;
}

// ── WebSocket event types ──────────────────────────────────────

export interface WSEvent {
  type: "iteration_event" | "llm_proposal" | "snapshot" | "run_complete" | "error";
  data?: RunEvent | LLMOutput | Record<string, unknown>;
  filename?: string;
  path?: string;
  code?: string;
  message?: string;
}
