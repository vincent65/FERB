"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api-client";

export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: 10000, // poll every 10s for new runs
  });
}

export function useRunDetail(runId: string) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    enabled: !!runId,
  });
}

export function useIterations(runId: string) {
  return useQuery({
    queryKey: ["iterations", runId],
    queryFn: () => api.getIterations(runId),
    enabled: !!runId,
    refetchInterval: 5000,
  });
}

export function useSnapshots(runId: string, problemId: number) {
  return useQuery({
    queryKey: ["snapshots", runId, problemId],
    queryFn: () => api.getSnapshots(runId, problemId),
    enabled: !!runId && !!problemId,
  });
}

export function useSnapshotCode(
  runId: string,
  problemId: number,
  iteration: number
) {
  return useQuery({
    queryKey: ["snapshot-code", runId, problemId, iteration],
    queryFn: () => api.getSnapshotCode(runId, problemId, iteration),
    enabled: !!runId && !!problemId && iteration >= 0,
  });
}

export function useReferenceCode(runId: string, problemId: number) {
  return useQuery({
    queryKey: ["reference-code", runId, problemId],
    queryFn: () => api.getReferenceCode(runId, problemId),
    enabled: !!runId && !!problemId,
  });
}

export function useLLMOutput(
  runId: string,
  problemId: number,
  iteration: number
) {
  return useQuery({
    queryKey: ["llm-output", runId, problemId, iteration],
    queryFn: () => api.getLLMOutput(runId, problemId, iteration),
    enabled: !!runId && !!problemId && iteration > 0,
  });
}

export function useExperiments() {
  return useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
  });
}
