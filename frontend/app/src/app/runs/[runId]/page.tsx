"use client";

import { useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { useQueryClient } from "@tanstack/react-query";
import { useRunDetail, useIterations } from "@/hooks/use-run-data";
import { useWebSocket } from "@/hooks/use-websocket";
import { useRunStore } from "@/stores/run-store";
import { IterationHistory } from "@/components/iteration-history";
import { CodeViewer } from "@/components/code-viewer";
import { BottomPanel } from "@/components/bottom-panel";
import type { WSEvent } from "@/lib/types";

function ConnectionIndicator({ connected }: { connected: boolean }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <div
        className={`w-2 h-2 rounded-full ${
          connected
            ? "bg-accent-green animate-breathe"
            : "bg-text-muted"
        }`}
      />
      <span className={connected ? "text-accent-green" : "text-text-muted"}>
        {connected ? "LIVE" : "DISCONNECTED"}
      </span>
    </div>
  );
}

export default function RunDetailPage() {
  const params = useParams();
  const runId = params.runId as string;
  const queryClient = useQueryClient();

  const { data: run, isLoading: runLoading } = useRunDetail(runId);
  const { data: iterations } = useIterations(runId);
  const {
    selectedIteration,
    setSelectedIteration,
    activeProblemId,
    setActiveProblemId,
  } = useRunStore();

  // WebSocket for live updates
  const handleWSEvent = useCallback(
    (event: WSEvent) => {
      // Refetch data when new events come in
      if (event.type === "iteration_event" || event.type === "llm_proposal") {
        queryClient.invalidateQueries({ queryKey: ["iterations", runId] });
        queryClient.invalidateQueries({ queryKey: ["run", runId] });
      }
      if (event.type === "snapshot") {
        queryClient.invalidateQueries({ queryKey: ["snapshots"] });
        queryClient.invalidateQueries({ queryKey: ["snapshot-code"] });
      }
    },
    [queryClient, runId]
  );

  const { connected } = useWebSocket({
    runId,
    enabled: run?.status === "running",
    onEvent: handleWSEvent,
  });

  // Auto-select first problem and latest iteration
  useEffect(() => {
    if (run?.problems?.length && !activeProblemId) {
      setActiveProblemId(run.problems[0]);
    }
  }, [run?.problems, activeProblemId, setActiveProblemId]);

  useEffect(() => {
    if (iterations?.length && selectedIteration === null) {
      const lastIter = iterations[iterations.length - 1];
      setSelectedIteration(lastIter.iteration);
    }
  }, [iterations, selectedIteration, setSelectedIteration]);

  if (runLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex items-center gap-3 text-text-muted">
          <div className="w-6 h-6 border-2 border-accent-green/30 border-t-accent-green rounded-full animate-spin" />
          Loading run...
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-text-muted text-lg mb-4">Run not found</p>
          <Link href="/" className="pill-btn pill-btn-ghost">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  const problemId = activeProblemId ?? run.problems[0] ?? 0;

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Top bar */}
      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between px-5 py-3 border-b border-border-subtle bg-bg-deep/80 backdrop-blur-md shrink-0"
      >
        <div className="flex items-center gap-4">
          <Link
            href="/"
            className="text-text-muted hover:text-text-primary transition-colors"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="19" y1="12" x2="5" y2="12" />
              <polyline points="12 19 5 12 12 5" />
            </svg>
          </Link>

          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-text-primary font-bold text-sm">
                {run.name}
              </h1>
              <span
                className={`status-pill ${
                  run.status === "running"
                    ? "bg-accent-red/15 text-accent-red"
                    : "bg-accent-green/15 text-accent-green"
                }`}
              >
                {run.status === "running" && (
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-red animate-pulse-live" />
                )}
                {run.status.toUpperCase()}
              </span>
            </div>
            <p className="text-text-muted text-[10px]">
              {run.id}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          {/* Iteration counter */}
          <div className="text-center">
            <p className="label-muted text-[10px]">Iteration</p>
            <p className="text-text-primary font-bold">
              <span className="text-accent-green text-lg">
                {selectedIteration ?? "—"}
              </span>
              <span className="text-text-muted text-sm mx-1">/</span>
              <span className="text-sm">{iterations?.length ?? 0}</span>
            </p>
          </div>

          {/* WS connection */}
          {run.status === "running" && (
            <ConnectionIndicator connected={connected} />
          )}
        </div>
      </motion.header>

      {/* Problem tabs */}
      {run.problems.length > 1 && (
        <div className="flex items-center gap-1 px-4 py-2 border-b border-border-subtle bg-bg-deep/50 shrink-0">
          {run.problems.map((pid) => {
            const bestScore = iterations
              ?.filter((i) => i.problems[String(pid)]?.score)
              .reduce(
                (best, i) =>
                  Math.max(best, i.problems[String(pid)]?.score ?? 0),
                0
              ) ?? 0;

            return (
              <button
                key={pid}
                onClick={() => setActiveProblemId(pid)}
                className={`px-4 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                  problemId === pid
                    ? "bg-accent-green/15 text-accent-green border border-accent-green/30"
                    : "text-text-muted hover:text-text-secondary hover:bg-bg-glass"
                }`}
              >
                Problem {pid}
                {bestScore > 0 && (
                  <span className="ml-2 text-accent-green">
                    {bestScore.toFixed(2)}x
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Main content area */}
      <div className="flex flex-1 min-h-0">
        {/* Iteration History Sidebar */}
        <div className="border-r border-border-subtle bg-bg-deep/30">
          <IterationHistory
            iterations={iterations ?? []}
            maxIterations={undefined}
          />
        </div>

        {/* Code viewer + bottom panel */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0">
          {/* Code viewer */}
          <div className="flex-1 min-h-0">
            <CodeViewer runId={runId} problemId={problemId} />
          </div>

          {/* Bottom panel with tabs */}
          <BottomPanel
            runId={runId}
            problemId={problemId}
            iterations={iterations ?? []}
          />
        </div>
      </div>
    </div>
  );
}
