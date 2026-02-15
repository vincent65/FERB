"use client";

import { useEffect, useRef } from "react";
import type { IterationData } from "@/lib/types";
import { useRunStore } from "@/stores/run-store";

export function ModalLogs({
  iterations,
  problemId,
}: {
  iterations: IterationData[];
  problemId: number;
}) {
  const { selectedIteration } = useRunStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  const iter = iterations.find((i) => i.iteration === selectedIteration);
  const problemData = iter?.problems[String(problemId)];

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [selectedIteration]);

  if (!selectedIteration || !iter) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm">
        Select an iteration to view logs
      </div>
    );
  }

  // Collect log-worthy events
  const logEntries: { level: string; message: string }[] = [];

  if (problemData) {
    const refSummary = problemData.reference_summary;
    const candSummary = problemData.candidate_summary;

    logEntries.push({
      level: "info",
      message: `[Iter ${selectedIteration}] Evaluating problem ${problemId}`,
    });

    if (refSummary) {
      logEntries.push({
        level: refSummary.status === "ok" ? "info" : "error",
        message: `[Reference] status=${refSummary.status}, n_ranks=${refSummary.n_ranks}`,
      });
      if (refSummary.aggregate?.reference_mean_ms) {
        logEntries.push({
          level: "info",
          message: `[Reference] mean=${refSummary.aggregate.reference_mean_ms.toFixed(4)}ms, p95=${refSummary.aggregate.reference_p95_ms?.toFixed(4)}ms`,
        });
      }
    }

    if (candSummary) {
      logEntries.push({
        level: candSummary.status === "ok" ? "info" : "error",
        message: `[Candidate] status=${candSummary.status}, n_ranks=${candSummary.n_ranks}`,
      });
      if (candSummary.error) {
        logEntries.push({
          level: "error",
          message: `[Candidate] Error: ${candSummary.error.slice(0, 300)}`,
        });
      }
      if (candSummary.aggregate?.candidate_mean_ms) {
        logEntries.push({
          level: "info",
          message: `[Candidate] mean=${candSummary.aggregate.candidate_mean_ms.toFixed(4)}ms, speedup=${candSummary.aggregate.speedup_vs_ref_mean?.toFixed(4)}x`,
        });
      }
    }

    if (problemData.eval_failed) {
      logEntries.push({
        level: "error",
        message: `[EVAL FAILED] kind=${problemData.eval_failure_kind}`,
      });
      if (problemData.eval_error) {
        logEntries.push({
          level: "error",
          message: problemData.eval_error.slice(0, 500),
        });
      }
    } else {
      logEntries.push({
        level: "info",
        message: `[Score] ${problemData.score.toFixed(4)}x speedup`,
      });
    }
  }

  return (
    <div className="p-4 h-full overflow-y-auto font-mono text-xs">
      <div className="space-y-1">
        {logEntries.map((entry, i) => (
          <div
            key={i}
            className={`py-0.5 ${
              entry.level === "error"
                ? "text-accent-red"
                : entry.level === "warn"
                ? "text-accent-amber"
                : "text-text-secondary"
            }`}
          >
            <span className="text-text-muted select-none mr-2">
              {String(i + 1).padStart(3, " ")}
            </span>
            {entry.message}
          </div>
        ))}
      </div>
      <div ref={bottomRef} />
    </div>
  );
}
