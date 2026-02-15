"use client";

import { motion } from "framer-motion";
import type { IterationData } from "@/lib/types";
import { useRunStore } from "@/stores/run-store";

function statusColor(status: string) {
  switch (status) {
    case "success":
      return "border-l-accent-green";
    case "rollback":
    case "failed":
      return "border-l-accent-red";
    case "correctness_fix":
      return "border-l-accent-amber";
    default:
      return "border-l-accent-blue";
  }
}

function statusIcon(status: string) {
  switch (status) {
    case "success":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-accent-green">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      );
    case "rollback":
    case "failed":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-accent-red">
          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      );
    case "correctness_fix":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-accent-amber">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      );
    default:
      return null;
  }
}

export function IterationHistory({
  iterations,
  maxIterations,
}: {
  iterations: IterationData[];
  maxIterations?: number;
}) {
  const { selectedIteration, setSelectedIteration } = useRunStore();

  return (
    <div className="w-64 shrink-0 flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-border-subtle">
        <h2 className="label-muted mb-1">Iteration History</h2>
        <p className="text-text-secondary text-xs">
          {iterations.length} iteration{iterations.length !== 1 ? "s" : ""}
          {maxIterations ? ` / ${maxIterations}` : ""}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {iterations.map((iter, i) => {
          const isSelected = selectedIteration === iter.iteration;
          return (
            <motion.button
              key={iter.iteration}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
              onClick={() => setSelectedIteration(iter.iteration)}
              className={`w-full text-left rounded-lg p-3 border-l-4 transition-all duration-200 ${statusColor(
                iter.status
              )} ${
                isSelected
                  ? "glass-card-active"
                  : "bg-bg-glass hover:bg-bg-glass-hover"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  {statusIcon(iter.status)}
                  <span className="text-text-primary text-xs font-semibold">
                    Iter {iter.iteration}
                  </span>
                </div>
                <span
                  className={`text-xs font-bold ${
                    iter.status === "success"
                      ? "text-accent-green"
                      : iter.status === "rollback" || iter.status === "failed"
                      ? "text-accent-red"
                      : "text-accent-amber"
                  }`}
                >
                  {iter.score > 0
                    ? `${iter.score.toFixed(2)}x`
                    : iter.status === "rollback"
                    ? "ROLLBACK"
                    : "FAIL"}
                </span>
              </div>
              <p className="text-text-muted text-[10px] uppercase tracking-wider">
                {iter.status.replace("_", " ")}
              </p>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
