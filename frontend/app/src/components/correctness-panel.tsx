"use client";

import { motion } from "framer-motion";
import type { IterationData } from "@/lib/types";
import { useRunStore } from "@/stores/run-store";

export function CorrectnessPanel({
  iterations,
  problemId,
}: {
  iterations: IterationData[];
  problemId: number;
}) {
  const { selectedIteration } = useRunStore();
  const iter = iterations.find((i) => i.iteration === selectedIteration);

  if (!selectedIteration || !iter) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm">
        Select an iteration to view correctness checks
      </div>
    );
  }

  const problemData = iter.problems[String(problemId)];
  if (!problemData) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm">
        No data for problem {problemId} in iteration {selectedIteration}
      </div>
    );
  }

  const candSummary = problemData.candidate_summary;
  const candFeedback = problemData.candidate_feedback;
  const allCorrect = candFeedback?.correctness?.all_ok ?? false;
  const failedRanks = candFeedback?.correctness?.failed_ranks ?? [];
  // Use candidate_summary.ranks for per-rank correctness — reference_summary.ranks
  // are from the reference evaluation (reference vs reference) and would show "OK"
  // even when the candidate failed. When eval failed, candidate_summary has no ranks.
  const ranks = candSummary?.ranks ?? [];

  return (
    <div className="p-4 overflow-y-auto h-full space-y-4">
      {/* Overall status */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`glass-card p-4 border-l-4 ${
          problemData.eval_failed
            ? "border-l-accent-red"
            : allCorrect
            ? "border-l-accent-green"
            : "border-l-accent-amber"
        }`}
      >
        <div className="flex items-center gap-3">
          {problemData.eval_failed ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent-red">
              <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
            </svg>
          ) : allCorrect ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent-green">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent-amber">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          )}
          <div>
            <h3 className="text-text-primary text-sm font-bold">
              {problemData.eval_failed
                ? "Evaluation Failed"
                : allCorrect
                ? "All Ranks Passed"
                : `Failures Detected (${failedRanks.length} rank${failedRanks.length !== 1 ? "s" : ""})`}
            </h3>
            {problemData.eval_error && (
              <p className="text-accent-red text-xs mt-1 leading-relaxed">
                {problemData.eval_error.slice(0, 200)}
                {problemData.eval_error.length > 200 ? "..." : ""}
              </p>
            )}
          </div>
        </div>
      </motion.div>

      {/* Per-rank table */}
      {ranks.length > 0 && (
        <div className="glass-card overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="p-3 text-left label-muted">Rank</th>
                <th className="p-3 text-left label-muted">Status</th>
                <th className="p-3 text-left label-muted">Correctness</th>
                <th className="p-3 text-left label-muted">Tolerance</th>
              </tr>
            </thead>
            <tbody>
              {ranks.map((rank) => (
                <tr
                  key={rank.rank}
                  className="border-b border-border-subtle last:border-0 hover:bg-bg-glass-hover transition-colors"
                >
                  <td className="p-3 text-text-primary font-semibold">
                    {rank.rank}
                  </td>
                  <td className="p-3">
                    <span
                      className={`status-pill ${
                        rank.status === "ok"
                          ? "bg-accent-green/15 text-accent-green"
                          : "bg-accent-red/15 text-accent-red"
                      }`}
                    >
                      {rank.status}
                    </span>
                  </td>
                  <td className="p-3">
                    {rank.correctness_ok ? (
                      <span className="text-accent-green">PASS</span>
                    ) : (
                      <span className="text-accent-red">FAIL</span>
                    )}
                  </td>
                  <td className="p-3 text-text-muted">
                    {rank.tolerances
                      ? `rtol=${rank.tolerances.rtol}, atol=${rank.tolerances.atol}`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
