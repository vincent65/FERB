"use client";

import { useState } from "react";
import type { IterationData } from "@/lib/types";
import { useRunStore } from "@/stores/run-store";

export function ProfilerPanel({
  iterations,
  problemId,
}: {
  iterations: IterationData[];
  problemId: number;
}) {
  const { selectedIteration } = useRunStore();
  const [selectedRank, setSelectedRank] = useState(0);

  const iter = iterations.find((i) => i.iteration === selectedIteration);
  const problemData = iter?.problems[String(problemId)];
  const ranks = problemData?.reference_summary?.ranks ?? [];
  const agg = problemData?.reference_summary?.aggregate;

  if (!selectedIteration || !iter) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm">
        Select an iteration to view profiler output
      </div>
    );
  }

  if (problemData?.eval_failed) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm">
        Evaluation failed for this iteration — no profiler data
      </div>
    );
  }

  return (
    <div className="p-4 overflow-y-auto h-full space-y-4">
      {/* Summary stats */}
      {agg && (
        <div className="grid grid-cols-5 gap-3">
          {[
            { label: "Candidate Mean", value: agg.candidate_mean_ms?.toFixed(4), unit: "ms", color: "text-accent-green" },
            { label: "Reference Mean", value: agg.reference_mean_ms?.toFixed(4), unit: "ms", color: "text-accent-blue" },
            { label: "Candidate P95", value: agg.candidate_p95_ms?.toFixed(4), unit: "ms", color: "text-accent-green" },
            { label: "Reference P95", value: agg.reference_p95_ms?.toFixed(4), unit: "ms", color: "text-accent-blue" },
            { label: "Speedup", value: agg.speedup_vs_ref_mean?.toFixed(4), unit: "x", color: "text-accent-green" },
          ].map((stat) => (
            <div key={stat.label} className="glass-card p-3 text-center">
              <p className="label-muted text-[10px] mb-1">{stat.label}</p>
              <p className={`${stat.color} font-bold text-lg`}>
                {stat.value ?? "—"}
                <span className="text-text-muted text-xs ml-1">{stat.unit}</span>
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Per-rank breakdown table */}
      {ranks.length > 0 && (
        <div className="glass-card overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="p-3 text-left label-muted">Rank</th>
                <th className="p-3 text-left label-muted">Status</th>
                <th className="p-3 text-right label-muted">Ref (ms)</th>
                <th className="p-3 text-right label-muted">Cand (ms)</th>
                <th className="p-3 text-right label-muted">Speedup</th>
                <th className="p-3 text-center label-muted">Correct</th>
                <th className="p-3 text-center label-muted">Trace</th>
              </tr>
            </thead>
            <tbody>
              {ranks.map((rank) => (
                <tr
                  key={rank.rank}
                  className={`border-b border-border-subtle last:border-0 cursor-pointer transition-colors ${
                    selectedRank === rank.rank
                      ? "bg-bg-glass-active"
                      : "hover:bg-bg-glass-hover"
                  }`}
                  onClick={() => setSelectedRank(rank.rank)}
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
                  <td className="p-3 text-right text-accent-blue">
                    {rank.reference?.wall_time_ms?.toFixed(4) ?? "—"}
                  </td>
                  <td className="p-3 text-right text-accent-green">
                    {rank.candidate?.wall_time_ms?.toFixed(4) ?? "—"}
                  </td>
                  <td className="p-3 text-right">
                    <span
                      className={
                        rank.speedup_vs_ref >= 1
                          ? "text-accent-green"
                          : "text-accent-red"
                      }
                    >
                      {rank.speedup_vs_ref?.toFixed(4) ?? "—"}x
                    </span>
                  </td>
                  <td className="p-3 text-center">
                    {rank.correctness_ok ? (
                      <span className="text-accent-green">&#10003;</span>
                    ) : (
                      <span className="text-accent-red">&#10007;</span>
                    )}
                  </td>
                  <td className="p-3 text-center">
                    {rank.trace_candidate ? (
                      <span className="text-accent-blue text-[10px]">
                        Available
                      </span>
                    ) : (
                      <span className="text-text-muted">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Trace viewer placeholder */}
      <div className="glass-card p-6 text-center">
        <p className="text-text-muted text-sm mb-2">
          Perfetto Trace Viewer
        </p>
        <p className="text-text-muted text-xs">
          Chrome trace files available for rank {selectedRank} — open in{" "}
          <a
            href="https://ui.perfetto.dev"
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-blue hover:underline"
          >
            Perfetto UI
          </a>
        </p>
      </div>
    </div>
  );
}
