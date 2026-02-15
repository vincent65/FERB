"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from "recharts";
import type { IterationData } from "@/lib/types";
import { useRunStore } from "@/stores/run-store";

const CHART_GREEN = "#00ff88";
const CHART_BLUE = "#3b82f6";
const CHART_RED = "#ef4444";
const CHART_GRID = "rgba(255,255,255,0.06)";
const CHART_TEXT = "rgba(255,255,255,0.4)";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload) return null;
  return (
    <div className="glass-card p-3 text-xs border border-border-subtle">
      <p className="text-text-primary font-semibold mb-1">Iteration {label}</p>
      {payload.map((entry: { name: string; value: number; color: string }, i: number) => (
        <p key={i} style={{ color: entry.color }}>
          {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
}

export function TimingCharts({
  iterations,
  problemId,
}: {
  iterations: IterationData[];
  problemId: number;
}) {
  const { selectedIteration } = useRunStore();

  // Speedup over iterations chart data
  const speedupData = iterations.map((iter) => {
    const problem = iter.problems[String(problemId)];
    const agg = problem?.reference_summary?.aggregate;
    return {
      iteration: iter.iteration,
      speedup: problem?.score ?? 0,
      candidate_ms: agg?.candidate_mean_ms ?? 0,
      reference_ms: agg?.reference_mean_ms ?? 0,
    };
  });

  // Per-rank timing for selected iteration
  const selectedIter = iterations.find(
    (i) => i.iteration === selectedIteration
  );
  const selectedProblem = selectedIter?.problems[String(problemId)];
  const ranks = selectedProblem?.reference_summary?.ranks ?? [];

  const rankData = ranks.map((rank) => ({
    rank: `R${rank.rank}`,
    reference: rank.reference?.wall_time_ms ?? 0,
    candidate: rank.candidate?.wall_time_ms ?? 0,
    speedup: rank.speedup_vs_ref ?? 0,
  }));

  return (
    <div className="p-4 space-y-6 overflow-y-auto h-full">
      {/* Speedup over iterations */}
      <div>
        <h3 className="label-muted mb-3">Speedup Over Iterations</h3>
        <div className="glass-card p-4">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={speedupData}>
              <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" />
              <XAxis
                dataKey="iteration"
                tick={{ fill: CHART_TEXT, fontSize: 10 }}
                axisLine={{ stroke: CHART_GRID }}
              />
              <YAxis
                tick={{ fill: CHART_TEXT, fontSize: 10 }}
                axisLine={{ stroke: CHART_GRID }}
                domain={["auto", "auto"]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="speedup"
                stroke={CHART_GREEN}
                strokeWidth={2}
                dot={{ fill: CHART_GREEN, r: 4 }}
                activeDot={{ r: 6, fill: CHART_GREEN, stroke: "#fff", strokeWidth: 1 }}
                name="Speedup"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-rank timing comparison */}
      {rankData.length > 0 && (
        <div>
          <h3 className="label-muted mb-3">
            Per-Rank Timing (Iteration {selectedIteration})
          </h3>
          <div className="glass-card p-4">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={rankData}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" />
                <XAxis
                  dataKey="rank"
                  tick={{ fill: CHART_TEXT, fontSize: 10 }}
                  axisLine={{ stroke: CHART_GRID }}
                />
                <YAxis
                  tick={{ fill: CHART_TEXT, fontSize: 10 }}
                  axisLine={{ stroke: CHART_GRID }}
                  label={{
                    value: "ms",
                    angle: -90,
                    position: "insideLeft",
                    fill: CHART_TEXT,
                    fontSize: 10,
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend
                  wrapperStyle={{ fontSize: 10, color: CHART_TEXT }}
                />
                <Bar dataKey="reference" fill={CHART_BLUE} name="Reference (ms)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="candidate" fill={CHART_GREEN} name="Candidate (ms)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Summary stats */}
      {selectedProblem?.reference_summary?.aggregate && (
        <div>
          <h3 className="label-muted mb-3">Aggregate Stats</h3>
          <div className="grid grid-cols-2 gap-3">
            {[
              {
                label: "Candidate Mean",
                value: selectedProblem.reference_summary.aggregate.candidate_mean_ms?.toFixed(4),
                unit: "ms",
                color: "text-accent-green",
              },
              {
                label: "Reference Mean",
                value: selectedProblem.reference_summary.aggregate.reference_mean_ms?.toFixed(4),
                unit: "ms",
                color: "text-accent-blue",
              },
              {
                label: "Candidate P95",
                value: selectedProblem.reference_summary.aggregate.candidate_p95_ms?.toFixed(4),
                unit: "ms",
                color: "text-accent-green",
              },
              {
                label: "Speedup Mean",
                value: selectedProblem.reference_summary.aggregate.speedup_vs_ref_mean?.toFixed(4),
                unit: "x",
                color: "text-accent-green",
              },
            ].map((stat) => (
              <div key={stat.label} className="glass-card p-3">
                <p className="label-muted text-[10px] mb-1">{stat.label}</p>
                <p className={`${stat.color} font-bold text-lg`}>
                  {stat.value ?? "—"}
                  <span className="text-text-muted text-xs ml-1">{stat.unit}</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
