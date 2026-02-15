"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import type { RunSummary } from "@/lib/types";

function StatusBadge({ status }: { status: string }) {
  if (status === "running") {
    return (
      <span className="status-pill bg-accent-red/15 text-accent-red">
        <span className="w-2 h-2 rounded-full bg-accent-red animate-pulse-live" />
        LIVE
      </span>
    );
  }
  if (status === "complete") {
    return (
      <span className="status-pill bg-accent-green/15 text-accent-green">
        COMPLETE
      </span>
    );
  }
  return (
    <span className="status-pill bg-accent-amber/15 text-accent-amber">
      {status.toUpperCase()}
    </span>
  );
}

export function RunCard({ run, index }: { run: RunSummary; index: number }) {
  const timestamp = new Date(run.created_at * 1000);
  const timeStr = timestamp.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
    >
      <Link href={`/runs/${run.id}`}>
        <div className="glass-card p-5 cursor-pointer group border-l-4 border-l-accent-green/30 hover:border-l-accent-green">
          <div className="flex items-start justify-between mb-3">
            <div>
              <h3 className="text-text-primary font-semibold text-sm group-hover:text-accent-green transition-colors">
                {run.name}
              </h3>
              <p className="text-text-muted text-xs mt-1">{timeStr}</p>
            </div>
            <StatusBadge status={run.status} />
          </div>

          <div className="flex items-end justify-between">
            <div className="flex gap-4">
              <div>
                <p className="label-muted mb-1">Iterations</p>
                <p className="text-text-primary font-bold text-lg">
                  {run.iterations}
                </p>
              </div>
              <div>
                <p className="label-muted mb-1">Problems</p>
                <p className="text-text-primary font-bold text-lg">
                  {run.problems.length}
                </p>
              </div>
            </div>
            {run.best_speedup > 0 && (
              <div className="text-right">
                <p className="label-muted mb-1">Best Speedup</p>
                <p className="text-accent-green font-extrabold text-2xl tracking-tight">
                  {run.best_speedup.toFixed(2)}x
                </p>
              </div>
            )}
          </div>

          {run.problems.length > 0 && (
            <div className="flex gap-2 mt-3 flex-wrap">
              {run.problems.map((pid) => (
                <span
                  key={pid}
                  className="status-pill bg-accent-blue/10 text-accent-blue"
                >
                  Problem {pid}
                </span>
              ))}
            </div>
          )}
        </div>
      </Link>
    </motion.div>
  );
}
