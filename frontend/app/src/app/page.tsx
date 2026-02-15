"use client";

import { motion } from "framer-motion";
import { useRuns } from "@/hooks/use-run-data";
import { RunCard } from "@/components/run-card";
import { RunLauncher } from "@/components/run-launcher";

export default function DashboardPage() {
  const { data: runs, isLoading } = useRuns();

  return (
    <div className="min-h-screen p-8 max-w-6xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="w-3 h-3 rounded-full bg-accent-green animate-breathe" />
          <h1 className="text-4xl font-extrabold tracking-tight text-text-primary">
            EcholoKernel
          </h1>
        </div>
        <p className="text-text-secondary text-sm ml-6">
          Iterative optimization with LLMs (OpenAI, Anthropic) + Modal H100 evaluation
        </p>
      </motion.div>

      {/* Launcher */}
      <RunLauncher />

      {/* Run List */}
      <div className="mb-6">
        <h2 className="label-muted mb-4">Optimization Runs</h2>
      </div>

      {isLoading ? (
        <div className="flex items-center gap-3 text-text-muted">
          <div className="w-4 h-4 border-2 border-accent-green/30 border-t-accent-green rounded-full animate-spin" />
          Loading runs...
        </div>
      ) : runs && runs.length > 0 ? (
        <div className="grid gap-4">
          {runs.map((run, i) => (
            <RunCard key={run.id} run={run} index={i} />
          ))}
        </div>
      ) : (
        <div className="glass-card p-10 text-center">
          <p className="text-text-muted text-lg">No optimization runs yet</p>
          <p className="text-text-muted text-sm mt-2">
            Launch a new run to get started
          </p>
        </div>
      )}
    </div>
  );
}
