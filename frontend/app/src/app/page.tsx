"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { useQueryClient } from "@tanstack/react-query";
import { useRuns } from "@/hooks/use-run-data";
import { RunCard } from "@/components/run-card";
import { RunLauncher } from "@/components/run-launcher";
import { ExperimentBuilder } from "@/components/experiment-builder";
import { api } from "@/lib/api-client";

export default function DashboardPage() {
  const { data: runs, isLoading } = useRuns();
  const [showBuilder, setShowBuilder] = useState(false);
  const [demoLaunching, setDemoLaunching] = useState(false);
  const router = useRouter();
  const queryClient = useQueryClient();

  const handleDemoLaunch = useCallback(
    async (name: string, delayMs = 30000) => {
      setDemoLaunching(true);
      try {
        const result = await api.launchDemo(name);
        setShowBuilder(false);
        queryClient.invalidateQueries({ queryKey: ["runs"] });
        router.push(`/runs/${result.run_id}?demo=true&delay=${delayMs}`);
      } catch (e) {
        console.error("Demo launch failed:", e);
      } finally {
        setDemoLaunching(false);
      }
    },
    [queryClient, router]
  );

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
          Iterative optimization with LLMs (OpenAI, Anthropic) + Modal H100
          evaluation
        </p>
      </motion.div>

      {/* Action Buttons */}
      <div className="flex items-center gap-3 mb-8">
        <RunLauncher />
        <button
          onClick={() => setShowBuilder(!showBuilder)}
          className="pill-btn pill-btn-ghost flex items-center gap-2"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Experiment
        </button>
        <button
          onClick={() => handleDemoLaunch("demo_problem4_optimization")}
          disabled={demoLaunching}
          className="pill-btn pill-btn-ghost flex items-center gap-2 border-accent-purple/30 text-accent-purple hover:bg-accent-purple/10 disabled:opacity-50"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          {demoLaunching ? "Launching..." : "Launch Demo"}
        </button>
      </div>

      {/* Experiment Builder */}
      <AnimatePresence>
        {showBuilder && (
          <div className="mb-8">
            <ExperimentBuilder
              onClose={() => setShowBuilder(false)}
              onLaunch={handleDemoLaunch}
            />
          </div>
        )}
      </AnimatePresence>

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
