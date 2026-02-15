"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useExperiments } from "@/hooks/use-run-data";
import { api } from "@/lib/api-client";

export function RunLauncher() {
  const { data: experiments } = useExperiments();
  const [selectedConfig, setSelectedConfig] = useState<string>("");
  const [launching, setLaunching] = useState(false);
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleLaunch = async () => {
    if (!selectedConfig) return;
    setLaunching(true);
    setResult(null);
    try {
      const res = await api.launchRun(selectedConfig);
      setResult(`Launched! PID: ${res.pid}`);
    } catch (e: unknown) {
      setResult(`Error: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="mb-8">
      <button
        onClick={() => setOpen(!open)}
        className="pill-btn pill-btn-primary flex items-center gap-2"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
        Start Optimization
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="glass-card p-5 mt-4">
              <h3 className="label-muted mb-3">Select Experiment Config</h3>
              <div className="flex gap-3 flex-wrap mb-4">
                {experiments?.map((exp) => (
                  <button
                    key={exp.filename}
                    onClick={() => setSelectedConfig(exp.filename)}
                    className={`pill-btn ${
                      selectedConfig === exp.filename
                        ? "bg-accent-green/20 text-accent-green border border-accent-green/40"
                        : "pill-btn-ghost"
                    }`}
                  >
                    {exp.name}
                    {exp.max_iterations && (
                      <span className="text-text-muted ml-2">
                        ({exp.max_iterations} iters)
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {selectedConfig && (
                <div className="flex items-center gap-4">
                  <button
                    onClick={handleLaunch}
                    disabled={launching}
                    className="pill-btn pill-btn-primary disabled:opacity-50"
                  >
                    {launching ? "Launching..." : "Launch Run"}
                  </button>
                  {result && (
                    <span
                      className={`text-sm ${
                        result.startsWith("Error")
                          ? "text-accent-red"
                          : "text-accent-green"
                      }`}
                    >
                      {result}
                    </span>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
