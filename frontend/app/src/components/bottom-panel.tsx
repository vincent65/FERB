"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useRunStore } from "@/stores/run-store";
import { LLMReasoning } from "./llm-reasoning";
import { ProfilerPanel } from "./profiler-panel";
import { CorrectnessPanel } from "./correctness-panel";
import { TimingCharts } from "./timing-charts";
import { ModalLogs } from "./modal-logs";
import type { IterationData } from "@/lib/types";

const TABS = [
  { id: "reasoning", label: "LLM Reasoning", icon: "brain" },
  { id: "profiler", label: "Profiler Output", icon: "chart" },
  { id: "correctness", label: "Correctness Checks", icon: "check" },
  { id: "timing", label: "Timing Comparison", icon: "clock" },
  { id: "logs", label: "Modal Logs", icon: "terminal" },
] as const;

function TabIcon({ icon }: { icon: string }) {
  switch (icon) {
    case "brain":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2a9 9 0 0 0-9 9c0 3.6 2.1 6.7 5.1 8.2.3.2.5.5.5.9V22h6.8v-1.9c0-.4.2-.7.5-.9A9 9 0 0 0 12 2z" />
        </svg>
      );
    case "chart":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />
        </svg>
      );
    case "check":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      );
    case "clock":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
        </svg>
      );
    case "terminal":
      return (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" />
        </svg>
      );
    default:
      return null;
  }
}

export function BottomPanel({
  runId,
  problemId,
  iterations,
}: {
  runId: string;
  problemId: number;
  iterations: IterationData[];
}) {
  const { activeTab, setActiveTab, bottomPanelOpen, setBottomPanelOpen } =
    useRunStore();

  return (
    <div className="flex flex-col border-t border-border-subtle bg-bg-deep/50">
      {/* Tab bar */}
      <div className="flex items-center justify-between px-4 border-b border-border-subtle">
        <div className="flex items-center gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                setBottomPanelOpen(true);
              }}
              className={`flex items-center gap-1.5 px-3 py-2.5 text-[11px] font-semibold uppercase tracking-wider transition-all ${
                activeTab === tab.id && bottomPanelOpen
                  ? "tab-active"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              <TabIcon icon={tab.icon} />
              {tab.label}
            </button>
          ))}
        </div>

        <button
          onClick={() => setBottomPanelOpen(!bottomPanelOpen)}
          className="p-2 text-text-muted hover:text-text-primary transition-colors"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform ${
              bottomPanelOpen ? "" : "rotate-180"
            }`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
      </div>

      {/* Panel content */}
      <AnimatePresence>
        {bottomPanelOpen && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 320 }}
            exit={{ height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="h-80">
              {activeTab === "reasoning" && (
                <LLMReasoning runId={runId} problemId={problemId} iterations={iterations} />
              )}
              {activeTab === "profiler" && (
                <ProfilerPanel iterations={iterations} problemId={problemId} />
              )}
              {activeTab === "correctness" && (
                <CorrectnessPanel
                  iterations={iterations}
                  problemId={problemId}
                />
              )}
              {activeTab === "timing" && (
                <TimingCharts iterations={iterations} problemId={problemId} />
              )}
              {activeTab === "logs" && (
                <ModalLogs iterations={iterations} problemId={problemId} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
