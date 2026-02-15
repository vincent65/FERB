"use client";

import { motion } from "framer-motion";
import { useLLMOutput } from "@/hooks/use-run-data";
import { useRunStore } from "@/stores/run-store";
import type { IterationData } from "@/lib/types";

function ReasoningSection({
  title,
  items,
  borderColor,
  dotColor,
  icon,
}: {
  title: string;
  items: string[];
  borderColor: string;
  dotColor: string;
  icon: React.ReactNode;
}) {
  if (!items || items.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-card p-4 border-l-4 ${borderColor}`}
    >
      <div className="flex items-center gap-2 mb-3">
        {icon}
        <span className="text-text-primary text-xs font-bold uppercase tracking-wider">
          {title}
        </span>
      </div>
      <ul className="space-y-2">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2 text-text-secondary text-xs leading-relaxed">
            <span className={`w-1.5 h-1.5 rounded-full ${dotColor} mt-1.5 shrink-0`} />
            {item}
          </li>
        ))}
      </ul>
    </motion.div>
  );
}

export function LLMReasoning({
  runId,
  problemId,
  iterations,
}: {
  runId: string;
  problemId: number;
  iterations: IterationData[];
}) {
  const { selectedIteration } = useRunStore();
  const { data: llmOutput, isLoading } = useLLMOutput(
    runId,
    problemId,
    selectedIteration ?? 0
  );
  const { data: prevLLMOutput } = useLLMOutput(
    runId,
    problemId,
    (selectedIteration ?? 1) - 1
  );

  const iter = iterations.find((i) => i.iteration === selectedIteration);
  const problemData = iter?.problems[String(problemId)];
  const currentIterFailed = problemData?.eval_failed ?? false;

  if (!selectedIteration) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-sm">
        Select an iteration to view LLM reasoning
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-3 p-6 text-text-muted text-sm">
        <div className="w-4 h-4 border-2 border-accent-purple/30 border-t-accent-purple rounded-full animate-spin" />
        Loading reasoning...
      </div>
    );
  }

  // When evaluation failed, the proposer never ran for this iteration — no LLM output.
  // Show the previous iteration's proposal (which produced the code that failed).
  const displayOutput = llmOutput ?? (currentIterFailed && prevLLMOutput ? prevLLMOutput : null);
  const isFromPreviousIter = !llmOutput && !!prevLLMOutput && currentIterFailed;

  if (!displayOutput) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-muted text-sm p-6 text-center">
        <p className="mb-2">No LLM output available for iteration {selectedIteration}</p>
        {selectedIteration === 0 ? (
          <p className="text-xs">Bootstrap iteration — no LLM proposal was generated.</p>
        ) : currentIterFailed ? (
          <p className="text-xs">
            Evaluation failed before the proposer could run. No new proposal was generated.
          </p>
        ) : null}
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4 overflow-y-auto h-full">
      {isFromPreviousIter && (
        <div className="glass-card p-3 border-l-4 border-l-accent-amber">
          <p className="text-accent-amber text-xs font-semibold">
            Proposal from iteration {displayOutput.iteration}
          </p>
          <p className="text-text-muted text-[11px] mt-1">
            This proposal produced the code that was evaluated in iteration {selectedIteration} and failed.
          </p>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <h3 className="text-text-primary text-sm font-bold">
          LLM Thought Process
        </h3>
        <span
          className={`status-pill ${
            displayOutput.proposal_mode === "perf_opt"
              ? "bg-accent-green/15 text-accent-green"
              : "bg-accent-amber/15 text-accent-amber"
          }`}
        >
          {displayOutput.proposal_mode === "perf_opt"
            ? "Performance"
            : "Correctness Fix"}
        </span>
        <span className="status-pill bg-accent-purple/15 text-accent-purple">
          {displayOutput.model}
        </span>
      </div>

      {/* Sections */}
      <div className="space-y-3">
        <ReasoningSection
          title="Initial Analysis"
          items={displayOutput.diagnosis}
          borderColor="border-l-accent-purple"
          dotColor="bg-accent-purple"
          icon={
            <span className="status-pill bg-accent-purple/15 text-accent-purple text-[10px]">
              analysis
            </span>
          }
        />

        <ReasoningSection
          title="Strategy Selection"
          items={displayOutput.hypotheses}
          borderColor="border-l-accent-blue"
          dotColor="bg-accent-blue"
          icon={
            <span className="status-pill bg-accent-blue/15 text-accent-blue text-[10px]">
              decision
            </span>
          }
        />

        <ReasoningSection
          title="Test Expectations"
          items={displayOutput.test_expectations}
          borderColor="border-l-accent-amber"
          dotColor="bg-accent-amber"
          icon={
            <span className="status-pill bg-accent-amber/15 text-accent-amber text-[10px]">
              validation
            </span>
          }
        />
      </div>
    </div>
  );
}
