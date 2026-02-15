"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useExperiments } from "@/hooks/use-run-data";
import { api } from "@/lib/api-client";
import type { ExperimentConfig } from "@/lib/types";

interface ProblemEntry {
  problem_id: number;
  rows: number;
  cols: number;
  dtype: string;
}

interface ExperimentFormData {
  name: string;
  max_iterations: number;
  early_stop_speedup: number;
  seed_from_backend: string;
  problems: ProblemEntry[];
  // LLM
  llm_provider: string;
  llm_model: string;
  llm_temperature: number;
  // Eval
  eval_timeout_s: number;
  warmup_iters: number;
  measure_iters: number;
  profile: boolean;
  // Strategies
  proposer: string;
  memory: string;
  scorer: string;
  // Retrieval
  retrieval_enabled: boolean;
  retrieval_top_k: number;
}

const DEFAULT_FORM: ExperimentFormData = {
  name: "",
  max_iterations: 10,
  early_stop_speedup: 1.5,
  seed_from_backend: "triton",
  problems: [{ problem_id: 4, rows: 1024, cols: 1024, dtype: "float32" }],
  llm_provider: "anthropic",
  llm_model: "claude-opus-4-6",
  llm_temperature: 0.2,
  eval_timeout_s: 300,
  warmup_iters: 1,
  measure_iters: 2,
  profile: true,
  proposer: "single_shot",
  memory: "best_so_far",
  scorer: "speedup_mean",
  retrieval_enabled: true,
  retrieval_top_k: 2,
};

function SectionHeader({
  children,
  icon,
}: {
  children: React.ReactNode;
  icon: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 mb-3 mt-6 first:mt-0">
      <span className="text-accent-green">{icon}</span>
      <h3 className="text-text-primary text-xs font-bold uppercase tracking-wider">
        {children}
      </h3>
      <div className="flex-1 h-px bg-border-subtle" />
    </div>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label className="block text-text-secondary text-[11px] font-semibold uppercase tracking-wider mb-1.5">
      {children}
    </label>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  type = "text",
}: {
  value: string | number;
  onChange: (val: string) => void;
  placeholder?: string;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full bg-bg-glass border border-border-subtle rounded-lg px-3 py-2 text-text-primary text-sm font-mono placeholder:text-text-muted focus:outline-none focus:border-accent-green/50 focus:ring-1 focus:ring-accent-green/20 transition-all"
    />
  );
}

function SelectInput({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (val: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-bg-glass border border-border-subtle rounded-lg px-3 py-2 text-text-primary text-sm font-mono focus:outline-none focus:border-accent-green/50 focus:ring-1 focus:ring-accent-green/20 transition-all appearance-none cursor-pointer"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value} className="bg-bg-surface">
          {opt.label}
        </option>
      ))}
    </select>
  );
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (val: boolean) => void;
  label: string;
}) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="flex items-center gap-3 group"
    >
      <div
        className={`w-9 h-5 rounded-full transition-all duration-200 relative ${
          checked
            ? "bg-accent-green/30 border border-accent-green/50"
            : "bg-bg-glass border border-border-subtle"
        }`}
      >
        <div
          className={`absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200 ${
            checked
              ? "left-4 bg-accent-green shadow-[0_0_8px_rgba(0,255,136,0.4)]"
              : "left-0.5 bg-text-muted"
          }`}
        />
      </div>
      <span className="text-text-secondary text-xs font-semibold group-hover:text-text-primary transition-colors">
        {label}
      </span>
    </button>
  );
}

export function ExperimentBuilder({
  onClose,
  onLaunch,
}: {
  onClose: () => void;
  onLaunch: (name: string, delayMs?: number) => void;
}) {
  const { data: experiments } = useExperiments();
  const [form, setForm] = useState<ExperimentFormData>(DEFAULT_FORM);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<{
    ok: boolean;
    msg: string;
  } | null>(null);

  // Load template
  const loadTemplate = useCallback(
    async (filename: string) => {
      setSelectedTemplate(filename);
      try {
        const config = await api.getExperimentDetail(filename);
        setForm({
          name: config.name || "",
          max_iterations: config.max_iterations || 10,
          early_stop_speedup: config.early_stop_speedup || 1.5,
          seed_from_backend: config.seed_from_backend || "triton",
          problems: config.problems || DEFAULT_FORM.problems,
          llm_provider: config.llm?.provider || "anthropic",
          llm_model: config.llm?.model || "claude-opus-4-6",
          llm_temperature: config.llm?.temperature ?? 0.2,
          eval_timeout_s: config.eval?.eval_timeout_s || 300,
          warmup_iters: config.eval?.warmup_iters || 1,
          measure_iters: config.eval?.measure_iters || 2,
          profile: config.eval?.profile ?? true,
          proposer: config.strategies?.proposer || "single_shot",
          memory: config.strategies?.memory || "best_so_far",
          scorer: config.strategies?.scorer || "speedup_mean",
          retrieval_enabled: config.retrieval?.enabled ?? true,
          retrieval_top_k: config.retrieval?.top_k || 2,
        });
      } catch {
        // If detail fetch fails, just use defaults
      }
    },
    []
  );

  const updateForm = useCallback(
    <K extends keyof ExperimentFormData>(key: K, value: ExperimentFormData[K]) => {
      setForm((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const updateProblem = useCallback(
    (idx: number, field: keyof ProblemEntry, value: string | number) => {
      setForm((prev) => {
        const problems = [...prev.problems];
        problems[idx] = { ...problems[idx], [field]: value };
        return { ...prev, problems };
      });
    },
    []
  );

  const addProblem = useCallback(() => {
    setForm((prev) => ({
      ...prev,
      problems: [
        ...prev.problems,
        { problem_id: 1, rows: 1024, cols: 1024, dtype: "float32" },
      ],
    }));
  }, []);

  const removeProblem = useCallback((idx: number) => {
    setForm((prev) => ({
      ...prev,
      problems: prev.problems.filter((_, i) => i !== idx),
    }));
  }, []);

  const handleSave = async () => {
    if (!form.name.trim()) {
      setSaveResult({ ok: false, msg: "Name is required" });
      return;
    }
    setSaving(true);
    setSaveResult(null);
    try {
      const res = await api.saveExperiment(form);
      setSaveResult({ ok: true, msg: `Saved as ${res.filename}` });
    } catch (e: unknown) {
      setSaveResult({
        ok: false,
        msg: e instanceof Error ? e.message : "Save failed",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleSaveAndLaunch = async () => {
    if (!form.name.trim()) {
      setSaveResult({ ok: false, msg: "Name is required" });
      return;
    }
    setSaving(true);
    setSaveResult(null);
    try {
      const res = await api.saveExperiment(form);
      setSaveResult({ ok: true, msg: `Saved as ${res.filename}` });
      onLaunch(form.name, 2500);
    } catch (e: unknown) {
      setSaveResult({
        ok: false,
        msg: e instanceof Error ? e.message : "Save failed",
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 20, scale: 0.98 }}
      transition={{ duration: 0.25 }}
      className="glass-card border border-border-subtle overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border-subtle bg-bg-glass">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-accent-green animate-breathe" />
          <h2 className="text-text-primary font-bold text-sm">
            Experiment Builder
          </h2>
        </div>
        <button
          onClick={onClose}
          className="text-text-muted hover:text-text-primary transition-colors p-1"
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
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <div className="p-5 max-h-[70vh] overflow-y-auto">
        {/* Template Selection */}
        <SectionHeader
          icon={
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
          }
        >
          Start from Template
        </SectionHeader>

        <div className="flex gap-2 flex-wrap mb-4">
          {experiments?.map((exp) => (
            <button
              key={exp.filename}
              onClick={() => loadTemplate(exp.filename)}
              className={`pill-btn text-xs py-1.5 px-3 ${
                selectedTemplate === exp.filename
                  ? "bg-accent-green/20 text-accent-green border border-accent-green/40"
                  : "pill-btn-ghost"
              }`}
            >
              {exp.name}
            </button>
          ))}
        </div>

        {/* Basic Config */}
        <SectionHeader
          icon={
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          }
        >
          Basic Configuration
        </SectionHeader>

        <div className="grid grid-cols-2 gap-4">
          <div className="col-span-2">
            <FieldLabel>Experiment Name</FieldLabel>
            <TextInput
              value={form.name}
              onChange={(v) => updateForm("name", v)}
              placeholder="my_optimization_run"
            />
          </div>
          <div>
            <FieldLabel>Max Iterations</FieldLabel>
            <TextInput
              type="number"
              value={form.max_iterations}
              onChange={(v) => updateForm("max_iterations", Number(v))}
            />
          </div>
          <div>
            <FieldLabel>Early Stop Speedup</FieldLabel>
            <TextInput
              type="number"
              value={form.early_stop_speedup}
              onChange={(v) => updateForm("early_stop_speedup", Number(v))}
            />
          </div>
        </div>

        {/* LLM */}
        <SectionHeader
          icon={
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M12 2a4 4 0 0 0-4 4v2H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2h-2V6a4 4 0 0 0-4-4z" />
            </svg>
          }
        >
          LLM Configuration
        </SectionHeader>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <FieldLabel>Provider</FieldLabel>
            <SelectInput
              value={form.llm_provider}
              onChange={(v) => updateForm("llm_provider", v)}
              options={[
                { value: "anthropic", label: "Anthropic" },
                { value: "openai", label: "OpenAI" },
              ]}
            />
          </div>
          <div>
            <FieldLabel>Model</FieldLabel>
            <SelectInput
              value={form.llm_model}
              onChange={(v) => updateForm("llm_model", v)}
              options={
                form.llm_provider === "anthropic"
                  ? [
                      { value: "claude-opus-4-6", label: "Claude Opus 4.6" },
                      { value: "claude-sonnet-4-5-20250514", label: "Claude Sonnet 4.5" },
                      { value: "claude-haiku-4-5-20251001", label: "Claude Haiku 4.5" },
                    ]
                  : [
                      { value: "gpt-5", label: "GPT-5" },
                      { value: "o3", label: "o3" },
                      { value: "o4-mini", label: "o4-mini" },
                      { value: "gpt-4.1", label: "GPT-4.1" },
                    ]
              }
            />
          </div>
          <div>
            <FieldLabel>Temperature</FieldLabel>
            <TextInput
              type="number"
              value={form.llm_temperature}
              onChange={(v) => updateForm("llm_temperature", Number(v))}
            />
          </div>
        </div>

        {/* Problems */}
        <SectionHeader
          icon={
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <line x1="3" y1="9" x2="21" y2="9" />
              <line x1="9" y1="21" x2="9" y2="9" />
            </svg>
          }
        >
          Problems
        </SectionHeader>

        <div className="space-y-3">
          {form.problems.map((prob, idx) => (
            <div
              key={idx}
              className="flex items-end gap-3 p-3 rounded-lg bg-bg-glass border border-border-subtle"
            >
              <div className="flex-1">
                <FieldLabel>Problem ID</FieldLabel>
                <TextInput
                  type="number"
                  value={prob.problem_id}
                  onChange={(v) => updateProblem(idx, "problem_id", Number(v))}
                />
              </div>
              <div className="flex-1">
                <FieldLabel>Rows</FieldLabel>
                <TextInput
                  type="number"
                  value={prob.rows}
                  onChange={(v) => updateProblem(idx, "rows", Number(v))}
                />
              </div>
              <div className="flex-1">
                <FieldLabel>Cols</FieldLabel>
                <TextInput
                  type="number"
                  value={prob.cols}
                  onChange={(v) => updateProblem(idx, "cols", Number(v))}
                />
              </div>
              <div className="flex-1">
                <FieldLabel>Dtype</FieldLabel>
                <SelectInput
                  value={prob.dtype}
                  onChange={(v) => updateProblem(idx, "dtype", v)}
                  options={[
                    { value: "float32", label: "float32" },
                    { value: "float16", label: "float16" },
                    { value: "bfloat16", label: "bfloat16" },
                  ]}
                />
              </div>
              {form.problems.length > 1 && (
                <button
                  onClick={() => removeProblem(idx)}
                  className="text-accent-red hover:text-accent-red/80 transition-colors p-2"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                </button>
              )}
            </div>
          ))}
          <button
            onClick={addProblem}
            className="pill-btn pill-btn-ghost text-xs py-1.5"
          >
            + Add Problem
          </button>
        </div>

        {/* Advanced Toggle */}
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 mt-6 text-text-muted hover:text-text-secondary transition-colors"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className={`transition-transform ${showAdvanced ? "rotate-90" : ""}`}
          >
            <polyline points="9 18 15 12 9 6" />
          </svg>
          <span className="text-xs font-semibold uppercase tracking-wider">
            Advanced Settings
          </span>
        </button>

        <AnimatePresence>
          {showAdvanced && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              {/* Eval Config */}
              <SectionHeader
                icon={
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                  </svg>
                }
              >
                Evaluation
              </SectionHeader>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <FieldLabel>Timeout (s)</FieldLabel>
                  <TextInput
                    type="number"
                    value={form.eval_timeout_s}
                    onChange={(v) => updateForm("eval_timeout_s", Number(v))}
                  />
                </div>
                <div>
                  <FieldLabel>Warmup Iters</FieldLabel>
                  <TextInput
                    type="number"
                    value={form.warmup_iters}
                    onChange={(v) => updateForm("warmup_iters", Number(v))}
                  />
                </div>
                <div>
                  <FieldLabel>Measure Iters</FieldLabel>
                  <TextInput
                    type="number"
                    value={form.measure_iters}
                    onChange={(v) => updateForm("measure_iters", Number(v))}
                  />
                </div>
              </div>
              <div className="mt-3">
                <Toggle
                  checked={form.profile}
                  onChange={(v) => updateForm("profile", v)}
                  label="Enable Profiling"
                />
              </div>

              {/* Strategies */}
              <SectionHeader
                icon={
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
                  </svg>
                }
              >
                Strategies
              </SectionHeader>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <FieldLabel>Proposer</FieldLabel>
                  <SelectInput
                    value={form.proposer}
                    onChange={(v) => updateForm("proposer", v)}
                    options={[
                      { value: "single_shot", label: "Single Shot" },
                      { value: "best_of_n", label: "Best of N" },
                    ]}
                  />
                </div>
                <div>
                  <FieldLabel>Memory</FieldLabel>
                  <SelectInput
                    value={form.memory}
                    onChange={(v) => updateForm("memory", v)}
                    options={[
                      { value: "best_so_far", label: "Best So Far" },
                      { value: "full_history", label: "Full History" },
                    ]}
                  />
                </div>
                <div>
                  <FieldLabel>Scorer</FieldLabel>
                  <SelectInput
                    value={form.scorer}
                    onChange={(v) => updateForm("scorer", v)}
                    options={[
                      { value: "speedup_mean", label: "Speedup Mean" },
                      { value: "speedup_p95", label: "Speedup P95" },
                    ]}
                  />
                </div>
              </div>

              {/* Retrieval */}
              <SectionHeader
                icon={
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                }
              >
                RAG Retrieval
              </SectionHeader>
              <div className="flex items-center gap-6">
                <Toggle
                  checked={form.retrieval_enabled}
                  onChange={(v) => updateForm("retrieval_enabled", v)}
                  label="Enable Retrieval"
                />
                {form.retrieval_enabled && (
                  <div className="w-32">
                    <FieldLabel>Top K</FieldLabel>
                    <TextInput
                      type="number"
                      value={form.retrieval_top_k}
                      onChange={(v) => updateForm("retrieval_top_k", Number(v))}
                    />
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-5 py-4 border-t border-border-subtle bg-bg-glass">
        {saveResult && (
          <span
            className={`text-xs font-semibold ${
              saveResult.ok ? "text-accent-green" : "text-accent-red"
            }`}
          >
            {saveResult.msg}
          </span>
        )}
        {!saveResult && <div />}
        <div className="flex items-center gap-3">
          <button
            onClick={handleSave}
            disabled={saving}
            className="pill-btn pill-btn-ghost text-xs disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save Config"}
          </button>
          <button
            onClick={handleSaveAndLaunch}
            disabled={saving}
            className="pill-btn pill-btn-primary text-xs disabled:opacity-50"
          >
            {saving ? "..." : "Save & Launch"}
          </button>
        </div>
      </div>
    </motion.div>
  );
}
