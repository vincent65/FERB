"use client";

import { useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { useRunStore } from "@/stores/run-store";
import { useReferenceCode, useSnapshotCode, useSnapshots } from "@/hooks/use-run-data";

// Dynamic import for Monaco to avoid SSR issues
const MonacoDiffEditor = dynamic(
  () => import("@monaco-editor/react").then((mod) => mod.DiffEditor),
  { ssr: false }
);
const MonacoEditor = dynamic(
  () => import("@monaco-editor/react").then((mod) => mod.default),
  { ssr: false }
);

function IterationSlider({
  snapshots,
  currentIteration,
  onChange,
}: {
  snapshots: { iteration: number }[];
  currentIteration: number;
  onChange: (iter: number) => void;
}) {
  const maxIter = Math.max(...snapshots.map((s) => s.iteration), 0);

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-bg-glass border-b border-border-subtle">
      <span className="label-muted text-[10px] shrink-0">ITERATION</span>
      <input
        type="range"
        min={0}
        max={maxIter}
        value={currentIteration}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 h-1 accent-accent-green cursor-pointer"
      />
      <span className="text-accent-green font-bold text-sm w-8 text-right">
        {currentIteration}
      </span>
    </div>
  );
}

export function CodeViewer({
  runId,
  problemId,
}: {
  runId: string;
  problemId: number;
}) {
  const { selectedIteration, setSelectedIteration, showDiff, setShowDiff } =
    useRunStore();
  const { data: referenceData } = useReferenceCode(runId, problemId);
  const { data: snapshots } = useSnapshots(runId, problemId);

  const effectiveIteration = selectedIteration ?? snapshots?.[snapshots.length - 1]?.iteration ?? 0;
  const { data: snapshotData } = useSnapshotCode(runId, problemId, effectiveIteration);

  // For diff: get previous iteration code
  const prevIteration = effectiveIteration > 0 ? effectiveIteration - 1 : 0;
  const { data: prevSnapshotData } = useSnapshotCode(
    runId,
    problemId,
    showDiff ? prevIteration : -1
  );

  const handleIterChange = useCallback(
    (iter: number) => setSelectedIteration(iter),
    [setSelectedIteration]
  );

  const editorOptions = useMemo(
    () => ({
      readOnly: true,
      minimap: { enabled: false },
      fontSize: 13,
      lineNumbers: "on" as const,
      scrollBeyondLastLine: false,
      wordWrap: "on" as const,
      renderSideBySide: true,
      fontFamily: "'JetBrains Mono', monospace",
      padding: { top: 12 },
    }),
    []
  );

  const monacoThemeConfig = useCallback((monaco: typeof import("monaco-editor")) => {
    monaco.editor.defineTheme("echolokernel-dark", {
      base: "vs-dark",
      inherit: true,
      rules: [
        { token: "comment", foreground: "4a6a8a", fontStyle: "italic" },
        { token: "keyword", foreground: "a855f7" },
        { token: "string", foreground: "00ff88" },
        { token: "number", foreground: "f59e0b" },
        { token: "type", foreground: "3b82f6" },
      ],
      colors: {
        "editor.background": "#0c1829",
        "editor.foreground": "#e2e8f0",
        "editor.lineHighlightBackground": "#ffffff08",
        "editorLineNumber.foreground": "#ffffff38",
        "editorLineNumber.activeForeground": "#00ff88",
        "editor.selectionBackground": "#00ff8820",
        "editorGutter.background": "#0a1628",
        "diffEditor.insertedTextBackground": "#00ff8818",
        "diffEditor.removedTextBackground": "#ef444418",
      },
    });
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-bg-glass border-b border-border-subtle">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-accent-blue">
              <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
            </svg>
            <span className="text-text-primary text-xs font-semibold">
              {showDiff ? "Diff View" : "Reference vs Candidate"}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowDiff(false)}
            className={`pill-btn text-xs py-1 px-3 ${
              !showDiff
                ? "bg-accent-green/20 text-accent-green border border-accent-green/40"
                : "pill-btn-ghost"
            }`}
          >
            Side by Side
          </button>
          <button
            onClick={() => setShowDiff(true)}
            className={`pill-btn text-xs py-1 px-3 ${
              showDiff
                ? "bg-accent-green/20 text-accent-green border border-accent-green/40"
                : "pill-btn-ghost"
            }`}
          >
            Diff
          </button>
        </div>
      </div>

      {/* Iteration Slider */}
      {snapshots && snapshots.length > 0 && (
        <IterationSlider
          snapshots={snapshots}
          currentIteration={effectiveIteration}
          onChange={handleIterChange}
        />
      )}

      {/* Code panels */}
      <div className="flex-1 min-h-0">
        {showDiff ? (
          <MonacoDiffEditor
            original={prevSnapshotData?.code || "// Previous iteration not available"}
            modified={snapshotData?.code || "// Loading..."}
            language="python"
            theme="echolokernel-dark"
            options={editorOptions}
            beforeMount={monacoThemeConfig}
          />
        ) : (
          <div className="flex h-full">
            {/* Reference pane */}
            <div className="flex-1 flex flex-col border-r border-border-subtle">
              <div className="px-4 py-2 bg-bg-glass border-b border-border-subtle">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-accent-blue" />
                  <span className="label-muted text-[10px]">
                    Reference PyTorch Kernel
                  </span>
                </div>
              </div>
              <div className="flex-1 min-h-0">
                <MonacoEditor
                  value={referenceData?.code || "// Loading reference code..."}
                  language="python"
                  theme="echolokernel-dark"
                  options={editorOptions}
                  beforeMount={monacoThemeConfig}
                />
              </div>
            </div>

            {/* Candidate pane */}
            <div className="flex-1 flex flex-col">
              <div className="px-4 py-2 bg-bg-glass border-b border-border-subtle">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-accent-green" />
                  <span className="label-muted text-[10px]">
                    Candidate Triton Kernel (Iteration {effectiveIteration})
                  </span>
                </div>
              </div>
              <div className="flex-1 min-h-0">
                <MonacoEditor
                  value={snapshotData?.code || "// Loading candidate code..."}
                  language="python"
                  theme="echolokernel-dark"
                  options={editorOptions}
                  beforeMount={monacoThemeConfig}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
