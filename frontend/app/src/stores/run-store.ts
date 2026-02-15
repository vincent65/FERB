import { create } from "zustand";
import type { IterationData } from "@/lib/types";

interface RunStore {
  // Currently selected iteration
  selectedIteration: number | null;
  setSelectedIteration: (iter: number | null) => void;

  // Currently active problem tab
  activeProblemId: number | null;
  setActiveProblemId: (pid: number | null) => void;

  // Bottom panel state
  activeTab: string;
  setActiveTab: (tab: string) => void;
  bottomPanelOpen: boolean;
  setBottomPanelOpen: (open: boolean) => void;

  // Code view mode
  showDiff: boolean;
  setShowDiff: (show: boolean) => void;

  // Real-time iteration data (from WebSocket)
  liveIterations: IterationData[];
  addLiveIteration: (iter: IterationData) => void;
  clearLiveIterations: () => void;
}

export const useRunStore = create<RunStore>((set) => ({
  selectedIteration: null,
  setSelectedIteration: (iter) => set({ selectedIteration: iter }),

  activeProblemId: null,
  setActiveProblemId: (pid) => set({ activeProblemId: pid }),

  activeTab: "reasoning",
  setActiveTab: (tab) => set({ activeTab: tab }),
  bottomPanelOpen: true,
  setBottomPanelOpen: (open) => set({ bottomPanelOpen: open }),

  showDiff: false,
  setShowDiff: (show) => set({ showDiff: show }),

  liveIterations: [],
  addLiveIteration: (iter) =>
    set((state) => ({ liveIterations: [...state.liveIterations, iter] })),
  clearLiveIterations: () => set({ liveIterations: [] }),
}));
