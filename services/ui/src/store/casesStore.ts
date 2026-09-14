import { create } from 'zustand';
import type { Case, TimelineEvent } from '@/lib/api';
import api from '@/lib/api';

interface CasesState {
  cases: Case[];
  activeCase: Case | null;
  timeline: TimelineEvent[];
  loading: boolean;
  error: string | null;
  fetchCases: () => Promise<void>;
  fetchCase: (id: string) => Promise<void>;
  fetchTimeline: (caseId: string) => Promise<void>;
  setActiveCase: (c: Case | null) => void;
}

export const useCasesStore = create<CasesState>((set) => ({
  cases: [],
  activeCase: null,
  timeline: [],
  loading: false,
  error: null,

  fetchCases: async () => {
    set({ loading: true, error: null });
    try {
      const cases = await api.getCases();
      set({ cases, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  fetchCase: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const activeCase = await api.getCase(id);
      set({ activeCase, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  fetchTimeline: async (caseId: string) => {
    set({ loading: true, error: null });
    try {
      const timeline = await api.getCaseTimeline(caseId);
      set({ timeline, loading: false });
    } catch (err) {
      set({ error: (err as Error).message, loading: false });
    }
  },

  setActiveCase: (c) => set({ activeCase: c }),
}));
