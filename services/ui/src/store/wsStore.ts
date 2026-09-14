import { create } from 'zustand';

export type PipelineEventType =
  | 'SPILL_DETECTED'
  | 'DRIFT_COMPLETE'
  | 'SUSPECT_RANKED'
  | 'NARRATIVE_READY'
  | 'CASE_FILE_READY'
  | 'PIPELINE_PROGRESS'
  | 'subscribed'
  | 'pong'
  | 'case_update'
  | 'new_spill'
  | 'vessel_alert'
  | 'drift_update'
  | 'heartbeat';

export interface WsMessage {
  type: PipelineEventType | string;
  payload: Record<string, unknown>;
}

interface WsState {
  connected: boolean;
  caseId: string | null;
  lastMessage: WsMessage | null;
  progress: { stage: string; pct_complete: number } | null;
  events: WsMessage[];
  connect: (caseId?: string) => void;
  disconnect: () => void;
}

let ws: WebSocket | null = null;
let currentCase: string | null = null;

function socketUrl(caseId?: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const base = `${proto}//${window.location.host}/api/v1`;
  if (caseId) return `${base}/ws/cases/${encodeURIComponent(caseId)}`;
  return `${base}/ws${currentCase ? `?case_id=${encodeURIComponent(currentCase)}` : ''}`;
}

export const useWsStore = create<WsState>((set, get) => ({
  connected: false,
  caseId: null,
  lastMessage: null,
  progress: null,
  events: [],

  connect: (caseId?: string) => {
    if (caseId) currentCase = caseId;
    if (ws?.readyState === WebSocket.OPEN) {
      if (caseId && caseId !== get().caseId) {
        ws.send(JSON.stringify({ type: 'subscribe', case_id: caseId }));
        set({ caseId });
      }
      return;
    }

    ws = new WebSocket(socketUrl(currentCase ?? undefined));

    ws.onopen = () => {
      set({ connected: true, caseId: currentCase });
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage;
        set((s) => {
          const events = [...s.events, msg].slice(-50);
          const progress =
            msg.type === 'PIPELINE_PROGRESS'
              ? {
                  stage: String((msg.payload as { stage?: unknown }).stage ?? ''),
                  pct_complete: Number(
                    (msg.payload as { pct_complete?: unknown }).pct_complete ?? 0,
                  ),
                }
              : s.progress;
          return { lastMessage: msg, events, progress };
        });
      } catch {
        // malformed message
      }
    };

    ws.onclose = () => set({ connected: false });
    ws.onerror = () => set({ connected: false });
  },

  disconnect: () => {
    ws?.close();
    ws = null;
    currentCase = null;
    set({ connected: false, caseId: null });
  },
}));
