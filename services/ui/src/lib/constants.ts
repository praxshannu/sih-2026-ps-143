// ── Color tokens (sync with tailwind.config.ts) ───────────────────────────
export const COLORS = {
  bg: '#060A11',
  surface: '#0B111D',
  panel: '#0F1726',
  border: '#1A273D',
  amber: '#F59E0B',
  amberGlow: '#FBBF24',
  danger: '#EF4444',
  caution: '#F59E0B',
  nominal: '#10B981',
  // Semantic aliases.
  primary: '#00D2FF',
  primaryHover: '#38BDF8',
  success: '#10B981',
  warning: '#F59E0B',
  text: '#CBD5E1',
  textHi: '#F8FAFC',
  muted: '#64748B',
  mutedHi: '#94A3B8',
  link: '#00D2FF',
} as const;

// ── Deck.gl / map fill colors (RGBA arrays) ───────────────────────────────
export const LAYER_COLORS = {
  spillFill: [180, 60, 10, 120] as [number, number, number, number],
  spillLine: [220, 80, 20, 220] as [number, number, number, number],
  driftFill: [180, 120, 10, 60] as [number, number, number, number],
  driftLine: [217, 119, 6, 160] as [number, number, number, number],
  originEllipse: [15, 120, 50, 40] as [number, number, number, number],
  vesselHigh: [220, 38, 38, 200] as [number, number, number, number],
  vesselMed: [202, 138, 4, 200] as [number, number, number, number],
  vesselLow: [21, 128, 61, 180] as [number, number, number, number],
} as const;

// ── API ───────────────────────────────────────────────────────────────────
export const API_BASE = '/api/v1';
export const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

// ── Map ───────────────────────────────────────────────────────────────────
// Dark-matter without labels — clean for ops use
export const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';

export const GLOBE_INITIAL_VIEW = {
  longitude: 73.0,
  latitude: 15.0,
  zoom: 4.5,
  pitch: 30,
  bearing: 0,
} as const;

// ── Confidence thresholds ─────────────────────────────────────────────────
export const CONFIDENCE = {
  high: 75,
  medium: 45,
} as const;

// ── Domain enumerations ───────────────────────────────────────────────────
export const CASE_STATUSES = ['ACTIVE', 'CLOSED', 'PENDING', 'ESCALATED'] as const;
export const SUSPECT_FACTORS = [
  'proximity', 'ais_gap', 'vessel_type', 'speed_anomaly', 'course_deviation', 'historical',
] as const;
export const VESSEL_TYPES = ['TANKER', 'CARGO', 'FISHING', 'PLEASURE', 'SAIL', 'UNKNOWN'] as const;

export type CaseStatus = (typeof CASE_STATUSES)[number];
export type SuspectFactor = (typeof SUSPECT_FACTORS)[number];
export type VesselType = (typeof VESSEL_TYPES)[number];

// ── System identity ───────────────────────────────────────────────────────
export const SYSTEM_ID = 'NTRO-SIH26143';
export const BUILD_VER = 'SENTINEL v1.0';
