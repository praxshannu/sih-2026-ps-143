// ── Color tokens (sync with tailwind.config.ts) ───────────────────────────
export const COLORS = {
  bg: '#080a0e',
  surface: '#0e1117',
  panel: '#121820',
  border: '#1e2530',
  amber: '#d97706',
  amberGlow: '#fbbf24',
  danger: '#dc2626',
  caution: '#ca8a04',
  nominal: '#15803d',
  // Semantic aliases. Several pages and the globe layers were written against
  // primary/success/warning; rather than leaving those broken (they render as
  // `undefined` at runtime, since esbuild does not typecheck), map them onto
  // the palette above.
  primary: '#3b82f6',
  success: '#15803d',
  warning: '#ca8a04',
  text: '#d1d5db',
  textHi: '#f3f4f6',
  muted: '#4b5563',
  link: '#3b82f6',
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
