import { create } from 'zustand';
import type { Case, TimelineEvent } from '@/lib/api';
import api from '@/lib/api';

export const DEMO_CASES: Case[] = [
  {
    id: 'case-wakashio-2020',
    title: 'MV WAKASHIO — MAURITIUS REEF GROUNDING',
    status: 'ACTIVE',
    created_at: new Date(Date.now() - 3600000 * 18).toISOString(),
    updated_at: new Date().toISOString(),
    spill: {
      id: 'spill-wakashio-01',
      detected_at: new Date(Date.now() - 3600000 * 18).toISOString(),
      centroid: { latitude: -20.44, longitude: 57.74 },
      area_km2: 1240.5,
      confidence: 94.2,
      polygon: [
        { latitude: -20.38, longitude: 57.68 },
        { latitude: -20.41, longitude: 57.80 },
        { latitude: -20.50, longitude: 57.84 },
        { latitude: -20.54, longitude: 57.76 },
        { latitude: -20.48, longitude: 57.66 },
        { latitude: -20.38, longitude: 57.68 },
      ],
      status: 'ACTIVE',
    },
    drift_forecast: {
      hours: 48,
      centroid: { latitude: -20.55, longitude: 57.65 },
      polygon: [
        { latitude: -20.44, longitude: 57.74 },
        { latitude: -20.52, longitude: 57.86 },
        { latitude: -20.66, longitude: 57.78 },
        { latitude: -20.62, longitude: 57.54 },
        { latitude: -20.44, longitude: 57.74 },
      ],
      probability_quantiles: [0.2, 0.5, 0.8],
      wind_speed_ms: 12.4,
      current_speed_ms: 0.85,
      sea_state: 'ROUGH',
    },
    suspects: [
      {
        rank: 1,
        score: 94.8,
        confidence: 91.4,
        narrative: 'Course anomaly 38 deg off designated shipping lane directly intersecting Pointe d Esny reef with 23 min AIS dark window prior to grounding.',
        scores: {
          proximity: 96,
          ais_gap: 92,
          vessel_type: 88,
          speed_anomaly: 95,
          course_deviation: 97,
          historical: 75,
        },
        vessel: {
          mmsi: '477218700',
          vessel_name: 'MV WAKASHIO',
          vessel_type: 'TANKER',
          waypoints: [
            { timestamp: new Date(Date.now() - 3600000 * 20).toISOString(), longitude: 57.95, latitude: -20.25, speed: 11.2, course: 220, heading: 220 },
            { timestamp: new Date(Date.now() - 3600000 * 19).toISOString(), longitude: 57.85, latitude: -20.35, speed: 10.8, course: 222, heading: 222 },
            { timestamp: new Date(Date.now() - 3600000 * 18).toISOString(), longitude: 57.74, latitude: -20.44, speed: 0.2, course: 225, heading: 225 },
          ],
          gaps: [
            {
              start: new Date(Date.now() - 3600000 * 18.5).toISOString(),
              end: new Date(Date.now() - 3600000 * 18).toISOString(),
              start_pos: { longitude: 57.79, latitude: -20.40 },
              end_pos: { longitude: 57.74, latitude: -20.44 },
            },
          ],
        },
      },
      {
        rank: 2,
        score: 38.2,
        confidence: 42.0,
        narrative: 'Container vessel transiting outer EEZ lane at normal cruising velocity.',
        scores: {
          proximity: 40,
          ais_gap: 15,
          vessel_type: 35,
          speed_anomaly: 20,
          course_deviation: 18,
          historical: 25,
        },
        vessel: {
          mmsi: '636019234',
          vessel_name: 'PACIFIC EXPLORER',
          vessel_type: 'CARGO',
          waypoints: [
            { timestamp: new Date(Date.now() - 3600000 * 21).toISOString(), longitude: 58.20, latitude: -20.10, speed: 18.5, course: 215, heading: 215 },
            { timestamp: new Date(Date.now() - 3600000 * 18).toISOString(), longitude: 57.90, latitude: -20.70, speed: 18.2, course: 215, heading: 215 },
          ],
          gaps: [],
        },
      },
    ],
    evidence: [
      {
        id: 'ev-01',
        title: 'Sentinel-1B C-Band SAR Pol-VV Ingestion',
        type: 'SAR_IMAGE',
        source: 'ESA CDSE Hub (Copernicus)',
        confidence: 96,
        timestamp: new Date(Date.now() - 3600000 * 18).toISOString(),
        description: 'Level-1 Ground Range Detected (GRD) amplitude anomaly indicating surface tension dampening across 1,240 km2.',
      },
      {
        id: 'ev-02',
        title: 'AIS Dark Window Log',
        type: 'AIS_DATA',
        source: 'Terrestrial & Satellite AIS Gateway',
        confidence: 91,
        timestamp: new Date(Date.now() - 3600000 * 18.5).toISOString(),
        description: '23.4-minute broadcast cessation while heading into coral reef boundary.',
      },
    ],
    narrative: 'Automated NTRO SAR detection identified high-contrast radar backscatter attenuation characteristic of heavy fuel oil (IFO-380) off Pointe d Esny reef. Backward Lagrangian drift inversion tightly correlates with MMSI 477218700 track deviation during 23-minute AIS silence.',
  },
  {
    id: 'case-arabian-sea-2026',
    title: 'ARABIAN SEA ANOMALOUS DISCHARGE',
    status: 'ESCALATED',
    created_at: new Date(Date.now() - 3600000 * 8).toISOString(),
    updated_at: new Date().toISOString(),
    spill: {
      id: 'spill-arabian-02',
      detected_at: new Date(Date.now() - 3600000 * 8).toISOString(),
      centroid: { latitude: 15.22, longitude: 73.45 },
      area_km2: 418.0,
      confidence: 88.5,
      polygon: [
        { latitude: 15.15, longitude: 73.35 },
        { latitude: 15.18, longitude: 73.55 },
        { latitude: 15.30, longitude: 73.58 },
        { latitude: 15.28, longitude: 73.38 },
        { latitude: 15.15, longitude: 73.35 },
      ],
      status: 'ACTIVE',
    },
    drift_forecast: {
      hours: 24,
      centroid: { latitude: 15.35, longitude: 73.60 },
      polygon: [
        { latitude: 15.22, longitude: 73.45 },
        { latitude: 15.25, longitude: 73.65 },
        { latitude: 15.42, longitude: 73.72 },
        { latitude: 15.38, longitude: 73.48 },
        { latitude: 15.22, longitude: 73.45 },
      ],
      probability_quantiles: [0.25, 0.55, 0.85],
      wind_speed_ms: 8.6,
      current_speed_ms: 0.45,
      sea_state: 'MODERATE',
    },
    suspects: [
      {
        rank: 1,
        score: 91.2,
        confidence: 88.0,
        narrative: 'Crude tanker altered ballast discharge pattern along Mumbai-Goa shipping corridor.',
        scores: {
          proximity: 92,
          ais_gap: 85,
          vessel_type: 95,
          speed_anomaly: 88,
          course_deviation: 78,
          historical: 82,
        },
        vessel: {
          mmsi: '413229980',
          vessel_name: 'GULF GLORY',
          vessel_type: 'TANKER',
          waypoints: [
            { timestamp: new Date(Date.now() - 3600000 * 10).toISOString(), longitude: 73.20, latitude: 15.00, speed: 13.5, course: 340, heading: 340 },
            { timestamp: new Date(Date.now() - 3600000 * 8).toISOString(), longitude: 73.45, latitude: 15.22, speed: 8.2, course: 338, heading: 338 },
            { timestamp: new Date(Date.now() - 3600000 * 6).toISOString(), longitude: 73.60, latitude: 15.45, speed: 14.1, course: 342, heading: 342 },
          ],
          gaps: [],
        },
      },
    ],
    evidence: [
      {
        id: 'ev-03',
        title: 'Sentinel-1A SAR Detection Strip',
        type: 'SAR_IMAGE',
        source: 'ISRO / ESA Payload Integration',
        confidence: 89,
        timestamp: new Date(Date.now() - 3600000 * 8).toISOString(),
        description: 'Linear hydrocarbon signature stretching 28.4 nautical miles along tanker route.',
      },
    ],
    narrative: 'High-confidence linear discharge corridor identified in eastern Arabian Sea. Hydrodynamic backtracking aligns with transit of crude carrier MMSI 413229980 during speed reduction window.',
  },
];

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
  cases: DEMO_CASES,
  activeCase: DEMO_CASES[0],
  timeline: [],
  loading: false,
  error: null,

  fetchCases: async () => {
    set({ loading: true, error: null });
    try {
      const cases = await api.getCases();
      if (cases && cases.length > 0) {
        set({ cases, activeCase: cases[0], loading: false });
      } else {
        set({ cases: DEMO_CASES, activeCase: DEMO_CASES[0], loading: false });
      }
    } catch {
      // Fallback seamlessly to demo cases so the UI is never blank
      set({ cases: DEMO_CASES, activeCase: DEMO_CASES[0], loading: false });
    }
  },

  fetchCase: async (id: string) => {
    set({ loading: true, error: null });
    try {
      const activeCase = await api.getCase(id);
      set({ activeCase, loading: false });
    } catch {
      const fallback = DEMO_CASES.find((c) => c.id === id) || DEMO_CASES[0];
      set({ activeCase: fallback, loading: false });
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
