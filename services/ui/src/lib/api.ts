import axios from 'axios';
import { API_BASE } from './constants';

export interface GeoPoint {
  longitude: number;
  latitude: number;
}

export interface SpillEvent {
  id: string;
  detected_at: string;
  centroid: GeoPoint;
  area_km2: number;
  confidence: number;
  polygon: GeoPoint[];
  status: 'ACTIVE' | 'DISPERSED' | 'RECOVERED';
}

export interface DriftForecast {
  hours: number;
  centroid: GeoPoint;
  polygon: GeoPoint[];
  probability_quantiles: number[];
  wind_speed_ms: number;
  current_speed_ms: number;
  sea_state: string;
}

export interface AisTrack {
  mmsi: string;
  vessel_name: string;
  vessel_type: string;
  waypoints: Array<{
    timestamp: string;
    longitude: number;
    latitude: number;
    speed: number;
    course: number;
    heading: number;
  }>;
  gaps: Array<{
    start: string;
    end: string;
    start_pos: GeoPoint;
    end_pos: GeoPoint;
  }>;
}

export interface Suspect {
  rank: number;
  vessel: AisTrack;
  score: number;
  scores: {
    proximity: number;
    ais_gap: number;
    vessel_type: number;
    speed_anomaly: number;
    course_deviation: number;
    historical: number;
  };
  confidence: number;
  narrative: string;
}

export interface Case {
  id: string;
  title: string;
  status: 'ACTIVE' | 'CLOSED' | 'PENDING' | 'ESCALATED';
  created_at: string;
  updated_at: string;
  spill: SpillEvent;
  suspects: Suspect[];
  drift_forecast: DriftForecast;
  evidence: Evidence[];
  narrative: string;
}

export interface Evidence {
  id: string;
  type: 'SAR_IMAGE' | 'AIS_DATA' | 'WEATHER' | 'SATELLITE' | 'MANIFEST';
  title: string;
  description: string;
  timestamp: string;
  confidence: number;
  source: string;
  url: string;
}

export interface TimelineEvent {
  id: string;
  timestamp: string;
  type: 'SPILL_DETECTED' | 'AIS_GAP' | 'VESSEL_SIGHTING' | 'SAR_PASS' | 'WEATHER_CHANGE' | 'SUSPECT_FLAGGED';
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  related_mmsi?: string;
}

export interface VesselProfile {
  mmsi: string;
  imo: string;
  name: string;
  flag: string;
  vessel_type: string;
  gross_tonnage: number;
  length: number;
  beam: number;
  built: number;
  owner: string;
  operator: string;
  registered_address: string;
  sanctions_history: boolean;
  prior_incidents: number;
  risk_score: number;
  last_known_position: GeoPoint;
  tracks: AisTrack[];
}

export interface PipelineSuspect {
  mmsi: string;
  vessel_name: string;
  composite_score: number;
  fuzzy_score: number;
  xgb_score: number | null;
  xgb_method: string;
  shap_breakdown: Record<string, number>;
  confidence_lower: number;
  confidence_upper: number;
  score_proximity: number;
  score_temporal: number;
  score_trajectory: number;
  score_anomaly: number;
  score_vessel_type: number;
  ais_gap_minutes: number;
  is_dark_vessel: boolean;
  rank: number;
  /**
   * Last-known fix, when the backend has one.
   *
   * The scoring pipeline reduces a vessel to `min_distance_nm` and does not
   * currently carry lat/lon through to `ScoreResult`, so these are usually
   * absent. They are declared so the UI can fly to a suspect when the
   * position IS available — and, just as importantly, can tell the difference
   * between "no position" and "position 0,0". Never guess a coordinate here.
   */
  latitude?: number | null;
  longitude?: number | null;
}

export interface PipelineDrift {
  case_id: string;
  backward?: {
    origin_ellipse: {
      center_lon: number;
      center_lat: number;
      semi_major_km: number;
      semi_minor_km: number;
      orientation_deg: number;
    };
    forcing_source: string;
    xgb_residual_applied: boolean;
    xgb_correction_m: Record<string, number>;
  };
  forward?: unknown;
  shoreline_risk?: unknown;
}

export interface LlmStatus {
  provider: string;
  model: string;
  available: boolean;
  latency_ms: number;
}

export interface PipelineTriggerResult {
  case_id: string;
  current_stage: string | null;
  stages_completed: string[];
  stages_failed: string[];
  queued: boolean;
  detail: string | null;
}

// ── Copernicus archive browser ───────────────────────────────────────────
export interface ArchiveScene {
  id: string;
  name: string;
  start: string;
  size_mb: number;
  footprint: GeoJSON.Geometry | null;
  platform: string;
  mode: string;
  product_type: string;
}

export interface ArchiveSearchResult {
  count: number;
  bbox: number[];
  window: string[];
  scenes: ArchiveScene[];
  source: string;
}

export interface ArchiveIngestResult {
  key: string;
  label: string;
  bbox: number[];
  time_window: string[];
  bands: string[];
  width: number;
  height: number;
  bytes: number;
  sha256: string;
  scene_name: string | null;
  fetched_utc: string;
  source: string;
  stats: {
    vv_db_p05: number;
    vv_db_median: number;
    vv_db_p95: number;
    valid_fraction: number;
  };
}

export interface ArchiveQuery {
  bbox: [number, number, number, number];
  start: string;
  end: string;
  product_type?: string;
  platform?: string;
  mode?: string;
  top?: number;
}

// ── AIS provenance ──────────────────────────────────────────────────────
/**
 * Every vessel payload carries one of these two provenance values. Anything
 * tagged `synthetic_mock` is SIMULATED — the UI is contractually required to
 * render `disclaimer` before a single synthetic track reaches the map.
 */
export type AisProvenance = 'live_terrestrial' | 'synthetic_mock';

export interface AisDisclaimer {
  severity: 'warning';
  provenance: AisProvenance;
  title: string;
  message: string;
  short: string;
}

/** Pre-flight verdict: does this AOI have real receivers, or will it be faked? */
export interface AisCoverage {
  bbox: number[];
  mode: 'live_terrestrial' | 'synthetic_only';
  use_synthetic: boolean;
  coastal_carveouts: number[][];
  basis: string;
  provenance: AisProvenance;
  notice: string | null;
  disclaimer: AisDisclaimer | null;
}

export interface AisGap {
  start: string;
  end: string | null;
  provenance: AisProvenance;
}

export interface AisFix {
  timestamp: string;
  longitude: number;
  latitude: number;
  sog: number;
  cog: number;
  heading: number;
  nav_status: string;
  source: string;
  provenance: AisProvenance;
}

export interface AisVessel {
  mmsi: string;
  name: string;
  vessel_type: string;
  flag: string;
  imo: string;
  track: AisFix[];
  gaps: AisGap[];
  provenance: AisProvenance;
}

export interface AisResult {
  provenance: AisProvenance;
  is_synthetic: boolean;
  notice: string | null;
  disclaimer: AisDisclaimer | null;
  coverage: AisCoverage;
  reason?: string;
  bbox: number[];
  window?: string[];
  count: number;
  vessels: AisVessel[];
  acknowledged?: boolean;
}

/** Resolved wind forcing. `synthetic_constant` means the numbers are invented. */
export type WindSource = 'era5' | 'gfs' | 'synthetic_constant';

// ── Drift attribution (OpenDrift backward hindcast) ────────────────────
/**
 * The 2-sigma PCA ellipse around the backtracked particle cluster.
 * `semi_*` are 2-sigma axes (~95%); p50/p95 are quantile radii from centre.
 */
export interface OriginEllipse {
  center_lon: number;
  center_lat: number;
  semi_major_km: number;
  semi_minor_km: number;
  orientation_deg: number;
  n_particles: number;
  p50_radius_km: number;
  p95_radius_km: number;
}

/** One vessel scored against the origin ellipse. */
export interface AttributionSuspect {
  mmsi: string;
  name: string;
  flag: string | null;
  distance_to_origin_km: number;
  inside_p95: boolean;
  inside_p50: boolean;
  age_h: number | null;
  wind_flag: string;
  provenance: AisProvenance | string;
}

export interface AttributionResult {
  run_utc: string;
  detection_time: string;
  detection_centroid: [number, number];
  detection_area_km2: number;
  /** Which forcing actually ran — the UI must label synthetic runs loudly. */
  wind_source: WindSource;
  current_source: 'cmems' | 'synthetic_constant';
  /** max |∇·K| seen across the run (Well-Mixed Criterion diagnostic). */
  wmc_divergence_max: number;
  config: Record<string, unknown>;
  origin: OriginEllipse;
  suspect_vessels: AttributionSuspect[];
  notes: string[];
}

// ── Tier-A detection summary ──────────────────────────────────────────
export interface DetectionSummary {
  scene_id: string;
  blob_candidates?: number;
  polygons_kept: number;
  best_confidence: number;
  wind_flag: string;
  ran_utc?: string | null;
  /** Real SAR acquisition instant — anchors the backward hindcast. */
  acquisition_time?: string | null;
  top_area_km2?: number | null;
  /** [lon, lat] of the largest surviving polygon. */
  top_centroid?: [number, number] | null;
  top_confidence?: number | null;
  top_confidence_low?: number | null;
  top_confidence_high?: number | null;
  top_age_hours_fay?: number | null;
  error?: string;
}

// ── Forward drift forecast (shoreline impact) ─────────────────────────
export interface ForecastConeStep {
  hours_ahead: number;
  valid_time: string;
  center_lon: number;
  center_lat: number;
  p50_radius_km: number;
  p95_radius_km: number;
  n_active: number;
  n_stranded: number;
}

export interface ForecastResult {
  run_utc: string;
  origin_time: string;
  origin_lon: number;
  origin_lat: number;
  duration_h: number;
  wind_source: WindSource;
  current_source: 'cmems' | 'synthetic_constant';
  config: Record<string, unknown>;
  cone: ForecastConeStep[];
  final_center: [number, number];
  /** Cumulative: beached at ANY point, not just at the final step. */
  stranded_fraction: number;
  first_stranding_h: number | null;
  notes: string[];
}

export interface ForecastRequest {
  origin_lon: number;
  origin_lat: number;
  origin_time: string;
  seed_radius_km?: number;
  duration_h?: number;
  use_era5?: boolean;
  use_cmems?: boolean;
  forcing?: 'auto' | 'era5' | 'gfs';
  use_landmask?: boolean;
  bbox?: [number, number, number, number];
  n_members?: number;
}

export interface AttributionRequest {
  detection_lon: number;
  detection_lat: number;
  detection_area_km2: number;
  detection_time: string;
  duration_h?: number;
  vessels?: Array<{
    mmsi: string | number;
    name: string;
    longitude: number;
    latitude: number;
    flag?: string;
    provenance?: string;
  }>;
  use_era5?: boolean;
  use_cmems?: boolean;
  /** auto = GFS when the window is recent (fast), ERA5 for older. */
  forcing?: 'auto' | 'era5' | 'gfs';
  bbox?: [number, number, number, number];
  n_members?: number;
  seed?: number;
}

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

const iso = (v: string) => (v.endsWith('Z') ? v : `${v}:00Z`);

/**
 * An AOI travels as four NAMED axes, never as a positional
 * "west,south,east,north" string. A lat-first string is made of four numbers
 * that are all legal in either slot, so no validator can detect the swap: the
 * AOI is silently relocated and the AIS real/synthetic verdict flips with it.
 * Named axes cannot be misordered.
 */
type Bbox = [number, number, number, number];

const axisParams = (b: Bbox) => ({
  min_lon: b[0],
  min_lat: b[1],
  max_lon: b[2],
  max_lat: b[3],
});

const axisQuery = (b: Bbox) =>
  `min_lon=${b[0]}&min_lat=${b[1]}&max_lon=${b[2]}&max_lat=${b[3]}`;

export const api = {
  getCases: () => client.get<Case[]>('/cases').then((r) => r.data),
  getCase: (id: string) => client.get<Case>(`/cases/${id}`).then((r) => r.data),
  getCaseTimeline: (id: string) =>
    client.get<TimelineEvent[]>(`/cases/${id}/timeline`).then((r) => r.data),
  getCaseSuspects: (id: string) =>
    client.get<Suspect[]>(`/cases/${id}/suspects`).then((r) => r.data),
  // DEAD ROUTES (verified 2026-09-14 against the gateway's 37 paths):
  // neither /vessels/{mmsi} nor /vessels/{mmsi}/tracks is routed — both 404.
  // Per-vessel lookups go through /cases/{case_id}/vessels/... instead.
  getVessel: (mmsi: string) =>
    client.get<VesselProfile>(`/vessels/${mmsi}`).then((r) => r.data),
  getVesselTracks: (mmsi: string) =>
    client.get<AisTrack[]>(`/vessels/${mmsi}/tracks`).then((r) => r.data),
  getDriftForecast: (caseId: string) =>
    client.get<DriftForecast>(`/cases/${caseId}/drift`).then((r) => r.data),
  getSpillEvents: () =>
    client.get<SpillEvent[]>('/spills').then((r) => r.data),
  // Live pipeline results (Redis-backed cache, 404 until the pipeline runs)
  getPipelineSuspects: (caseId: string) =>
    client
      .get<{ case_id: string; suspects: PipelineSuspect[]; total: number }>(
        `/cases/${caseId}/suspects`,
      )
      .then((r) => r.data),
  getPipelineDrift: (caseId: string) =>
    client.get<PipelineDrift>(`/drift/forecast/${caseId}`).then((r) => r.data),
  getLlmStatus: () => client.get<LlmStatus>('/llm/status').then((r) => r.data),
  triggerPipeline: (caseId: string, stages?: string[], payload?: Record<string, unknown>) =>
    client
      .post<PipelineTriggerResult>('/pipeline/trigger', { case_id: caseId, stages, payload })
      .then((r) => r.data),

  // ── Archive browser (proxied from the ingest service → Copernicus CDSE) ──
  getArchiveHealth: () =>
    client
      .get<{ configured: boolean; catalogue: string; detail: string | null }>('/archive/health')
      .then((r) => r.data),
  searchArchive: (q: ArchiveQuery) =>
    client
      .get<ArchiveSearchResult>('/archive/search', {
        params: {
          ...axisParams(q.bbox),
          start: iso(q.start),
          end: iso(q.end),
          product_type: q.product_type ?? 'GRD',
          platform: q.platform || undefined,
          mode: q.mode || undefined,
          top: q.top ?? 50,
        },
      })
      .then((r) => r.data),
  /** Direct image URL — usable as an <img src> or a MapLibre raster source. */
  archiveQuicklookUrl: (q: ArchiveQuery, size = 512) =>
    `${API_BASE}/archive/quicklook?${axisQuery(q.bbox)}&start=${encodeURIComponent(
      iso(q.start),
    )}&end=${encodeURIComponent(iso(q.end))}&size=${size}`,
  ingestArchive: (body: {
    bbox: [number, number, number, number];
    start: string;
    end: string;
    size?: number;
    label?: string;
    scene_name?: string;
  }) =>
    client
      .post<ArchiveIngestResult>('/archive/ingest', {
        ...axisParams(body.bbox),
        start: body.start,
        end: body.end,
        size: body.size,
        label: body.label,
        scene_name: body.scene_name,
      })
      .then((r) => r.data),

  // ── AIS provenance / synthetic fallback ────────────────────────────────
  /**
   * Call this the moment an AOI is set. It tells the operator BEFORE any data
   * is drawn whether the vessel layer will be real or simulated.
   */
  getAisCoverage: (bbox: [number, number, number, number]) =>
    client
      .get<AisCoverage>('/archive/ais/coverage', { params: axisParams(bbox) })
      .then((r) => r.data),
  getAis: (q: {
    bbox: [number, number, number, number];
    start: string;
    end: string;
    n_vessels?: number;
    seed?: number;
    acknowledge_synthetic?: boolean;
  }) =>
    client
      .get<AisResult>('/archive/ais', {
        params: {
          ...axisParams(q.bbox),
          start: iso(q.start),
          end: iso(q.end),
          n_vessels: q.n_vessels ?? 6,
          seed: q.seed ?? 20200725,
          acknowledge_synthetic: q.acknowledge_synthetic ?? false,
        },
      })
      .then((r) => r.data),

  // ── Deterministic Tier-A detector ──────────────────────────────────
  /**
   * One row per scene that has a persisted detection. `acquisition_time` +
   * `top_centroid` + `top_area_km2` are what anchor a backward drift
   * hindcast, so a row missing any of them is not backtrackable.
   */
  listDetections: () =>
    client
      .get<{
        count: number;
        sar_dir: string;
        results: DetectionSummary[];
      }>('/detect/deterministic/results')
      .then((r) => r.data),
  runAllDetections: (wind_speed_ms?: number) =>
    client
      .post<{
        count: number;
        results: DetectionSummary[];
      }>('/detect/deterministic/run_all', null, {
        params: wind_speed_ms !== undefined ? { wind_speed_ms } : {},
      })
      .then((r) => r.data),

  // ── Deterministic Tier-A detector ──────────────────────────────────
  getDetectHealth: () =>
    client
      .get<{
        ok: boolean;
        service: string;
        variant: 'deterministic' | 'ml';
        sar_dir: string;
        scene_count: number;
        ml_available: boolean;
        detector: string;
      }>('/detect/health')
      .then((r) => r.data),

  // ── Drift attribution (OpenDrift backward hindcast) ────────────────
  getDriftHealth: () =>
    client
      .get<{ ok: boolean; service: string; opendrift: string }>('/drift/health')
      .then((r) => r.data),
  /**
   * Backtrack a detection to its likely origin. EXPECT THIS TO BE SLOW:
   * an ERA5 download plus a 64-member ensemble is routinely 60-180 s, so the
   * request gets its own generous timeout instead of inheriting the default.
   */
  runAttribution: (req: AttributionRequest) =>
    client
      .post<AttributionResult>('/drift/attribution', req, { timeout: 600_000 })
      .then((r) => r.data),
  /** Forward projection + shoreline impact. Same slow-forcing caveat. */
  runForecast: (req: ForecastRequest) =>
    client
      .post<ForecastResult>('/drift/forecast', req, { timeout: 600_000 })
      .then((r) => r.data),
};

export default api;
