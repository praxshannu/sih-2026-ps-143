/**
 * Archive — the in-platform Copernicus Browser.
 *
 * Same two APIs as the Copernicus Browser itself:
 *   1. CDSE OData catalogue  -> authoritative scene metadata + footprints
 *   2. CDSE Sentinel Hub Process API -> server-side rendered SAR quicklook
 *
 * No data is mocked. When the catalogue returns no scenes, the map is empty
 * and the list is empty. When Copernicus 502s, the page shows its message.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Search,
  Download,
  Crosshair,
  Eye,
  EyeOff,
  Loader2,
  TriangleAlert,
  CheckCircle2,
  RefreshCw,
  Ship,
  Scan,
} from 'lucide-react';
import Map, { Layer, Source, type MapRef } from 'react-map-gl/maplibre';
import type { StyleSpecification } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

import {
  api,
  type ArchiveScene,
  type ArchiveQuery,
  type AisCoverage,
  type AisResult,
  type AttributionResult,
  type ForecastResult,
} from '@/lib/api';
import { COLORS, MAP_STYLE, SYSTEM_ID } from '@/lib/constants';
import SyntheticDataNotice from '@/components/shared/SyntheticDataNotice';
import AttributionPanel, { type DetectionSeed } from '@/components/investigation/AttributionPanel';

const BASE_STYLE = MAP_STYLE as unknown as string | StyleSpecification;

// ── Preset AOIs (Indian Ocean + Arabian Sea focus) ────────────────────────
const PRESETS: { name: string; bbox: [number, number, number, number] }[] = [
  { name: 'MUMBAI OFFSHORE', bbox: [72.0, 15.0, 73.0, 16.0] },
  { name: 'WAKASHIO ZONE', bbox: [57.6, -21.0, 58.2, -20.4] },
  { name: 'STRAIT OF HORMUZ', bbox: [56.0, 26.0, 57.5, 27.5] },
  { name: 'GULF OF GUINEA', bbox: [-5.0, 3.0, 9.0, 8.0] },
  { name: 'BAY OF BENGAL', bbox: [85.0, 16.0, 92.0, 22.0] },
];

const FOOTPRINT_STYLE: any = {
  type: 'line',
  paint: {
    'line-color': COLORS.amber,
    'line-width': 1,
    'line-opacity': 0.55,
  },
};
const FOOTPRINT_FILL: any = {
  type: 'fill',
  paint: { 'fill-color': COLORS.amber, 'fill-opacity': 0.05 },
};
const SELECTED_STYLE: any = {
  type: 'line',
  paint: { 'line-color': COLORS.amberGlow, 'line-width': 2 },
};
const SELECTED_FILL: any = {
  type: 'fill',
  paint: { 'fill-color': COLORS.amber, 'fill-opacity': 0.12 },
};
const AOI_STYLE: any = {
  type: 'line',
  paint: {
    'line-color': '#3b82f6',
    'line-width': 1.2,
    'line-dasharray': [2, 2],
  },
};
const AOI_FILL: any = {
  type: 'fill',
  paint: { 'fill-color': '#3b82f6', 'fill-opacity': 0.05 },
};
// Synthetic tracks are drawn in hatched amber — visually distinct from any
// real feed so a screenshot can never be mistaken for live data.
const AIS_TRACK_STYLE: any = {
  type: 'line',
  paint: { 'line-color': COLORS.amber, 'line-width': 1, 'line-opacity': 0.7 },
};
const AIS_TRACK_STYLE_SYNTHETIC: any = {
  type: 'line',
  paint: {
    'line-color': COLORS.amber,
    'line-width': 1,
    'line-opacity': 0.7,
    'line-dasharray': [3, 2],
  },
};
const AIS_POINT_STYLE: any = {
  type: 'circle',
  paint: {
    'circle-radius': 2,
    'circle-color': COLORS.amberGlow,
    'circle-stroke-width': 0.5,
    'circle-stroke-color': '#080a0e',
  },
};
// Backtracked origin — green fill so it never reads as another slick polygon.
const ORIGIN_FILL: any = {
  type: 'fill',
  paint: { 'fill-color': '#15803d', 'fill-opacity': 0.14 },
};
const ORIGIN_LINE: any = {
  type: 'line',
  paint: { 'line-color': '#22c55e', 'line-width': 1.2, 'line-dasharray': [3, 2] },
};
// Dashed connector from the origin back to where the slick was found.
const DRIFT_VECTOR_STYLE: any = {
  type: 'line',
  paint: {
    'line-color': '#22c55e',
    'line-width': 1,
    'line-opacity': 0.7,
    'line-dasharray': [1, 3],
  },
};
// Forward forecast track. Colour is data-driven off stranded_fraction:
// green (offshore) -> amber -> red (major beaching).
const FORECAST_TRACK_STYLE: any = {
  type: 'line',
  paint: {
    'line-color': ['get', 'color'],
    'line-width': 1.6,
    'line-opacity': 0.9,
  },
};

function fmtIsoUTC(s: string) {
  return s ? s.replace('T', ' ').replace('.000Z', 'Z').replace(/:\d{2}Z$/, 'Z') : '—';
}

function footprintFeature(scene: ArchiveScene): GeoJSON.Feature<GeoJSON.Polygon | GeoJSON.MultiPolygon> | null {
  const fp = scene.footprint;
  if (!fp || fp.type === 'GeometryCollection') return null;
  return {
    type: 'Feature',
    geometry: fp as GeoJSON.Polygon | GeoJSON.MultiPolygon,
    properties: { id: scene.id, name: scene.name, start: scene.start },
  };
}

function bboxPolygon(b: [number, number, number, number]): GeoJSON.Feature<GeoJSON.Polygon> {
  const [w, s, e, n] = b;
  return {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [[
        [w, s], [e, s], [e, n], [w, n], [w, s],
      ]],
    },
    properties: {},
  };
}

export default function Archive() {
  const mapRef = useRef<MapRef | null>(null);

  const [query, setQuery] = useState<ArchiveQuery>({
    bbox: PRESETS[1].bbox,
    // Full ISO with Z — the backend validator and Copernicus both require it.
    start: '2020-08-01T00:00:00Z',
    end: '2020-08-31T23:59:59Z',
    product_type: 'GRD',
    platform: undefined,
    mode: 'IW',
    top: 50,
  });

  const [scenes, setScenes] = useState<ArchiveScene[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showQuicklook, setShowQuicklook] = useState(true);
  const [drawMode, setDrawMode] = useState(false);
  const [drawStart, setDrawStart] = useState<[number, number] | null>(null);
  const [drawEnd, setDrawEnd] = useState<[number, number] | null>(null);
  const [viewport, setViewport] = useState({
    longitude: (PRESETS[1].bbox[0] + PRESETS[1].bbox[2]) / 2,
    latitude: (PRESETS[1].bbox[1] + PRESETS[1].bbox[3]) / 2,
    zoom: 8,
  });

  const [health, setHealth] = useState<{ configured: boolean; catalogue: string; detail: string | null } | null>(null);
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingestLog, setIngestLog] = useState<string | null>(null);

  // ── Tier-A detection panel ──────────────────────────────────────────
  const [detectList, setDetectList] = useState<Awaited<ReturnType<typeof api.listDetections>> | null>(null);
  const [detectLoading, setDetectLoading] = useState(false);
  const refreshDetections = useCallback(async () => {
    setDetectLoading(true);
    try { setDetectList(await api.listDetections()); }
    catch { /* leave previous state */ }
    finally { setDetectLoading(false); }
  }, []);
  useEffect(() => { refreshDetections(); }, [refreshDetections]);

  // ── Backward attribution (OpenDrift) ─────────────────────────────────
  // Clicking a detection row in the Tier-A panel seeds the hindcast. We keep
  // the picked scene_id (not the object) so a refresh of the detection list
  // re-derives the seed from fresh data instead of a stale copy.
  const [attrSceneId, setAttrSceneId] = useState<string | null>(null);
  const [attribution, setAttribution] = useState<AttributionResult | null>(null);
  const [forecast, setForecast] = useState<ForecastResult | null>(null);

  const attrSeed: DetectionSeed | null = useMemo(() => {
    if (!attrSceneId || !detectList) return null;
    const r = detectList.results.find((d) => d.scene_id === attrSceneId);
    // The hindcast needs a real acquisition instant; without one it would
    // backtrack from "now" and fetch forcing for the wrong year entirely.
    if (!r?.acquisition_time || !r.top_centroid || !r.top_area_km2) return null;
    return {
      scene_id: r.scene_id,
      detection_lon: r.top_centroid[0],
      detection_lat: r.top_centroid[1],
      detection_area_km2: r.top_area_km2,
      detection_time: r.acquisition_time,
      best_confidence: r.best_confidence,
    };
  }, [attrSceneId, detectList]);

  // ── AIS layer ────────────────────────────────────────────────────────
  const [showVessels, setShowVessels] = useState(false);
  const [aisCoverage, setAisCoverage] = useState<AisCoverage | null>(null);
  const [ais, setAis] = useState<AisResult | null>(null);
  const [aisLoading, setAisLoading] = useState(false);
  const [aisError, setAisError] = useState<string | null>(null);
  const [noticeDismissed, setNoticeDismissed] = useState(false);

  // Vessels currently on the map are the suspect pool for the origin ellipse.
  // Declared after `ais` because it derives from it.
  const attrVessels = useMemo(() => {
    if (!ais) return [];
    return ais.vessels
      .map((v) => {
        const last = v.track[v.track.length - 1];
        if (!last) return null;
        return {
          mmsi: v.mmsi,
          name: v.name,
          longitude: last.longitude,
          latitude: last.latitude,
          flag: v.flag,
          provenance: v.provenance,
        };
      })
      .filter((v): v is NonNullable<typeof v> => v !== null);
  }, [ais]);

  // Probe Copernicus on first load so the page can render an honest health bar.
  useEffect(() => {
    api.getArchiveHealth().then(setHealth).catch((e) =>
      setHealth({ configured: false, catalogue: 'unreachable', detail: String(e) }),
    );
  }, []);

  // PRE-FLIGHT: the moment the AOI changes, ask whether real AIS exists here.
  // This warns the operator before any vessel data is requested, so an Indian
  // Ocean AOI announces its own synthetic status up front.
  useEffect(() => {
    let cancelled = false;
    setNoticeDismissed(false);
    setAis(null);
    setAisError(null);
    api
      .getAisCoverage(query.bbox)
      .then((c) => { if (!cancelled) setAisCoverage(c); })
      .catch(() => { if (!cancelled) setAisCoverage(null); });
    return () => { cancelled = true; };
  }, [query.bbox]);

  // Fetch tracks when the vessel layer is switched on.
  useEffect(() => {
    if (!showVessels) return;
    let cancelled = false;
    setAisLoading(true);
    setAisError(null);
    api
      .getAis({
        bbox: query.bbox,
        start: query.start,
        end: query.end,
        n_vessels: 6,
      })
      .then((r) => { if (!cancelled) setAis(r); })
      .catch((e: any) => {
        if (!cancelled) setAisError(e?.response?.data?.detail ?? String(e));
      })
      .finally(() => { if (!cancelled) setAisLoading(false); });
    return () => { cancelled = true; };
  }, [showVessels, query.bbox, query.start, query.end]);

  const runSearch = useCallback(async (q: ArchiveQuery) => {
    setLoading(true);
    setError(null);
    setSelectedId(null);
    try {
      const r = await api.searchArchive(q);
      setScenes(r.scenes);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
      setScenes([]);
    } finally {
      setLoading(false);
    }
  }, []);

  // First load
  useEffect(() => { runSearch(query); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  const selected = useMemo(() => scenes.find((s) => s.id === selectedId) ?? null, [scenes, selectedId]);

  // Quicklook URL: only when a scene is selected and the user opted in.
  const quicklookUrl = useMemo(() => {
    if (!selected || !showQuicklook) return null;
    return api.archiveQuicklookUrl(
      {
        bbox: query.bbox,
        start: selected.start,
        end: addOneSecond(selected.start),
        product_type: query.product_type,
      },
      512,
    );
  }, [selected, showQuicklook, query]);

  const features = useMemo(
    () => scenes.map(footprintFeature).filter(Boolean) as GeoJSON.Feature[],
    [scenes],
  );
  const selectedFeature = selected ? footprintFeature(selected) : null;

  // Vessel tracks: dashed amber when synthetic so a screenshot is self-labelling.
  const vesselGeo = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!ais || ais.vessels.length === 0) return null;
    const features: GeoJSON.Feature[] = [];
    for (const v of ais.vessels) {
      if (v.track.length < 2) continue;
      // Split the track at each AIS gap so a dark vessel's absence is visible
      // as a break in the line rather than an impossible straight jump.
      const gapStarts = new Set(v.gaps.map((g) => g.start));
      const gapEnds = new Set(v.gaps.map((g) => g.end));
      let seg: [number, number][] = [];
      for (const f of v.track) {
        if (gapStarts.has(f.timestamp)) {
          if (seg.length > 1) features.push(trackFeature(seg, v));
          seg = [];
          continue;
        }
        seg.push([f.longitude, f.latitude]);
        if (gapEnds.has(f.timestamp)) {
          if (seg.length > 1) features.push(trackFeature(seg, v));
          seg = [];
        }
      }
      if (seg.length > 1) features.push(trackFeature(seg, v));
    }
    return { type: 'FeatureCollection', features };
  }, [ais]);

  const vesselPoints = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!ais || ais.vessels.length === 0) return null;
    const features: GeoJSON.Feature[] = [];
    for (const v of ais.vessels) {
      const last = v.track[v.track.length - 1];
      if (!last) continue;
      features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [last.longitude, last.latitude] },
        properties: { mmsi: v.mmsi, name: v.name, type: v.vessel_type },
      });
    }
    return { type: 'FeatureCollection', features };
  }, [ais]);

  /**
   * Backtracked origin as a map polygon. The backend reports a 2-sigma PCA
   * ellipse (semi-axes in km + orientation clockwise from north); we sample it
   * into a lon/lat ring, converting km per degree at the ellipse's own
   * latitude so it isn't sheared.
   */
  const originGeo = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!attribution) return null;
    const { center_lon, center_lat, semi_major_km, semi_minor_km, orientation_deg } =
      attribution.origin;
    if (!Number.isFinite(center_lon) || !Number.isFinite(center_lat)) return null;
    const a = Math.max(semi_major_km, 0.01);
    const b = Math.max(semi_minor_km, 0.01);
    const kmPerDegLat = 110.57;
    const kmPerDegLon = 111.32 * Math.max(Math.cos((center_lat * Math.PI) / 180), 1e-3);
    // Rotate from ellipse axes into east/north. orientation is clockwise from
    // north for the major axis, so build the ring in (north, east).
    const rad = (orientation_deg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const ring: [number, number][] = [];
    for (let i = 0; i < 64; i++) {
      const t = (i / 64) * 2 * Math.PI;
      const ex = a * Math.cos(t);
      const ey = b * Math.sin(t);
      const north = ex * cos - ey * sin;
      const east = ex * sin + ey * cos;
      ring.push([
        center_lon + east / kmPerDegLon,
        center_lat + north / kmPerDegLat,
      ]);
    }
    ring.push(ring[0]);
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [ring] },
          properties: {
            kind: 'origin_ellipse',
            n_particles: attribution.origin.n_particles,
            wind_source: attribution.wind_source,
          },
        },
      ],
    };
  }, [attribution]);

  /**
   * Forward forecast: the track the slick's centroid is predicted to follow,
   * drawn from the detection out to the final projected centre. Colour ramps
   * green -> amber -> red as the beached fraction climbs, so the line itself
   * carries the shoreline risk without reading the panel.
   */
  const forecastTrackGeo = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!forecast || forecast.cone.length < 2) return null;
    const coords: [number, number][] = forecast.cone.map((c) => [c.center_lon, c.center_lat]);
    const risk = forecast.stranded_fraction;
    const color = risk >= 0.5 ? '#dc2626' : risk > 0 ? '#ca8a04' : '#22c55e';
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: coords },
          properties: {
            kind: 'forecast_track',
            color,
            stranded_fraction: risk,
            first_stranding_h: forecast.first_stranding_h,
          },
        },
      ],
    };
  }, [forecast]);

  /** Dashed connector: origin -> where the slick was actually seen. */
  const driftVectorGeo = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!attribution) return null;
    const [dlon, dlat] = attribution.detection_centroid;
    return {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: [
              [attribution.origin.center_lon, attribution.origin.center_lat],
              [dlon, dlat],
            ],
          },
          properties: { kind: 'drift_vector' },
        },
      ],
    };
  }, [attribution]);

  const updateBbox = (b: [number, number, number, number]) =>
    setQuery((q) => ({ ...q, bbox: b }));

  // ── Draw a rectangle on the map to set the AOI ─────────────────────────
  const onMapClick = (e: any) => {
    if (!drawMode) return;
    const ll = e.lngLat;
    if (!drawStart) setDrawStart([ll.lng, ll.lat]);
    else {
      setDrawEnd([ll.lng, ll.lat]);
      const [w, e2] = orderX(drawStart[0], ll.lng);
      const [s, n] = orderY(drawStart[1], ll.lat);
      updateBbox([w, s, e2, n]);
      setDrawMode(false);
      setDrawStart(null);
      setDrawEnd(null);
    }
  };
  const onMapMove = (e: any) => {
    if (drawMode && drawStart) setDrawEnd([e.lngLat.lng, e.lngLat.lat]);
  };

  const useCurrentView = () => {
    const b = mapRef.current?.getMap()?.getBounds();
    if (!b) return;
    const w = b.getWest(), e = b.getEast(), s = b.getSouth(), n = b.getNorth();
    updateBbox([w, s, e, n]);
  };

  const sendToDetection = async () => {
    if (!selected) return;
    setIngesting(true);
    setIngestLog(null);
    setError(null);
    try {
      const r = await api.ingestArchive({
        bbox: query.bbox,
        start: selected.start,
        end: addOneSecond(selected.start),
        size: 1024,
        label: `archive_${selected.name.slice(0, 20).replace(/[^A-Z0-9]/gi, '_').toLowerCase()}`,
        scene_name: selected.name,
      });
      setIngestLog(
        `Saved ${(r.bytes / 1048576).toFixed(1)} MB → data/sar/${r.key}.tif  (sha ${r.sha256.slice(0, 12)}…)`,
      );
      // Trigger a refresh of the detection panel so this new scene appears.
      refreshDetections();
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
    } finally {
      setIngesting(false);
    }
  };

  return (
    <div className="flex h-full">
      {/* ── LEFT: search rail ─────────────────────────────────────────── */}
      <aside className="flex w-72 flex-col border-r border-sentinel-border bg-sentinel-surface">
        <div className="ops-header">
          <Search size={12} className="text-sentinel-data" />
          <span className="ops-label">ARCHIVE QUERY</span>
          <span className="ml-auto font-mono text-2xs text-sentinel-muted-hi">{SYSTEM_ID}</span>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-3 text-2xs">
          <Section label="AREA OF INTEREST (W,S,E,N)">
            <div className="grid grid-cols-2 gap-1">
              {(['W', 'S', 'E', 'N'] as const).map((k) => {
                const idx = { W: 0, S: 1, E: 2, N: 3 }[k] as 0 | 1 | 2 | 3;
                return (
                  <div key={k} className="flex items-center gap-1">
                    <span className="font-mono text-2xs text-sentinel-muted-hi w-3">{k}</span>
                    <input
                      type="number"
                      step="0.01"
                      value={query.bbox[idx]}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        if (!Number.isFinite(v)) return;
                        const b = [...query.bbox] as [number, number, number, number];
                        b[idx] = v;
                        updateBbox(b);
                      }}
                      className="w-full bg-sentinel-surface border border-sentinel-border px-1.5 py-1 font-mono text-2xs tabular-nums text-sentinel-text-hi focus:border-sentinel-data focus:outline-none focus:ring-0"
                    />
                  </div>
                );
              })}
            </div>
            <div className="mt-1 flex gap-1">
              <button
                onClick={useCurrentView}
                className="flex-1 border border-sentinel-border bg-sentinel-surface px-2 py-1 font-mono text-2xs uppercase tracking-wider text-sentinel-text hover:border-sentinel-data hover:text-sentinel-data transition-colors"
              >
                <Crosshair size={10} className="mr-1 inline-block" /> Use View
              </button>
              <button
                onClick={() => { setDrawMode((v) => !v); setDrawStart(null); setDrawEnd(null); }}
                className={
                  'flex-1 border px-2 py-1 font-mono text-2xs uppercase tracking-wider transition-colors ' +
                  (drawMode
                    ? 'border-sentinel-data bg-sentinel-data-dim text-sentinel-data'
                    : 'border-sentinel-border bg-sentinel-surface text-sentinel-text hover:border-sentinel-data hover:text-sentinel-data')
                }
              >
                {drawMode ? 'Click corner' : 'Draw AOI'}
              </button>
            </div>
          </Section>

          <Section label="PRESETS">
            <div className="space-y-1">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  onClick={() => { updateBbox(p.bbox); }}
                  className="block w-full border border-sentinel-border bg-sentinel-surface px-2 py-1 text-left font-mono text-2xs text-sentinel-text hover:border-sentinel-data hover:text-sentinel-data transition-colors"
                >
                  {p.name}
                </button>
              ))}
            </div>
          </Section>

          <Section label="TIME WINDOW">
            <div className="space-y-1">
              <TimeInput label="FROM UTC" value={query.start} onChange={(v) => setQuery((q) => ({ ...q, start: v }))} />
              <TimeInput label="TO UTC" value={query.end} onChange={(v) => setQuery((q) => ({ ...q, end: v }))} />
            </div>
          </Section>

          <Section label="FILTERS">
            <div className="grid grid-cols-3 gap-1 text-2xs">
              <SelectField label="TYPE" value={query.product_type} options={['GRD', 'SLC', 'OCN']}
                onChange={(v) => setQuery((q) => ({ ...q, product_type: v }))} />
              <SelectField label="MODE" value={query.mode ?? 'ALL'} options={['ALL', 'IW', 'EW', 'WV', 'SM']}
                onChange={(v) => setQuery((q) => ({ ...q, mode: v }))} />
              <SelectField label="PLAT" value={query.platform ?? 'ALL'} options={['ALL', 'S1A', 'S1B', 'S1C', 'S1D']}
                onChange={(v) => setQuery((q) => ({ ...q, platform: v }))} />
            </div>
          </Section>

          <div className="pt-2">
            <button
              onClick={() => runSearch(query)}
              disabled={loading}
              className="w-full border border-sentinel-amber bg-sentinel-amber/10 px-3 py-2 font-mono text-2xs uppercase tracking-widest text-sentinel-amber hover:bg-sentinel-amber/20 disabled:opacity-50"
            >
              {loading ? <Loader2 size={11} className="mr-1 inline animate-spin" /> : <Search size={11} className="mr-1 inline" />}
              {loading ? 'Querying Copernicus' : 'Search Copernicus'}
            </button>
          </div>
        </div>

        <HealthBar health={health} />
      </aside>

      {/* ── CENTER: map ──────────────────────────────────────────────── */}
      <main className="relative flex-1 bg-sentinel-bg">
        <Map
          ref={mapRef}
          {...viewport}
          onMove={(e) => setViewport(e.viewState)}
          mapStyle={BASE_STYLE as any}
          style={{ width: '100%', height: '100%' }}
          onClick={onMapClick}
          onMouseMove={onMapMove}
          cursor={drawMode ? 'crosshair' : 'grab'}
        >
          {/* AOI rectangle (search) */}
          <Source id="aoi" type="geojson" data={bboxPolygon(query.bbox)}>
            <Layer {...AOI_FILL} />
            <Layer {...AOI_STYLE} />
          </Source>

          {/* In-progress draw */}
          {drawMode && drawStart && (
            <Source
              id="draw"
              type="geojson"
              data={drawEnd
                ? bboxPolygon(orderBbox([drawStart[0], drawStart[1], drawEnd[0], drawEnd[1]]))
                : { type: 'Feature', geometry: { type: 'Point', coordinates: drawStart }, properties: {} }}
            >
              <Layer {...AOI_FILL} />
              <Layer {...AOI_STYLE} />
            </Source>
          )}

          {/* Footprints */}
          {features.length > 0 && (
            <Source id="footprints" type="geojson" data={{ type: 'FeatureCollection', features }}>
              <Layer {...FOOTPRINT_FILL} />
              <Layer {...FOOTPRINT_STYLE} />
            </Source>
          )}

          {/* Selected footprint highlighted */}
          {selectedFeature && (
            <Source id="selected-fp" type="geojson" data={selectedFeature}>
              <Layer {...SELECTED_FILL} />
              <Layer {...SELECTED_STYLE} />
            </Source>
          )}

          {/* Quicklook raster (image overlay) */}
          {quicklookUrl && (
            <Source
              id="quicklook"
              type="image"
              coordinates={bboxToRasterCoords(query.bbox)}
              url={quicklookUrl}
            >
              <Layer id="quicklook-layer" type="raster" paint={{ 'raster-opacity': 0.85 }} />
            </Source>
          )}

          {/* Vessel tracks — dashed when synthetic */}
          {showVessels && vesselGeo && (
            <Source id="vessels" type="geojson" data={vesselGeo}>
              <Layer
                {...(ais?.is_synthetic ? AIS_TRACK_STYLE_SYNTHETIC : AIS_TRACK_STYLE)}
              />
            </Source>
          )}
          {showVessels && vesselPoints && (
            <Source id="vessel-pts" type="geojson" data={vesselPoints}>
              <Layer {...AIS_POINT_STYLE} />
            </Source>
          )}

          {/* Backtracked origin ellipse + the drift vector that produced it */}
          {originGeo && (
            <Source id="origin" type="geojson" data={originGeo}>
              <Layer {...ORIGIN_FILL} />
              <Layer {...ORIGIN_LINE} />
            </Source>
          )}
          {driftVectorGeo && (
            <Source id="drift-vector" type="geojson" data={driftVectorGeo}>
              <Layer {...DRIFT_VECTOR_STYLE} />
            </Source>
          )}
          {forecastTrackGeo && (
            <Source id="forecast-track" type="geojson" data={forecastTrackGeo}>
              <Layer {...FORECAST_TRACK_STYLE} />
            </Source>
          )}
        </Map>

        {/* ── Synthetic-AIS warning: full width, impossible to miss ── */}
        {showVessels && ais?.is_synthetic && !noticeDismissed && (
          <div className="absolute inset-x-0 top-0 z-10">
            <SyntheticDataNotice
              variant="banner"
              reason={ais.coverage?.basis ?? ais.reason ?? null}
              count={ais.count}
              window={(ais.window as [string, string]) ?? null}
              onDismiss={() => setNoticeDismissed(true)}
            />
          </div>
        )}

        {/* ── Pre-flight notice: warns BEFORE the layer is switched on ── */}
        {!showVessels && aisCoverage?.use_synthetic && !noticeDismissed && (
          <div className="absolute inset-x-0 bottom-0 z-10">
            <SyntheticDataNotice
              variant="inline"
              reason={aisCoverage.basis}
              onDismiss={() => setNoticeDismissed(true)}
            />
          </div>
        )}

        {/* ── Top-left status / count ─────────────────────────────── */}
        <div className="absolute left-3 top-3 space-y-2">
          <div className="ops-panel px-3 py-2">
            <div className="ops-label">COPERNICUS CDSE</div>
            <div className="mt-0.5 flex items-baseline gap-2 font-mono text-xs text-sentinel-text-hi">
              <span className="text-xl tabular-nums">{scenes.length}</span>
              <span className="text-2xs text-sentinel-muted-hi">scenes in AOI</span>
            </div>
            {error && (
              <div className="mt-1 flex items-center gap-1 text-2xs text-sentinel-danger">
                <TriangleAlert size={10} /> {error.slice(0, 80)}
              </div>
            )}
          </div>
        </div>

        {/* ── Top-right: layer toggles ────────────────────────────── */}
        <div className="absolute right-3 top-3 space-y-1">
          <LayerToggle label="QUICKLOOK" on={showQuicklook} onChange={setShowQuicklook} />
          <LayerToggle
            label={
              aisLoading
                ? 'VESSELS…'
                : aisCoverage?.use_synthetic
                  ? 'VESSELS (SYNTH)'
                  : 'VESSELS'
            }
            on={showVessels}
            onChange={setShowVessels}
          />
        </div>

        {/* ── Attribution provenance strip ────────────────────────────
            A screenshot of the ellipse must be self-labelling: it has to say
            whether the winds behind it were real ERA5 or a synthetic constant. */}
        {attribution && (
          <div className="absolute bottom-11 left-3 flex items-center gap-2 border border-sentinel-border bg-sentinel-panel px-2 py-1 font-mono text-2xs">
            <Crosshair size={10} className="text-[#22c55e]" />
            <span className="uppercase tracking-wider text-sentinel-muted">Origin</span>
            <span className="tabular-nums text-sentinel-text-hi">
              {attribution.origin.center_lon.toFixed(3)}, {attribution.origin.center_lat.toFixed(3)}
            </span>
            <span className="text-sentinel-muted-hi">
              ±{attribution.origin.p95_radius_km.toFixed(1)} km
            </span>
            <span
              className={
                'font-semibold uppercase tracking-wider ' +
                (attribution.wind_source === 'era5'
                  ? 'text-sentinel-nominal'
                  : 'text-sentinel-amber')
              }
            >
              · {attribution.wind_source === 'era5' ? 'ERA5' : 'SYNTH WIND'}
            </span>
            <span className="text-sentinel-muted-hi">
              · {attribution.origin.n_particles}p
            </span>
          </div>
        )}

        {/* ── AIS provenance strip (always visible) ───────────────── */}
        <div className="absolute bottom-3 left-3 flex items-center gap-2 border border-sentinel-border bg-sentinel-panel px-2 py-1 font-mono text-2xs">
          <Ship size={10} className="text-sentinel-muted-hi" />
          <span className="uppercase tracking-wider text-sentinel-muted">AIS</span>
          {aisCoverage ? (
            aisCoverage.use_synthetic ? (
              <span className="font-semibold uppercase tracking-wider text-sentinel-amber">
                Synthetic — no receiver coverage
              </span>
            ) : (
              <span className="font-semibold uppercase tracking-wider text-sentinel-nominal">
                Live terrestrial
              </span>
            )
          ) : (
            <span className="text-sentinel-muted-hi">probing…</span>
          )}
          {showVessels && ais && !aisLoading && (
            <span className="text-sentinel-muted-hi">
              · {ais.count} track{ais.count === 1 ? '' : 's'}
            </span>
          )}
          {aisError && (
            <span className="text-sentinel-danger">· {aisError.slice(0, 60)}</span>
          )}
        </div>
      </main>

      {/* ── RIGHT: scene list + detail ──────────────────────────────── */}
      <aside className="flex w-[360px] flex-col border-l border-sentinel-border bg-sentinel-panel">
        {/* ── Tier-A detection panel (sticky) ─────────────────────── */}
        <DetectionPanel
          list={detectList}
          loading={detectLoading}
          onRefresh={refreshDetections}
          selectedId={attrSceneId}
          onSelect={(id) => setAttrSceneId((cur) => (cur === id ? null : id))}
        />

        <AttributionPanel
          seed={attrSeed}
          vessels={attrVessels}
          onResult={setAttribution}
          onForecast={setForecast}
        />

        <div className="ops-header">
          <span className="ops-label">SCENES</span>
          <span className="ml-auto font-mono text-2xs text-sentinel-muted-hi">
            {scenes.length > 0 ? `${scenes.length} results` : '—'}
          </span>
          <button
            onClick={() => runSearch(query)}
            disabled={loading}
            className="ml-2 text-sentinel-muted-hi hover:text-sentinel-amber disabled:opacity-50"
            aria-label="Refresh"
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>

        {selected ? (
          <SceneDetail scene={selected} ingesting={ingesting} onIngest={sendToDetection} log={ingestLog} error={error} onClose={() => setSelectedId(null)} />
        ) : (
          <SceneList
            scenes={scenes}
            onSelect={(s) => setSelectedId(s.id)}
            loading={loading}
          />
        )}
      </aside>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="ops-label mb-1">{label}</div>
      {children}
    </div>
  );
}

function TimeInput({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex items-center gap-1">
      <span className="w-12 font-mono text-2xs uppercase tracking-wider text-sentinel-muted">{label}</span>
      <input
        type="text"
        placeholder="2020-08-01T00:00:00"
        value={value.replace(/Z$/, '')}
        onChange={(e) => onChange(normalizeIso(e.target.value))}
        className="w-full bg-sentinel-surface border border-sentinel-border px-1.5 py-1 font-mono text-2xs tabular-nums text-sentinel-text-hi focus:border-sentinel-amber focus:outline-none"
      />
    </div>
  );
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <span className="font-mono text-2xs uppercase tracking-wider text-sentinel-muted">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-sentinel-surface border border-sentinel-border px-1 py-1 font-mono text-2xs text-sentinel-text-hi focus:border-sentinel-amber focus:outline-none"
      >
        {options.map((o) => (<option key={o} value={o}>{o}</option>))}
      </select>
    </label>
  );
}

function LayerToggle({ label, on, onChange }: { label: string; on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!on)}
      className={
        'flex items-center gap-2 border px-2 py-1 font-mono text-2xs uppercase tracking-wider ' +
        (on
          ? 'border-sentinel-amber bg-sentinel-amber/10 text-sentinel-amber'
          : 'border-sentinel-border bg-sentinel-panel text-sentinel-muted-hi')
      }
    >
      {on ? <Eye size={11} /> : <EyeOff size={11} />} {label}
    </button>
  );
}

function HealthBar({ health }: { health: { configured: boolean; catalogue: string; detail: string | null } | null }) {
  if (!health) return null;
  const ok = health.configured && health.catalogue === 'reachable';
  return (
    <div className="border-t border-sentinel-border px-3 py-2 font-mono text-2xs">
      <div className="flex items-center gap-2">
        <span
          className={
            'inline-block h-1.5 w-1.5 ' +
            (ok ? 'bg-sentinel-nominal' : health.configured ? 'bg-sentinel-caution' : 'bg-sentinel-danger')
          }
        />
        <span className="uppercase tracking-wider text-sentinel-muted-hi">CDSE {health.catalogue}</span>
      </div>
      {health.detail && <div className="mt-1 text-sentinel-danger">{health.detail.slice(0, 100)}</div>}
    </div>
  );
}

function SceneList({
  scenes, onSelect, loading,
}: { scenes: ArchiveScene[]; onSelect: (s: ArchiveScene) => void; loading: boolean }) {
  if (loading && scenes.length === 0) {
    return <div className="p-3 font-mono text-2xs text-sentinel-muted">Querying Copernicus…</div>;
  }
  if (scenes.length === 0) {
    return (
      <div className="p-3 font-mono text-2xs text-sentinel-muted-hi">
        No Sentinel-1 scenes cover this AOI in the chosen window.
        <br /><br />
        Try a different time window or a wider preset.
      </div>
    );
  }
  return (
    <div className="flex-1 overflow-y-auto">
      {scenes.map((s) => (
        <button
          key={s.id}
          onClick={() => onSelect(s)}
          className="block w-full border-b border-sentinel-border px-3 py-2 text-left hover:bg-sentinel-surface"
        >
          <div className="flex items-center justify-between">
            <span className="font-mono text-2xs uppercase tracking-wider text-sentinel-amber">{s.platform} {s.mode}</span>
            <span className="font-mono text-2xs text-sentinel-muted-hi tabular-nums">{s.size_mb.toFixed(0)} MB</span>
          </div>
          <div className="mt-0.5 font-mono text-2xs text-sentinel-text-hi tabular-nums">{fmtIsoUTC(s.start)}</div>
          <div className="mt-0.5 truncate font-mono text-2xs text-sentinel-muted">{s.name}</div>
        </button>
      ))}
    </div>
  );
}

function SceneDetail({
  scene, ingesting, onIngest, log, error, onClose,
}: {
  scene: ArchiveScene; ingesting: boolean; onIngest: () => void;
  log: string | null; error: string | null; onClose: () => void;
}) {
  return (
    <motion.div initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} className="flex flex-1 flex-col">
      <div className="ops-header">
        <span className="ops-label">SCENE</span>
        <button onClick={onClose} className="ml-auto text-sentinel-muted-hi hover:text-sentinel-amber text-2xs">← BACK</button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-2 font-mono text-2xs">
        <Field label="PLATFORM / MODE" value={`${scene.platform} · ${scene.mode} · ${scene.product_type}`} />
        <Field label="NAME" value={scene.name} mono />
        <Field label="ACQUISITION UTC" value={fmtIsoUTC(scene.start)} />
        <Field label="SIZE" value={`${scene.size_mb.toFixed(1)} MB`} />
        <Field label="ID" value={scene.id} mono />

        <div className="border border-sentinel-border bg-sentinel-surface p-2">
          <div className="ops-label mb-1">SOURCE</div>
          <div className="text-sentinel-text-hi">Copernicus Data Space Ecosystem</div>
          <div className="text-sentinel-muted-hi">CDSE OData catalogue</div>
        </div>

        {log && (
          <div className="flex items-start gap-1 border border-sentinel-nominal/40 bg-sentinel-nominal/10 p-2 text-sentinel-nominal">
            <CheckCircle2 size={11} className="mt-0.5 shrink-0" />
            <span className="break-all">{log}</span>
          </div>
        )}
        {error && (
          <div className="flex items-start gap-1 border border-sentinel-danger/40 bg-sentinel-danger/10 p-2 text-sentinel-danger">
            <TriangleAlert size={11} className="mt-0.5 shrink-0" />
            <span className="break-all">{error}</span>
          </div>
        )}
      </div>

      <div className="border-t border-sentinel-border p-3">
        <button
          onClick={onIngest}
          disabled={ingesting}
          className="w-full border border-sentinel-amber bg-sentinel-amber/10 px-3 py-2 font-mono text-2xs uppercase tracking-widest text-sentinel-amber hover:bg-sentinel-amber/20 disabled:opacity-50"
        >
          {ingesting ? <Loader2 size={11} className="mr-1 inline animate-spin" /> : <Download size={11} className="mr-1 inline" />}
          {ingesting ? 'Fetching GeoTIFF' : 'Send to Detection'}
        </button>
        <div className="mt-1 font-mono text-2xs text-sentinel-muted-hi">
          Renders calibrated σ⁰ via Copernicus, saves a Cloud-Optimised GeoTIFF
          into <span className="text-sentinel-text-hi">data/sar/</span>.
        </div>
      </div>
    </motion.div>
  );
}

function Field({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="ops-label">{label}</div>
      <div className={(mono ? 'font-mono' : '') + ' break-all text-sentinel-text-hi'}>{value}</div>
    </div>
  );
}

interface DetectionRow {
  scene_id: string;
  polygons_kept: number;
  best_confidence: number;
  wind_flag: string;
  ran_utc?: string;
  acquisition_time?: string | null;
  top_area_km2?: number | null;
  top_centroid?: [number, number] | null;
}

function DetectionPanel({
  list, loading, onRefresh, selectedId, onSelect,
}: {
  list: { count: number; results: DetectionRow[] } | null;
  loading: boolean;
  onRefresh: () => void;
  selectedId: string | null;
  onSelect: (sceneId: string) => void;
}) {
  const realDetections = (list?.results ?? []).filter((r) => r.polygons_kept > 0);
  return (
    <div className="border-b border-sentinel-border bg-sentinel-surface">
      <div className="flex items-center gap-2 px-3 py-2">
        <Scan size={11} className="text-sentinel-amber" />
        <span className="font-mono text-2xs font-semibold uppercase tracking-[0.15em] text-sentinel-muted-hi">
          TIER-A DETECTIONS
        </span>
        <span className="ml-auto font-mono text-2xs text-sentinel-muted-hi tabular-nums">
          {list ? `${realDetections.length}/${list.count}` : '—'}
        </span>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="text-sentinel-muted-hi hover:text-sentinel-amber disabled:opacity-50"
          aria-label="Refresh detections"
        >
          <RefreshCw size={10} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>
      <div className="max-h-44 overflow-y-auto border-t border-sentinel-border">
        {!list && loading && (
          <div className="px-3 py-2 font-mono text-2xs text-sentinel-muted-hi">
            Loading detections…
          </div>
        )}
        {list && list.results.length === 0 && (
          <div className="px-3 py-2 font-mono text-2xs text-sentinel-muted-hi">
            No detections on disk. Run the operator after ingest.
          </div>
        )}
        {realDetections.map((r) => {
          const selectable = !!r.acquisition_time && !!r.top_centroid && !!r.top_area_km2;
          const active = r.scene_id === selectedId;
          return (
          <div
            key={r.scene_id}
            role={selectable ? 'button' : undefined}
            tabIndex={selectable ? 0 : undefined}
            onClick={() => selectable && onSelect(r.scene_id)}
            onKeyDown={(e) => {
              if (selectable && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault();
                onSelect(r.scene_id);
              }
            }}
            title={
              selectable
                ? 'Backtrack this detection to its likely origin'
                : 'No acquisition time on file — re-run the detector to enable backtracking'
            }
            className={
              'flex items-center justify-between border-b border-sentinel-border px-3 py-1.5 font-mono text-2xs last:border-b-0 ' +
              (active
                ? 'border-l-2 border-l-sentinel-data bg-sentinel-data/15'
                : selectable
                  ? 'cursor-pointer hover:bg-sentinel-panel'
                  : 'opacity-60')
            }
          >
            <div className="min-w-0 flex-1">
              <div className="truncate text-sentinel-text-hi" title={r.scene_id}>
                {r.scene_id}
              </div>
              <div className="mt-0.5 flex items-center gap-2">
                <span className="uppercase tracking-wider text-sentinel-muted-hi">
                  {r.polygons_kept} POLY
                </span>
                {r.top_area_km2 != null && (
                  <span className="uppercase tracking-wider text-sentinel-muted-hi">
                    · {r.top_area_km2.toFixed(2)} KM²
                  </span>
                )}
                <span
                  className={
                    r.wind_flag === 'OK'
                      ? 'text-sentinel-nominal'
                      : 'text-sentinel-caution'
                  }
                >
                  · {r.wind_flag === 'OK' ? 'WIND OK' : 'NO WIND'}
                </span>
              </div>
            </div>
            <div className="ml-2 flex flex-col items-end">
              <span
                className={
                  r.best_confidence >= 0.75
                    ? 'text-sentinel-nominal'
                    : r.best_confidence >= 0.45
                      ? 'text-sentinel-caution'
                      : 'text-sentinel-muted-hi'
                }
              >
                {(r.best_confidence * 100).toFixed(0)}%
              </span>
              <span className="text-sentinel-muted-hi">best conf</span>
            </div>
          </div>
          );
        })}
        {list && realDetections.length === 0 && list.results.length > 0 && (
          <div className="px-3 py-2 font-mono text-2xs text-sentinel-muted-hi">
            All {list.count} scenes run — none returned spills above threshold.
          </div>
        )}
      </div>
    </div>
  );
}

// ── helpers ──────────────────────────────────────────────────────────────

function addOneSecond(iso: string) {
  const d = new Date(iso);
  d.setUTCSeconds(d.getUTCSeconds() + 1);
  return d.toISOString().replace('.000Z', 'Z');
}

/** Accept `YYYY-MM-DDTHH:mm` or `...THH:mm:ss` (with or without Z) → canonical `...Z`. */
function normalizeIso(v: string): string {
  let s = v.trim().replace(/Z$/, '');
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(s)) s = `${s}:00`;
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) s = `${s}T00:00:00`;
  return `${s}Z`;
}

function orderX(a: number, b: number): [number, number] { return a < b ? [a, b] : [b, a]; }
function orderY(a: number, b: number): [number, number] { return a < b ? [a, b] : [b, a]; }
function orderBbox(b: [number, number, number, number]): [number, number, number, number] {
  const [w, e] = orderX(b[0], b[2]);
  const [s, n] = orderY(b[1], b[3]);
  return [w, s, e, n];
}

function trackFeature(seg: [number, number][], v: { mmsi: string; name: string }): GeoJSON.Feature {
  return {
    type: 'Feature',
    geometry: { type: 'LineString', coordinates: seg },
    properties: { mmsi: v.mmsi, name: v.name },
  };
}

/** MapLibre `image` source coordinates: top-left, top-right, bottom-right, bottom-left. */
function bboxToRasterCoords(b: [number, number, number, number]): [[number, number], [number, number], [number, number], [number, number]] {
  const [w, s, e, n] = b;
  return [[w, n], [e, n], [e, s], [w, s]];
}
