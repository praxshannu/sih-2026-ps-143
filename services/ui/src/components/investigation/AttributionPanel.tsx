import { useCallback, useState } from 'react';
import { AlertTriangle, Crosshair, Ship, Waves, Wind } from 'lucide-react';
import {
  api,
  type AttributionRequest,
  type AttributionResult,
  type ForecastResult,
} from '../../lib/api';

/**
 * Backward drift attribution — "where did this slick come from?".
 *
 * Wraps POST /drift/attribution (OpenDrift OpenOil, integrated backwards in
 * time from the detected slick). Two honesty rules are enforced here, not
 * just in the backend:
 *
 *  1. Forcing provenance is shown BEFORE the numbers. A hindcast driven by
 *     synthetic constant wind is labelled SYNTHETIC and the ellipse is drawn
 *     muted — presenting it as a real result would be misleading.
 *  2. Particle survival is shown. An ellipse fitted to 3 survivors is not a
 *     95% confidence region and must not look like one.
 */

export interface DetectionSeed {
  scene_id: string;
  detection_lon: number;
  detection_lat: number;
  detection_area_km2: number;
  detection_time: string;
  best_confidence?: number;
}

const DURATION_PRESETS = [6, 12, 24, 48, 72] as const;

/** Human labels for the resolved forcing. GFS and ERA5 are both real. */
const WIND_LABEL: Record<string, string> = {
  era5: 'ERA5',
  gfs: 'GFS',
  synthetic_constant: 'SYNTHETIC',
};

function fmt(v: number, digits = 2): string {
  return Number.isFinite(v) ? v.toFixed(digits) : '—';
}

export default function AttributionPanel({
  seed,
  vessels,
  onResult,
  onForecast,
}: {
  /** The detection to backtrack. Null until the operator picks a scene. */
  seed: DetectionSeed | null;
  /** Vessels to score against the origin ellipse (AIS payload). */
  vessels?: Array<{ mmsi: string; name: string; longitude: number; latitude: number; flag?: string; provenance?: string }>;
  /** Fired on every successful run so the map can draw the ellipse. */
  onResult?: (r: AttributionResult | null) => void;
  /** Fired on every successful forecast so the map can draw the cone. */
  onForecast?: (r: ForecastResult | null) => void;
}) {
  const [durationH, setDurationH] = useState<number>(24);
  // Wind source. `auto` picks GFS for recent detections (no CDS queue) and
  // ERA5 for anything older, since NOMADS only retains ~10 days.
  const [forcing, setForcing] = useState<'auto' | 'era5' | 'gfs'>('auto');
  const [useCmems, setUseCmems] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<AttributionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  // ── Forward forecast (shoreline impact) ────────────────────────────
  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const [fcHours, setFcHours] = useState<number>(48);
  const [fcRunning, setFcRunning] = useState(false);
  const [fcError, setFcError] = useState<string | null>(null);

  const runForecast = useCallback(async () => {
    if (!result) return;
    setFcRunning(true);
    setFcError(null);
    try {
      // Seed from the OBSERVED slick, not the inferred origin.
      // The origin ellipse answers "who did it"; the forecast answers "where
      // does the oil I can actually see go next?". Seeding at the origin would
      // mean starting the clock at (detection - backtrack) and re-simulating
      // drift we already inferred — so the +48 h forecast would in fact be a
      // +24 h forecast that duplicates the hindcast.
      const [dlon, dlat] = result.detection_centroid;
      const r = await api.runForecast({
        origin_lon: dlon,
        origin_lat: dlat,
        origin_time: result.detection_time,
        seed_radius_km: Math.max(0.5, Math.sqrt(result.detection_area_km2 / Math.PI)),
        duration_h: fcHours,
        use_era5: true,
        use_cmems: useCmems,
        forcing,
        n_members: 64,
      });
      setForecast(r);
      onForecast?.(r);
    } catch (e) {
      setFcError(e instanceof Error ? e.message : String(e));
      setForecast(null);
      onForecast?.(null);
    } finally {
      setFcRunning(false);
    }
  }, [result, fcHours, forcing, useCmems, onForecast]);

  const run = useCallback(async () => {
    if (!seed) return;
    setRunning(true);
    setError(null);
    setElapsed(null);
    // A new origin invalidates any forecast built on the old one.
    setForecast(null);
    onForecast?.(null);
    const t0 = performance.now();
    try {
      const body: AttributionRequest = {
        detection_lon: seed.detection_lon,
        detection_lat: seed.detection_lat,
        detection_area_km2: seed.detection_area_km2,
        detection_time: seed.detection_time,
        duration_h: durationH,
        use_era5: true,
        use_cmems: useCmems,
        forcing,
        n_members: 64,
        vessels,
      };
      const r = await api.runAttribution(body);
      setResult(r);
      onResult?.(r);
      setElapsed((performance.now() - t0) / 1000);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setResult(null);
      onResult?.(null);
    } finally {
      setRunning(false);
    }
  }, [seed, durationH, forcing, useCmems, vessels, onResult, onForecast]);

  const syntheticForcing =
    !!result && (result.wind_source !== 'era5' || result.current_source !== 'cmems');
  const weakEnsemble = !!result && result.origin.n_particles < 16;

  return (
    <div className="border-b border-sentinel-border bg-sentinel-surface">
      <div className="flex items-center gap-2 px-3 py-2">
        <Crosshair size={11} className="text-sentinel-amber" />
        <span className="font-mono text-2xs font-semibold uppercase tracking-[0.15em] text-sentinel-muted-hi">
          Backward Attribution
        </span>
        <span className="ml-auto font-mono text-2xs text-sentinel-muted-hi tabular-nums">
          {result ? `${result.origin.n_particles}p` : '—'}
        </span>
        <button
          onClick={run}
          disabled={running || !seed}
          className="border border-sentinel-border px-2 py-0.5 font-mono text-2xs uppercase tracking-wider text-sentinel-muted-hi hover:border-sentinel-amber hover:text-sentinel-amber disabled:opacity-40"
          aria-label="Run backward attribution"
        >
          {running ? 'Running…' : 'Run'}
        </button>
      </div>

      {/* ── Controls ─────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-sentinel-border px-3 py-2">
        <span className="ops-label">Backtrack</span>
        {DURATION_PRESETS.map((h) => (
          <button
            key={h}
            onClick={() => setDurationH(h)}
            className={
              'border px-1.5 py-0.5 font-mono text-2xs tabular-nums transition-colors ' +
              (durationH === h
                ? 'border-sentinel-amber text-sentinel-amber'
                : 'border-sentinel-border text-sentinel-muted-hi hover:text-sentinel-text-hi')
            }
          >
            {h}h
          </button>
        ))}
        <label
          className="ml-auto flex cursor-pointer items-center gap-1 font-mono text-2xs uppercase tracking-wider text-sentinel-muted-hi"
          title="auto: GFS for recent windows (~20s, no queue), ERA5 for older (minutes, but full history)"
        >
          wind
          <select
            value={forcing}
            onChange={(e) => setForcing(e.target.value as 'auto' | 'era5' | 'gfs')}
            className="border border-sentinel-border bg-sentinel-panel px-1 py-0.5 font-mono text-2xs uppercase text-sentinel-text-hi"
          >
            <option value="auto">auto</option>
            <option value="gfs">gfs</option>
            <option value="era5">era5</option>
          </select>
        </label>
        <label className="flex cursor-pointer items-center gap-1 font-mono text-2xs uppercase tracking-wider text-sentinel-muted-hi">
          <input
            type="checkbox"
            checked={useCmems}
            onChange={(e) => setUseCmems(e.target.checked)}
            className="h-2.5 w-2.5 accent-[#d97706]"
          />
          CMEMS currents
        </label>
      </div>

      {/* ── Body ─────────────────────────────────────────────────── */}
      {!seed && (
        <p className="px-3 py-2 font-mono text-2xs text-sentinel-muted-hi">
          Select a detection to backtrack.
        </p>
      )}

      {running && (
        <p className="flex items-center gap-2 px-3 py-2 font-mono text-2xs text-sentinel-amber">
          <Wind size={10} className="animate-pulse" />
          Fetching forcing + integrating {durationH} h backward…
        </p>
      )}

      {error && (
        <p className="border-l-2 border-sentinel-danger px-3 py-2 font-mono text-2xs text-sentinel-danger">
          {error.slice(0, 220)}
        </p>
      )}

      {result && (
        <>
          {/* Provenance badges — shown before any number. */}
          <div className="flex flex-wrap items-center gap-1.5 border-b border-sentinel-border px-3 py-1.5">
            <span
              className={
                'border px-1.5 py-0.5 font-mono text-2xs uppercase tracking-wider ' +
                (result.wind_source === 'synthetic_constant'
                  ? 'border-sentinel-amber text-sentinel-amber'
                  : 'border-sentinel-nominal text-sentinel-nominal')
              }
            >
              wind: {WIND_LABEL[result.wind_source]}
            </span>
            <span
              className={
                'border px-1.5 py-0.5 font-mono text-2xs uppercase tracking-wider ' +
                (result.current_source === 'cmems'
                  ? 'border-sentinel-nominal text-sentinel-nominal'
                  : 'border-sentinel-border text-sentinel-muted-hi')
              }
            >
              cur: {result.current_source === 'cmems' ? 'CMEMS' : 'SYNTHETIC'}
            </span>
            {elapsed !== null && (
              <span className="ml-auto font-mono text-2xs text-sentinel-muted-hi tabular-nums">
                {elapsed.toFixed(1)}s
              </span>
            )}
          </div>

          {syntheticForcing && (
            <div className="flex items-start gap-1.5 border-b border-sentinel-border bg-[#1a1206] px-3 py-1.5">
              <AlertTriangle size={10} className="mt-0.5 flex-shrink-0 text-sentinel-amber" />
              <p className="font-mono text-2xs leading-relaxed text-sentinel-amber">
                Forcing is partly synthetic. The origin below is a model artefact,
                not evidence — do not cite it in a report.
              </p>
            </div>
          )}

          {weakEnsemble && (
            <div className="flex items-start gap-1.5 border-b border-sentinel-border bg-[#1a1206] px-3 py-1.5">
              <AlertTriangle size={10} className="mt-0.5 flex-shrink-0 text-sentinel-amber" />
              <p className="font-mono text-2xs leading-relaxed text-sentinel-amber">
                Only {result.origin.n_particles} particles survived the backtrack —
                the ellipse is not a stable 95% region.
              </p>
            </div>
          )}

          {/* Origin ellipse */}
          <div className="border-b border-sentinel-border px-3 py-2">
            <div className="ops-label mb-1">Origin ellipse (2σ)</div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-2xs">
              <Kv k="centre" v={`${fmt(result.origin.center_lon, 4)}, ${fmt(result.origin.center_lat, 4)}`} />
              <Kv k="semi-major" v={`${fmt(result.origin.semi_major_km)} km`} />
              <Kv k="semi-minor" v={`${fmt(result.origin.semi_minor_km)} km`} />
              <Kv k="orient" v={`${fmt(result.origin.orientation_deg, 1)}°`} />
              <Kv k="p50" v={`${fmt(result.origin.p50_radius_km)} km`} />
              <Kv k="p95" v={`${fmt(result.origin.p95_radius_km)} km`} />
            </div>
          </div>

          {/* Suspects */}
          <div className="max-h-40 overflow-y-auto">
            {result.suspect_vessels.length === 0 ? (
              <p className="px-3 py-2 font-mono text-2xs text-sentinel-muted-hi">
                No vessels supplied to score.
              </p>
            ) : (
              result.suspect_vessels.map((s, i) => (
                <div
                  key={`${s.mmsi}-${i}`}
                  className="flex items-center justify-between border-b border-sentinel-border px-3 py-1.5 font-mono text-2xs last:border-b-0"
                >
                  <div className="flex min-w-0 flex-1 items-center gap-1.5">
                    <Ship
                      size={10}
                      className={
                        s.inside_p95 ? 'text-sentinel-danger' : 'text-sentinel-muted-hi'
                      }
                    />
                    <div className="min-w-0">
                      <div className="truncate text-sentinel-text-hi">{s.name || s.mmsi}</div>
                      <div className="truncate text-sentinel-muted-hi">
                        MMSI {s.mmsi}
                        {s.flag ? ` · ${s.flag}` : ''}
                        {s.provenance === 'synthetic_mock' ? ' · SYNTH' : ''}
                      </div>
                    </div>
                  </div>
                  <div className="ml-2 flex flex-col items-end">
                    <span
                      className={
                        s.inside_p50
                          ? 'text-sentinel-danger'
                          : s.inside_p95
                            ? 'text-sentinel-caution'
                            : 'text-sentinel-muted-hi'
                      }
                    >
                      {fmt(s.distance_to_origin_km)} km
                    </span>
                    <span className="text-sentinel-muted-hi">
                      {s.inside_p50 ? 'in p50' : s.inside_p95 ? 'in p95' : 'outside'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>

          {result.notes.length > 0 && (
            <div className="border-t border-sentinel-border px-3 py-1.5">
              {result.notes.map((n, i) => (
                <p key={i} className="font-mono text-2xs leading-relaxed text-sentinel-muted-hi">
                  · {n}
                </p>
              ))}
            </div>
          )}

          {/* ── Forward forecast / shoreline impact ─────────────────── */}
          <div className="border-t border-sentinel-border px-3 py-2">
            <div className="mb-1.5 flex items-center gap-2">
              <Waves size={10} className="text-sentinel-muted-hi" />
              <span className="ops-label">Project forward</span>
              <div className="ml-auto flex gap-1">
                {[24, 48, 72].map((h) => (
                  <button
                    key={h}
                    onClick={() => setFcHours(h)}
                    className={
                      'border px-1.5 py-0.5 font-mono text-2xs tabular-nums ' +
                      (fcHours === h
                        ? 'border-sentinel-amber text-sentinel-amber'
                        : 'border-sentinel-border text-sentinel-muted-hi hover:text-sentinel-text-hi')
                    }
                  >
                    {h}h
                  </button>
                ))}
                <button
                  onClick={runForecast}
                  disabled={fcRunning}
                  className="border border-sentinel-border px-2 py-0.5 font-mono text-2xs uppercase tracking-wider text-sentinel-muted-hi hover:border-sentinel-amber hover:text-sentinel-amber disabled:opacity-40"
                >
                  {fcRunning ? '…' : 'Run'}
                </button>
              </div>
            </div>

            {fcError && (
              <p className="font-mono text-2xs text-sentinel-danger">{fcError.slice(0, 160)}</p>
            )}

            {forecast && (
              <div className="space-y-1.5">
                {/* Shoreline impact is the headline — show it first and big. */}
                <div className="flex items-baseline gap-2">
                  <span
                    className={
                      'font-mono text-base font-semibold tabular-nums ' +
                      (forecast.stranded_fraction >= 0.5
                        ? 'text-sentinel-danger'
                        : forecast.stranded_fraction > 0
                          ? 'text-sentinel-caution'
                          : 'text-sentinel-nominal')
                    }
                  >
                    {(forecast.stranded_fraction * 100).toFixed(0)}%
                  </span>
                  <span className="ops-label">beached</span>
                  <span className="ml-auto font-mono text-2xs text-sentinel-muted-hi tabular-nums">
                    {forecast.first_stranding_h !== null
                      ? `landfall +${forecast.first_stranding_h}h`
                      : 'no landfall'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-x-3 font-mono text-2xs">
                  <Kv
                    k="final centre"
                    v={`${fmt(forecast.final_center[0], 3)}, ${fmt(forecast.final_center[1], 3)}`}
                  />
                  <Kv
                    k="cone p95"
                    v={
                      forecast.cone.length
                        ? `${fmt(forecast.cone[forecast.cone.length - 1].p95_radius_km)} km`
                        : '—'
                    }
                  />
                </div>
                {forecast.notes.map((n, i) => (
                  <p key={i} className="font-mono text-2xs leading-relaxed text-sentinel-amber">
                    · {n}
                  </p>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-sentinel-muted-hi">{k}</span>
      <span className="text-sentinel-text-hi tabular-nums">{v}</span>
    </div>
  );
}
