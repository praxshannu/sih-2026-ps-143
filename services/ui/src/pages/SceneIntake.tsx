import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Download,
  FileUp,
  Loader2,
  RotateCcw,
  Ship,
  Trash2,
  Waves,
} from 'lucide-react';

import api, {
  type AttributionResult,
  type ForecastResult,
  type SceneDetection,
  type SceneHealth,
  type SceneInference,
  type SceneListItem,
  type SceneState,
} from '../lib/api';

/**
 * Scene Intake — the analyst's front door.
 *
 * The whole page is organised around one rule: **a number without provenance
 * and a confidence interval is not a result.** That is why every panel leads
 * with where its data came from, and why the state banner is the loudest thing
 * on screen. An analyst must be able to tell, at a glance and without reading
 * carefully, the difference between:
 *
 *   nothing here · this scene is unusable · we could not check · here is a spill
 *
 * Four states, four different visual treatments, no shared styling. Collapsing
 * any two of them is how a tool starts lying.
 */

// ── Small, shared instruments ─────────────────────────────────────────────

type Provenance = 'real' | 'synthetic' | 'unavailable' | 'failed';

const PROVENANCE_LABEL: Record<Provenance, string> = {
  real: 'REAL',
  synthetic: 'SYNTHETIC',
  unavailable: 'UNAVAILABLE',
  failed: 'FAILED',
};

/** Glyph + letter code, never a coloured pill. Colour-blind safe, and it reads
 *  like equipment rather than like a status badge from a component library. */
function StatusGlyph({ state }: { state: Provenance }) {
  const glyph = state === 'real' ? '■' : state === 'synthetic' ? '▲' : state === 'failed' ? '✕' : '□';
  return (
    <span className={`ops-status ops-provenance-${state}`}>
      <span aria-hidden="true">{glyph}</span>
      {PROVENANCE_LABEL[state]}
    </span>
  );
}

/** A self-labelling band. Designed to survive a screenshot crop: solid, not a
 *  subtle badge, because a cropped screenshot is how most of this ends up
 *  being read by someone who never saw the tool. */
function ProvenanceStrip({
  state,
  children,
}: {
  state: Provenance;
  children?: React.ReactNode;
}) {
  return (
    <div className={`ops-provenance ops-provenance-${state}`}>
      <StatusGlyph state={state} />
      <span className="text-sentinel-muted-hi">·</span>
      <span className="truncate normal-case tracking-normal">{children}</span>
    </div>
  );
}

function Field({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="ops-row">
      <span className="ops-label">{label}</span>
      <span className="flex flex-col items-end">
        <span className="ops-value-hi">{value}</span>
        {hint && <span className="font-mono text-3xs text-sentinel-muted">{hint}</span>}
      </span>
    </div>
  );
}

/** Confidence is always shown as an interval. A bare 0.93 is not an answer. */
function ConfidenceBar({
  value,
  low,
  high,
}: {
  value: number;
  low: number;
  high: number;
}) {
  const pct = (n: number) => `${Math.max(0, Math.min(1, n)) * 100}%`;
  const tone = value >= 0.8 ? 'bg-sentinel-nominal' : value >= 0.6 ? 'bg-sentinel-caution' : 'bg-sentinel-danger';
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-2">
        <span className="ops-value-hi">{value.toFixed(3)}</span>
        <span className="font-mono text-3xs text-sentinel-muted">
          95% CI {low.toFixed(3)}–{high.toFixed(3)}
        </span>
      </div>
      {/* The track is the full interval; the fill is the point estimate. The
          gap between them is the uncertainty, and it is visible on purpose. */}
      <div className="conf-bar-track relative">
        <div className="absolute inset-y-0 bg-sentinel-border-hi" style={{ left: pct(low), width: pct(high - low) }} />
        <div className={`conf-bar-fill absolute inset-y-0 left-0 ${tone}`} style={{ width: pct(value) }} />
      </div>
    </div>
  );
}

/** The state banner. This is the single most important element on the page. */
const STATE_TREATMENT: Record<
  SceneState,
  { glyph: string; label: string; tone: string; blurb: string }
> = {
  ok: {
    glyph: '■',
    label: 'DETECTION',
    tone: 'text-sentinel-oil border-sentinel-oil/40 bg-sentinel-oil/5',
    blurb: 'A slick candidate survived every gate. Confidence below carries a Wilson 95% interval.',
  },
  no_detection: {
    glyph: '□',
    label: 'NO DETECTION',
    tone: 'text-sentinel-muted-hi border-sentinel-border bg-sentinel-surface',
    blurb: 'The scene is valid and nothing crossed the threshold. This is a result, not a failure.',
  },
  low_confidence: {
    glyph: '▲',
    label: 'LOW CONFIDENCE',
    tone: 'text-sentinel-caution border-sentinel-caution/40 bg-sentinel-caution/5',
    blurb: 'Something was found, but even the upper bound of the interval is below the usable threshold.',
  },
  out_of_distribution: {
    glyph: '▲',
    label: 'OUT OF DISTRIBUTION',
    tone: 'text-sentinel-caution border-sentinel-caution/40 bg-sentinel-caution/5',
    blurb:
      'Scene statistics fall outside the envelope the thresholds were calibrated on. Detections are shown but must not be trusted as in-envelope.',
  },
  missing_forcing_data: {
    glyph: '▲',
    label: 'FORCING MISSING',
    tone: 'text-sentinel-caution border-sentinel-caution/40 bg-sentinel-caution/5',
    blurb:
      'Wind was required and unavailable, so look-alike discrimination — the gate that separates oil from a wind shadow — could not run.',
  },
  invalid_scene: {
    glyph: '✕',
    label: 'INVALID SCENE',
    tone: 'text-sentinel-danger border-sentinel-danger/40 bg-sentinel-danger/5',
    blurb: 'The scene never reached inference. The reason below is the whole story; nothing was computed.',
  },
};

function StateBanner({ result }: { result: SceneInference }) {
  const treatment = STATE_TREATMENT[result.state] ?? STATE_TREATMENT.invalid_scene;
  return (
    <div className={`border ${treatment.tone} px-4 py-3`}>
      <div className="flex items-center gap-2.5">
        <span aria-hidden="true" className="font-mono text-sm">
          {treatment.glyph}
        </span>
        <span className="font-display text-sm font-medium uppercase tracking-instrument">
          {treatment.label}
        </span>
        <span className="font-mono text-3xs uppercase tracking-instrument text-sentinel-muted">
          {result.detector}
        </span>
        <span className="ml-auto font-mono text-3xs tabular-nums text-sentinel-muted">
          {result.inference_ms.toFixed(0)} ms
        </span>
      </div>
      <p className="mt-1.5 max-w-[80ch] font-sans text-xs text-sentinel-text">{treatment.blurb}</p>
      {result.state_reason && (
        <p className="mt-1.5 font-mono text-2xs text-sentinel-muted-hi">
          reason · {result.state_reason}
        </p>
      )}
      {result.flags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {result.flags.map((flag) => (
            <span
              key={flag}
              className="border border-sentinel-border bg-sentinel-bg px-1.5 py-0.5 font-mono text-3xs text-sentinel-muted-hi"
            >
              {flag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Detection table ───────────────────────────────────────────────────────

function DetectionRow({
  detection,
  selected,
  onSelect,
}: {
  detection: SceneDetection;
  selected: boolean;
  onSelect: () => void;
}) {
  const [lon, lat] = Array.isArray(detection.centroid)
    ? detection.centroid
    : [detection.centroid.longitude, detection.centroid.latitude];

  return (
    <tr
      onClick={onSelect}
      className={`cursor-pointer border-l-2 ${
        selected ? 'border-l-sentinel-oil bg-sentinel-raised' : 'border-l-transparent'
      }`}
    >
      <td className="ops-stencil">{detection.id.replace('det_', '')}</td>
      <td className="text-right text-sentinel-oil">{detection.area_km2.toFixed(3)}</td>
      <td className="text-right">{detection.length_km.toFixed(2)}</td>
      <td className="text-right">{detection.width_km.toFixed(2)}</td>
      <td className="text-right">{detection.orientation_deg.toFixed(1)}</td>
      <td className="text-right">{detection.scene_contrast_db?.toFixed(2) ?? '—'}</td>
      <td className="text-right">
        <span className="text-sentinel-text-hi">{detection.confidence.value?.toFixed(3)}</span>
        <span className="ml-1 text-3xs text-sentinel-muted">
          [{detection.confidence.low?.toFixed(3)}–{detection.confidence.high?.toFixed(3)}]
        </span>
      </td>
      <td className="text-right text-sentinel-muted-hi">
        {lat.toFixed(4)}, {lon.toFixed(4)}
      </td>
    </tr>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────

export default function SceneIntake() {
  const [health, setHealth] = useState<SceneHealth | null>(null);
  const [scenes, setScenes] = useState<SceneListItem[]>([]);
  const [result, setResult] = useState<SceneInference | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [wind, setWind] = useState<string>('');
  const [dragging, setDragging] = useState(false);
  const [drift, setDrift] = useState<AttributionResult | null>(null);
  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refreshScenes = useCallback(async () => {
    try {
      const list = await api.listScenes();
      setScenes(list.scenes);
    } catch {
      // Listing is a convenience, never a blocker: if the detect service is
      // down the analyst can still upload once it returns.
      setScenes([]);
    }
  }, []);

  useEffect(() => {
    api.getSceneHealth().then(setHealth).catch(() => setHealth(null));
    void refreshScenes();
  }, [refreshScenes]);

  const windValue = useMemo(() => {
    const parsed = Number.parseFloat(wind);
    return Number.isFinite(parsed) ? parsed : undefined;
  }, [wind]);

  const run = useCallback(
    async (label: string, task: () => Promise<SceneInference>) => {
      setBusy(label);
      setError(null);
      setDrift(null);
      setForecast(null);
      try {
        const next = await task();
        setResult(next);
        setSelectedId(next.detections[0]?.id ?? null);
        await refreshScenes();
      } catch (exc) {
        setError(
          exc instanceof Error
            ? exc.message
            : 'The request failed. The detect service may be unreachable — see the health readout.',
        );
      } finally {
        setBusy(null);
        setProgress(0);
      }
    },
    [refreshScenes],
  );

  const onFile = useCallback(
    (file: File) => {
      void run(`Uploading ${file.name}`, () =>
        api.uploadScene(file, {
          wind_speed_ms: windValue,
          onProgress: setProgress,
        }),
      );
    },
    [run, windValue],
  );

  const selected = useMemo(
    () => result?.detections.find((d) => d.id === selectedId) ?? result?.detections[0] ?? null,
    [result, selectedId],
  );

  const scenePath = result?.provenance?.path as string | undefined;
  const acquisition = (result?.provenance?.acquisition_time as string | undefined) ?? undefined;

  const runHindcast = useCallback(async () => {
    if (!selected || !acquisition) return;
    const [lon, lat] = Array.isArray(selected.centroid)
      ? selected.centroid
      : [selected.centroid.longitude, selected.centroid.latitude];
    setBusy('Hindcast');
    setError(null);
    try {
      setDrift(
        await api.runAttribution({
          detection_lon: lon,
          detection_lat: lat,
          detection_area_km2: selected.area_km2,
          detection_time: acquisition,
          duration_h: 24,
        }),
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Hindcast failed.');
    } finally {
      setBusy(null);
    }
  }, [selected, acquisition]);

  const runForecast = useCallback(async () => {
    if (!drift) return;
    setBusy('Forecast');
    setError(null);
    try {
      setForecast(
        await api.runForecast({
          origin_lon: drift.origin.center_lon ?? drift.detection_centroid[0],
          origin_lat: drift.origin.center_lat ?? drift.detection_centroid[1],
          origin_time: drift.detection_time,
          duration_h: 48,
        }),
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Forecast failed.');
    } finally {
      setBusy(null);
    }
  }, [drift]);

  const downloadGeojson = useCallback(() => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.geojson, null, 2)], { type: 'application/geo+json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${scenePath?.split('/').pop() ?? 'scene'}.geojson`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [result, scenePath]);

  const modelStatus = health?.model?.status ?? 'unknown';

  return (
    <div className="grid h-full grid-cols-[19rem_minmax(0,1fr)_21rem] overflow-hidden">
      {/* ── INTAKE ─────────────────────────────────────────────────────── */}
      <section className="flex min-h-0 flex-col overflow-y-auto border-r border-sentinel-border bg-sentinel-surface">
        <header className="ops-header sticky top-0 z-10">
          <span className="ops-header-label">Intake</span>
          <span className="ml-auto font-mono text-3xs text-sentinel-muted">{scenes.length} on disk</span>
        </header>

        <div className="px-3.5 py-3">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const file = e.dataTransfer.files?.[0];
              if (file) onFile(file);
            }}
            className={`ops-drop ${dragging ? 'ops-drop-active' : ''}`}
          >
            <FileUp size={18} strokeWidth={1.5} className="text-sentinel-muted-hi" />
            <p className="mt-2 font-display text-xs text-sentinel-text-hi">Drop a Sentinel-1 GeoTIFF</p>
            <p className="mt-1 font-mono text-3xs text-sentinel-muted">.tif / .tiff · up to 512 MB</p>
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={busy !== null}
              className="mt-3 border border-sentinel-border-hi px-3 py-1 font-display text-3xs uppercase tracking-instrument text-sentinel-text-hi transition-colors hover:border-sentinel-data hover:text-sentinel-data disabled:opacity-40"
            >
              Choose file
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".tif,.tiff"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onFile(file);
                e.target.value = '';
              }}
            />
          </div>

          {busy === 'Uploading' || progress > 0 ? (
            <div className="mt-2">
              <div className="conf-bar-track">
                <div className="conf-bar-fill bg-sentinel-data" style={{ width: `${progress * 100}%` }} />
              </div>
              <p className="mt-1 font-mono text-3xs tabular-nums text-sentinel-muted">
                {(progress * 100).toFixed(0)}% transferred
              </p>
            </div>
          ) : null}

          {/* Wind is optional and that is deliberate. Leaving it blank runs the
              detector wind-blind and the result says so — far better than
              inventing a plausible 5 m/s. */}
          <label className="mt-3 block">
            <span className="ops-label">Wind speed (m/s) — optional</span>
            <input
              value={wind}
              onChange={(e) => setWind(e.target.value)}
              inputMode="decimal"
              placeholder="leave blank = run wind-blind, flagged"
              className="mt-1 w-full border border-sentinel-border bg-sentinel-bg px-2 py-1 font-mono text-xs text-sentinel-text placeholder:text-sentinel-muted focus:border-sentinel-data focus:outline-none"
            />
          </label>
          <p className="mt-1.5 font-sans text-3xs leading-relaxed text-sentinel-muted">
            Slicks are separable from look-alikes in roughly 2–10 m/s. Supplying a value only
            removes the wind-blind flag; it is not used to fabricate a detection.
          </p>
        </div>

        <div className="ops-hairline" />

        <div className="px-3.5 py-3">
          <span className="ops-label">Detector</span>
          <div className="mt-1.5 border border-sentinel-border bg-sentinel-bg">
            <Field label="detector" value={health?.detector ?? '—'} />
            <Field
              label="unet++"
              value={
                <span className={modelStatus === 'loaded' ? 'ops-nominal' : 'ops-caution'}>
                  {modelStatus.toUpperCase()}
                </span>
              }
              hint={modelStatus === 'untrained' ? 'no checkpoint · no probabilities emitted' : undefined}
            />
            <Field label="scenes" value={String(health?.scene_count ?? 0)} />
          </div>
        </div>

        <div className="ops-hairline" />

        <header className="ops-header">
          <span className="ops-header-label">On disk</span>
          <button
            type="button"
            onClick={() => void refreshScenes()}
            className="ml-auto text-sentinel-muted transition-colors hover:text-sentinel-text-hi"
            aria-label="Refresh scene list"
          >
            <RotateCcw size={11} strokeWidth={1.75} />
          </button>
        </header>

        {scenes.length === 0 ? (
          <p className="px-3.5 py-3 font-sans text-2xs text-sentinel-muted">
            No scenes found. Upload one, or ingest a scene from the Archive.
          </p>
        ) : (
          <ul>
            {scenes.map((scene) => (
              <li key={scene.name} className="border-b border-sentinel-border">
                <button
                  type="button"
                  onClick={() => void run(`Loading ${scene.name}`, () => api.inferScene({ path: scene.path, wind_speed_ms: windValue }))}
                  className="w-full px-3.5 py-2 text-left transition-colors hover:bg-sentinel-raised"
                >
                  <span className="block truncate font-mono text-2xs text-sentinel-text-hi">
                    {scene.name}
                  </span>
                  <span className="mt-0.5 flex items-center gap-2 font-mono text-3xs text-sentinel-muted">
                    <span>{(scene.bytes / 1e6).toFixed(1)} MB</span>
                    {scene.has_result && (
                      <span className="text-sentinel-data">· cached {scene.result_state ?? ''}</span>
                    )}
                  </span>
                </button>
                {scene.has_result && (
                  <div className="flex border-t border-sentinel-border">
                    <button
                      type="button"
                      onClick={() =>
                        void run(`Reopening ${scene.name}`, () => api.getSceneResult(scene.path))
                      }
                      className="flex-1 px-3.5 py-1 font-mono text-3xs uppercase tracking-instrument text-sentinel-data transition-colors hover:bg-sentinel-raised"
                    >
                      reopen
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        await api.deleteScene(scene.name);
                        await refreshScenes();
                      }}
                      className="border-l border-sentinel-border px-2 py-1 text-sentinel-muted transition-colors hover:text-sentinel-danger"
                      aria-label={`Delete ${scene.name}`}
                    >
                      <Trash2 size={10} strokeWidth={1.75} />
                    </button>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* ── SEGMENTATION & GEOMETRY ────────────────────────────────────── */}
      <section className="flex min-h-0 flex-col overflow-y-auto">
        <header className="ops-header sticky top-0 z-10">
          <span className="ops-header-label">Segmentation &amp; geometry</span>
          {result && (
            <button
              type="button"
              onClick={downloadGeojson}
              className="ml-auto flex items-center gap-1.5 font-mono text-3xs uppercase tracking-instrument text-sentinel-muted transition-colors hover:text-sentinel-data"
            >
              <Download size={10} strokeWidth={1.75} /> geojson
            </button>
          )}
        </header>

        {error && (
          <div className="border-b border-sentinel-danger/40 bg-sentinel-danger/5 px-4 py-2.5">
            <div className="flex items-center gap-2">
              <AlertTriangle size={12} className="text-sentinel-danger" />
              <span className="font-mono text-2xs text-sentinel-danger">{error}</span>
            </div>
          </div>
        )}

        {!result ? (
          <div className="flex flex-1 flex-col items-center justify-center px-8 text-center">
            <p className="font-display text-sm text-sentinel-muted-hi">No scene loaded</p>
            <p className="mt-1.5 max-w-[46ch] font-sans text-xs leading-relaxed text-sentinel-muted">
              Upload a GeoTIFF or pick a scene from the list. Validation runs before inference, so an
              unusable scene is reported as unusable rather than silently returning nothing.
            </p>
          </div>
        ) : (
          <div className="flex flex-col">
            <ProvenanceStrip
              state={
                result.state === 'invalid_scene'
                  ? 'failed'
                  : result.provenance?.checksum_sha256
                    ? 'real'
                    : 'unavailable'
              }
            >
              {scenePath ?? 'unknown scene'} · sha256{' '}
              {(result.provenance?.checksum_sha256 as string | undefined)?.slice(0, 12) ?? 'not recorded'} ·
              acquired {acquisition ?? 'unknown'}
            </ProvenanceStrip>

            <div className="px-4 py-3.5">
              <StateBanner result={result} />
            </div>

            <div className="grid grid-cols-2 gap-x-6 gap-y-0 px-4 pb-4">
              <div>
                <span className="ops-label">Validation</span>
                <div className="mt-1.5 border border-sentinel-border">
                  <Field label="valid" value={result.validation.valid ? 'YES' : 'NO'} />
                  <Field label="size" value={`${result.validation.width ?? '—'} × ${result.validation.height ?? '—'}`} />
                  <Field label="bands" value={String(result.validation.bands ?? '—')} />
                  <Field label="crs" value={result.validation.crs ?? 'missing'} />
                  <Field
                    label="pixel"
                    value={
                      result.validation.pixel_size_m
                        ? `${result.validation.pixel_size_m.toFixed(2)} m`
                        : '—'
                    }
                  />
                  <Field
                    label="valid px"
                    value={
                      result.validation.valid_fraction != null
                        ? `${(result.validation.valid_fraction * 100).toFixed(1)}%`
                        : '—'
                    }
                  />
                  {result.validation.reason_code && (
                    <Field label="reject" value={result.validation.reason_code} />
                  )}
                </div>
              </div>
              <div>
                <span className="ops-label">Tiling &amp; model</span>
                <div className="mt-1.5 border border-sentinel-border">
                  {Object.entries(result.tiling)
                    .slice(0, 4)
                    .map(([key, value]) => (
                      <Field key={key} label={key.replace(/_/g, ' ')} value={String(value)} />
                    ))}
                  <Field label="model" value={String(result.model.status ?? '—')} />
                  <Field
                    label="probabilities"
                    value={result.model.probabilities == null ? 'NONE EMITTED' : 'present'}
                  />
                </div>
              </div>
            </div>

            {result.detections.length > 0 && (
              <div className="px-4 pb-4">
                <div className="flex items-baseline gap-3">
                  <span className="ops-label">Detections</span>
                  <span className="font-mono text-3xs text-sentinel-muted">
                    ranked by area · click a row to inspect
                  </span>
                </div>
                <div className="mt-1.5 overflow-x-auto border border-sentinel-border">
                  <table className="ops-table">
                    <thead>
                      <tr>
                        <th>id</th>
                        <th className="text-right">km²</th>
                        <th className="text-right">len km</th>
                        <th className="text-right">wid km</th>
                        <th className="text-right">orient°</th>
                        <th className="text-right">ΔdB</th>
                        <th className="text-right">confidence</th>
                        <th className="text-right">centroid</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.detections.map((det) => (
                        <DetectionRow
                          key={det.id}
                          detection={det}
                          selected={det.id === selected?.id}
                          onSelect={() => setSelectedId(det.id)}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {selected && (
              <div className="grid grid-cols-2 gap-x-6 px-4 pb-4">
                <div>
                  <span className="ops-label">Geometry · {selected.id}</span>
                  <div className="mt-1.5 border border-sentinel-border">
                    <Field label="area" value={`${selected.area_km2.toFixed(4)} km²`} />
                    <Field label="perimeter" value={`${selected.perimeter_km?.toFixed(3)} km`} />
                    <Field label="length" value={`${selected.length_km.toFixed(3)} km`} />
                    <Field label="width" value={`${selected.width_km.toFixed(3)} km`} />
                    <Field label="orientation" value={`${selected.orientation_deg.toFixed(2)}°`} />
                    <Field label="elongation" value={String((selected as { elongation?: number }).elongation?.toFixed(3) ?? '—')} />
                    <Field label="method" value={String(selected.method ?? '—')} />
                  </div>
                </div>
                <div>
                  <span className="ops-label">Evidence</span>
                  <div className="mt-1.5 border border-sentinel-border">
                    <Field label="mean σ⁰" value={`${selected.mean_db?.toFixed(2) ?? '—'} dB`} />
                    <Field label="local contrast" value={`${selected.contrast_db?.toFixed(2) ?? '—'} dB`} />
                    <Field label="scene contrast" value={`${selected.scene_contrast_db?.toFixed(2) ?? '—'} dB`} />
                    <Field
                      label="fay age"
                      value={
                        selected.age_hours_fay != null ? `${selected.age_hours_fay.toFixed(1)} h` : '—'
                      }
                      hint="inverse spreading · regime-dependent"
                    />
                    <Field
                      label="wind viability"
                      value={selected.wind_viability ?? '—'}
                      hint={selected.wind_viability === 'OK' ? undefined : 'not validated against look-alikes'}
                    />
                  </div>
                  <div className="mt-2 border border-sentinel-border p-2.5">
                    <ConfidenceBar
                      value={selected.confidence.value ?? 0}
                      low={selected.confidence.low ?? 0}
                      high={selected.confidence.high ?? 0}
                    />
                    <p className="mt-1.5 font-mono text-3xs text-sentinel-muted">
                      {String(selected.confidence.method ?? 'wilson_95')} · n=
                      {String(selected.confidence.n_effective ?? '—')}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {result.explanation && Object.keys(result.explanation).length > 0 && (
              <div className="border-t border-sentinel-border px-4 py-3.5">
                <div className="flex items-center gap-2">
                  <span className="ops-label">Explanation</span>
                  <span className="font-mono text-3xs text-sentinel-muted">
                    {String(result.explanation.method ?? '')}
                  </span>
                  {result.explanation.gradcam === false && (
                    <span className="border border-sentinel-caution/40 px-1.5 py-0.5 font-mono text-3xs uppercase tracking-instrument text-sentinel-caution">
                      not grad-cam
                    </span>
                  )}
                </div>
                {/* Saying what this is NOT matters more than saying what it is:
                    an unlabelled heatmap reads as a neural attribution. */}
                <p className="mt-2 max-w-[78ch] font-serif text-base leading-relaxed text-sentinel-text">
                  {String(result.explanation.not_gradcam_reason ?? '')}
                </p>
                {Array.isArray(result.explanation.components) && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {(result.explanation.components as string[]).map((component) => (
                      <span
                        key={component}
                        className="border border-sentinel-border bg-sentinel-surface px-1.5 py-0.5 font-mono text-3xs text-sentinel-muted-hi"
                      >
                        {component}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── DRIFT & VESSELS ────────────────────────────────────────────── */}
      <section className="flex min-h-0 flex-col overflow-y-auto border-l border-sentinel-border bg-sentinel-surface">
        <header className="ops-header sticky top-0 z-10">
          <span className="ops-header-label">Drift &amp; vessels</span>
        </header>

        <div className="px-3.5 py-3">
          <button
            type="button"
            disabled={!selected || !acquisition || busy !== null}
            onClick={() => void runHindcast()}
            className="flex w-full items-center justify-center gap-2 border border-sentinel-border-hi px-3 py-1.5 font-display text-3xs uppercase tracking-instrument text-sentinel-text-hi transition-colors hover:border-sentinel-data hover:text-sentinel-data disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy === 'Hindcast' ? (
              <Loader2 size={11} className="animate-spin" />
            ) : (
              <Waves size={11} strokeWidth={1.75} />
            )}
            Backtrack 24 h
          </button>
          <p className="mt-1.5 font-sans text-3xs leading-relaxed text-sentinel-muted">
            {acquisition
              ? 'Anchored to the real acquisition instant, not to now — backtracking a 2020 slick with 2026 forcing would be nonsense.'
              : 'Load a scene with a recorded acquisition time to enable the hindcast.'}
          </p>
        </div>

        {drift && (
          <div className="px-3.5 pb-3">
            <ProvenanceStrip
              state={
                drift.wind_source === 'synthetic_constant' ||
                drift.current_source === 'synthetic_constant'
                  ? 'synthetic'
                  : 'real'
              }
            >
              wind · {drift.wind_source} / current · {drift.current_source}
            </ProvenanceStrip>

            <div className="mt-2 border border-sentinel-border">
              <Field label="origin lon" value={drift.origin.center_lon?.toFixed(4) ?? '—'} />
              <Field label="origin lat" value={drift.origin.center_lat?.toFixed(4) ?? '—'} />
              <Field label="semi-major (2σ)" value={`${drift.origin.semi_major_km.toFixed(2)} km`} />
              <Field label="semi-minor (2σ)" value={`${drift.origin.semi_minor_km.toFixed(2)} km`} />
              {/* p50/p95 are quantile radii: the ellipse assumes a Gaussian
                  cloud, these do not. Showing both is the difference between
                  reporting a distribution and reporting a fit. */}
              <Field label="p50 radius" value={`${drift.origin.p50_radius_km.toFixed(2)} km`} />
              <Field label="p95 radius" value={`${drift.origin.p95_radius_km.toFixed(2)} km`} />
              <Field label="particles" value={String(drift.origin.n_particles)} hint="surviving members" />
              <Field
                label="WMC ∇·K"
                value={String(drift.wmc_divergence_max)}
                hint="0 with constant K is a genuine no-op, not a skipped step"
              />
            </div>

            {drift.notes?.length > 0 && (
              <ul className="mt-2 space-y-1">
                {drift.notes.map((note) => (
                  <li key={note} className="font-sans text-3xs leading-relaxed text-sentinel-muted-hi">
                    · {note}
                  </li>
                ))}
              </ul>
            )}

            <button
              type="button"
              disabled={busy !== null}
              onClick={() => void runForecast()}
              className="mt-3 flex w-full items-center justify-center gap-2 border border-sentinel-border-hi px-3 py-1.5 font-display text-3xs uppercase tracking-instrument text-sentinel-text-hi transition-colors hover:border-sentinel-oil hover:text-sentinel-oil disabled:opacity-40"
            >
              {busy === 'Forecast' ? <Loader2 size={11} className="animate-spin" /> : <ArrowRight size={11} strokeWidth={1.75} />}
              Forecast 48 h
            </button>
          </div>
        )}

        {forecast && (
          <div className="px-3.5 pb-3">
            <span className="ops-label">Forecast · shoreline impact</span>
            <div className="mt-1.5 border border-sentinel-border">
              <Field
                label="beached"
                value={`${(forecast.stranded_fraction * 100).toFixed(1)}%`}
                hint="cumulative: beached at any step"
              />
              <Field
                label="first landfall"
                value={forecast.first_stranding_h != null ? `+${forecast.first_stranding_h} h` : 'none in window'}
              />
              <Field label="cone steps" value={String(forecast.cone.length)} />
            </div>
            {forecast.notes?.length > 0 && (
              <ul className="mt-2 space-y-1">
                {forecast.notes.map((note) => (
                  <li key={note} className="font-sans text-3xs leading-relaxed text-sentinel-muted-hi">
                    · {note}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <div className="ops-hairline" />

        <header className="ops-header">
          <Ship size={11} strokeWidth={1.75} className="text-sentinel-muted" />
          <span className="ops-header-label">Vessel evidence</span>
        </header>

        <div className="px-3.5 py-3">
          {!drift ? (
            <p className="font-sans text-2xs leading-relaxed text-sentinel-muted">
              Run the hindcast first. Vessel scoring depends on the inferred origin, so ranking
              suspects before an origin exists would be a guess dressed as evidence.
            </p>
          ) : drift.suspect_vessels?.length ? (
            <>
              {/* This endpoint ranks by proximity to the inferred origin. It is
                  NOT a scored attribution: there is no Wilson interval here, so
                  none is shown. Presenting a distance as if it were a
                  probability would be the exact failure this platform exists to
                  avoid — the scored, intervaled ranking lives on the case page. */}
              <p className="mb-2 font-sans text-3xs leading-relaxed text-sentinel-muted">
                Proximity ranking to the inferred origin. This is not a scored attribution and
                carries no confidence interval — open the case for the scored ranking.
              </p>
              <ul className="space-y-2">
                {drift.suspect_vessels.map((vessel, index) => (
                  <li
                    key={`${vessel.mmsi}-${index}`}
                    className="border border-sentinel-border p-2.5"
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="truncate font-mono text-xs text-sentinel-text-hi">
                        {vessel.name || vessel.mmsi}
                      </span>
                      <span className="ops-stencil">{String(index + 1).padStart(2, '0')}</span>
                    </div>
                    <div className="mt-1 font-mono text-3xs text-sentinel-muted">
                      MMSI {vessel.mmsi} · {vessel.flag ?? 'flag unknown'}
                    </div>
                    <div className="mt-1.5 flex items-baseline justify-between font-mono text-2xs">
                      <span className="text-sentinel-data">
                        {vessel.distance_to_origin_km.toFixed(2)} km
                      </span>
                      <span
                        className={
                          vessel.inside_p50
                            ? 'text-sentinel-oil'
                            : vessel.inside_p95
                              ? 'text-sentinel-caution'
                              : 'text-sentinel-muted'
                        }
                      >
                        {vessel.inside_p50 ? 'inside p50' : vessel.inside_p95 ? 'inside p95' : 'outside p95'}
                      </span>
                    </div>
                    <div className="mt-1 flex items-center gap-2">
                      <StatusGlyph
                        state={
                          vessel.provenance === 'synthetic_mock' ? 'synthetic' : 'real'
                        }
                      />
                      <span className="font-mono text-3xs text-sentinel-muted">
                        {vessel.wind_flag}
                        {vessel.age_h != null ? ` · ${vessel.age_h.toFixed(1)} h before pass` : ''}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <div className="border border-sentinel-border bg-sentinel-bg px-3 py-2.5">
              <StatusGlyph state="unavailable" />
              <p className="mt-1.5 font-sans text-2xs leading-relaxed text-sentinel-muted-hi">
                No real AIS coverage for this AOI. SENTINEL does not generate a suspect ranking
                without vessel evidence — an empty ranking with a reason is the honest answer.
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
