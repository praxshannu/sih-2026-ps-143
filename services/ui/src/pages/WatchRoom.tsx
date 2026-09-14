/**
 * WatchRoom — primary operator view.
 *
 * Design intent: NTRO ops console, not a SaaS dashboard.
 * - No hero typography, no gradient cards
 * - Left: globe (full height)
 * - Right: 280px data column — live case list, system status
 * - Top bar: UTC clock (updates every second), feed counts, LLM status
 * - Layer toggles: text checkboxes, not toggle switches
 */
import { useEffect, useRef, useState } from 'react';
import SentinelGlobe from '@/components/globe/SentinelGlobe';
import StatusPulse from '@/components/shared/StatusPulse';
import ConfidenceBadge from '@/components/shared/ConfidenceBadge';
import { useCasesStore } from '@/store/casesStore';
import { useMapStore } from '@/store/mapStore';
import { useWsStore } from '@/store/wsStore';
import api from '@/lib/api';

// ── UTC clock ─────────────────────────────────────────────────────────────
function UtcClock() {
  const [ts, setTs] = useState('');
  useEffect(() => {
    const tick = () => setTs(new Date().toISOString().replace('T', ' ').slice(0, 19) + 'Z');
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <span className="font-mono text-xs tabular-nums text-sentinel-amber cursor-blink">
      {ts}
    </span>
  );
}

// ── LLM status badge ──────────────────────────────────────────────────────
function LlmBadge() {
  const [state, setState] = useState<'checking' | 'live' | 'template'>('checking');
  const [model, setModel] = useState('');

  useEffect(() => {
    api.getLlmStatus()
      .then((s) => {
        setState(s.available ? 'live' : 'template');
        setModel(s.model ?? '');
      })
      .catch(() => setState('template'));
  }, []);

  const label =
    state === 'checking' ? 'LLM INIT'
    : state === 'live'    ? `LLM ${model.toUpperCase()}`
    :                       'LLM TEMPLATE';
  const cls =
    state === 'live' ? 'ops-nominal' : state === 'template' ? 'ops-caution' : 'text-sentinel-muted';

  return (
    <span className={`font-mono text-2xs uppercase tracking-wider ${cls}`}>
      [{label}]
    </span>
  );
}

// ── Layer toggle row ──────────────────────────────────────────────────────
function LayerRow({
  label, active, onToggle,
}: { label: string; active: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="ops-row cursor-pointer select-none hover:bg-sentinel-panel w-full text-left"
    >
      <span className="ops-label">{label}</span>
      <span
        className={`font-mono text-2xs uppercase ${active ? 'ops-amber' : 'text-sentinel-muted'}`}
      >
        {active ? '[ON]' : '[OFF]'}
      </span>
    </button>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────
export default function WatchRoom() {
  const cases        = useCasesStore((s) => s.cases);
  const fetchCases   = useCasesStore((s) => s.fetchCases);
  const loading      = useCasesStore((s) => s.loading);
  const connected    = useWsStore((s) => s.connected);
  const connect      = useWsStore((s) => s.connect);
  const showDrift    = useMapStore((s) => s.showDrift);
  const showVessels  = useMapStore((s) => s.showVesselTrails);
  const showSpills   = useMapStore((s) => s.showSpillPolygons);
  const autoRotate   = useMapStore((s) => s.isAutoRotating);
  const toggleDrift  = useMapStore((s) => s.toggleDrift);
  const toggleVessels= useMapStore((s) => s.toggleVesselTrails);
  const toggleSpills = useMapStore((s) => s.toggleSpillPolygons);
  const setRotate    = useMapStore((s) => s.setAutoRotating);

  useEffect(() => {
    fetchCases();
    connect();
  }, [fetchCases, connect]);

  const activeCases  = cases.filter((c) => c.status === 'ACTIVE').length;
  const totalSuspects = cases.reduce((n, c) => n + (c.suspects?.length ?? 0), 0);

  return (
    <div className="flex h-full flex-col">

      {/* ── Top bar ───────────────────────────────────────────────────── */}
      <div
        className="flex flex-shrink-0 items-center justify-between border-b border-sentinel-border bg-sentinel-surface px-4"
        style={{ height: 36 }}
      >
        <div className="flex items-center gap-4">
          <UtcClock />
          <span className="text-sentinel-muted text-2xs">|</span>
          <span className="font-mono text-2xs text-sentinel-muted uppercase tracking-wider">
            CASES&nbsp;<span className="text-sentinel-text-hi">{activeCases}</span>&nbsp;ACT
            &nbsp;/&nbsp;
            SUSPECTS&nbsp;<span className="text-sentinel-text-hi">{totalSuspects}</span>
          </span>
        </div>
        <div className="flex items-center gap-4">
          <LlmBadge />
          <StatusPulse
            status={connected ? 'active' : 'danger'}
            label={connected ? 'WS LIVE' : 'WS DOWN'}
            size="sm"
          />
        </div>
      </div>

      {/* ── Body: globe + right column ────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Globe */}
        <div className="relative flex-1 scanline-overlay">
          <SentinelGlobe />
        </div>

        {/* Right column */}
        <aside
          className="flex flex-shrink-0 flex-col border-l border-sentinel-border bg-sentinel-surface"
          style={{ width: 260 }}
        >

          {/* Layer control */}
          <div className="ops-header">
            <span className="ops-header-label">LAYER CONTROL</span>
          </div>
          <div className="ops-inset">
            <LayerRow label="SPILL ZONES"     active={showSpills}  onToggle={toggleSpills}  />
            <LayerRow label="DRIFT FORECAST"  active={showDrift}   onToggle={toggleDrift}   />
            <LayerRow label="VESSEL TRAILS"   active={showVessels} onToggle={toggleVessels} />
            <LayerRow
              label="AUTO ROTATE"
              active={autoRotate}
              onToggle={() => setRotate(!autoRotate)}
            />
          </div>

          {/* System status */}
          <div className="ops-header mt-0 border-t border-sentinel-border">
            <span className="ops-header-label">SYSTEM STATUS</span>
          </div>
          <div className="ops-inset">
            {[
              { label: 'SAR INGEST',   st: 'active'  as const },
              { label: 'AIS RECEIVER', st: 'active'  as const },
              { label: 'DRIFT ENGINE', st: 'active'  as const },
              { label: 'ML DETECT',    st: 'warning' as const },
              { label: 'LLM PIPELINE', st: 'warning' as const },
              { label: 'REDIS BROKER', st: connected ? 'active' as const : 'danger' as const },
            ].map(({ label, st }) => (
              <div key={label} className="ops-row">
                <span className="ops-label">{label}</span>
                <StatusPulse status={st} size="sm" />
              </div>
            ))}
          </div>

          {/* Active cases */}
          <div className="ops-header border-t border-sentinel-border">
            <span className="ops-header-label">ACTIVE CASES</span>
            {loading && (
              <span className="ml-auto font-mono text-2xs text-sentinel-muted">LOADING…</span>
            )}
          </div>
          <div className="flex-1 overflow-y-auto">
            {cases.length === 0 && !loading && (
              <div className="px-3 py-4 font-mono text-2xs text-sentinel-muted">
                NO CASES FOUND
              </div>
            )}
            {cases.map((c) => (
              <div
                key={c.id}
                className="border-b border-sentinel-border px-3 py-2 hover:bg-sentinel-panel cursor-pointer"
              >
                {/* Case ID row */}
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-2xs text-sentinel-amber font-semibold uppercase">
                    {c.id.slice(0, 12).toUpperCase()}
                  </span>
                  <span
                    className={`font-mono text-2xs uppercase ${
                      c.status === 'ACTIVE'    ? 'ops-amber'
                      : c.status === 'ESCALATED' ? 'ops-danger'
                      : 'text-sentinel-muted'
                    }`}
                  >
                    {c.status}
                  </span>
                </div>

                {/* Title */}
                <div className="font-mono text-xs text-sentinel-text-hi mb-1.5 leading-snug">
                  {c.title}
                </div>

                {/* Metrics row */}
                <div className="flex items-center justify-between">
                  <div className="flex gap-3">
                    <span className="ops-label">
                      {c.suspects?.length ?? 0} SUS
                    </span>
                    {c.spill?.area_km2 != null && (
                      <span className="ops-label">
                        {c.spill.area_km2.toFixed(1)} KM²
                      </span>
                    )}
                  </div>
                  {c.spill?.confidence != null && (
                    <ConfidenceBadge
                      value={c.spill.confidence * 100}
                      showBar={false}
                      showLabel={false}
                      size="sm"
                    />
                  )}
                </div>
              </div>
            ))}
          </div>

        </aside>
      </div>

    </div>
  );
}
