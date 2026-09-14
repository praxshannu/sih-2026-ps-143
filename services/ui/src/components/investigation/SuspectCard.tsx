/**
 * SuspectCard — shows one ranked suspect with dual scoring + SHAP breakdown.
 * No card shadows, no glassmorphism. Data-dense ops table style.
 */
import type { Suspect, PipelineSuspect } from '@/lib/api';
import ConfidenceBadge from '@/components/shared/ConfidenceBadge';

type AnySuspect = Suspect | PipelineSuspect;

function isPipeline(s: AnySuspect): s is PipelineSuspect {
  return 'fuzzy_score' in s;
}

function rank(s: AnySuspect) {
  return isPipeline(s) ? s.rank : s.rank;
}
function name(s: AnySuspect) {
  return isPipeline(s) ? s.vessel_name : s.vessel?.vessel_name ?? '—';
}
function mmsi(s: AnySuspect) {
  return isPipeline(s) ? s.mmsi : s.vessel?.mmsi ?? '—';
}
function score(s: AnySuspect) {
  return isPipeline(s) ? s.composite_score * 100 : s.score * 100;
}
function gap(s: AnySuspect) {
  return isPipeline(s) ? s.ais_gap_minutes : undefined;
}
function isDark(s: AnySuspect) {
  return isPipeline(s) ? s.is_dark_vessel : false;
}

interface Props {
  suspect: AnySuspect;
  selected?: boolean;
  onClick?: () => void;
}

// SHAP waterfall — simple bar list
function ShapRow({ label, value }: { label: string; value: number }) {
  const abs = Math.abs(value);
  const pct = Math.min(abs * 200, 100); // scale for display
  const pos = value > 0;
  return (
    <div className="flex items-center gap-2 py-0.5">
      <span className="ops-label w-28 flex-shrink-0">{label}</span>
      <div className="flex-1 h-1 bg-sentinel-border">
        <div
          className={`h-1 ${pos ? 'bg-sentinel-danger' : 'ops-nominal'}`}
          style={{ width: `${pct}%`, background: pos ? '#dc2626' : '#15803d' }}
        />
      </div>
      <span className={`font-mono text-2xs tabular-nums w-12 text-right ${pos ? 'ops-danger' : 'ops-nominal'}`}>
        {value > 0 ? '+' : ''}{value.toFixed(3)}
      </span>
    </div>
  );
}

export default function SuspectCard({ suspect, selected, onClick }: Props) {
  const pip = isPipeline(suspect) ? suspect as PipelineSuspect : null;
  const rankNum = rank(suspect);
  const vesselName = name(suspect);
  const mmsiVal = mmsi(suspect);
  const scoreVal = score(suspect);
  const gapMin = gap(suspect);
  const dark = isDark(suspect);

  return (
    <div
      onClick={onClick}
      className={`border-b border-sentinel-border cursor-pointer ${
        selected ? 'bg-sentinel-panel border-l-2 border-l-sentinel-amber' : 'hover:bg-sentinel-panel'
      }`}
    >
      {/* Header row */}
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-2xs text-sentinel-muted w-4">#{rankNum}</span>
          <div>
            <div className="font-mono text-xs text-sentinel-text-hi font-semibold">{vesselName}</div>
            <div className="font-mono text-2xs text-sentinel-muted">MMSI {mmsiVal}</div>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <ConfidenceBadge value={scoreVal} showBar size="sm" />
          {dark && (
            <span className="font-mono text-2xs ops-danger">[DARK VESSEL]</span>
          )}
        </div>
      </div>

      {/* Scores breakdown */}
      {pip && (
        <div className="border-t border-sentinel-border px-3 py-2">
          <div className="ops-row" style={{ borderBottom: 'none', padding: '2px 0' }}>
            <span className="ops-label">FUZZY SCORE</span>
            <span className="ops-value tabular-nums">{(pip.fuzzy_score * 100).toFixed(1)}%</span>
          </div>
          {pip.xgb_score != null && (
            <div className="ops-row" style={{ borderBottom: 'none', padding: '2px 0' }}>
              <span className="ops-label">XGB SCORE</span>
              <span className="ops-value tabular-nums">{(pip.xgb_score * 100).toFixed(1)}%</span>
            </div>
          )}
          {gapMin != null && (
            <div className="ops-row" style={{ borderBottom: 'none', padding: '2px 0' }}>
              <span className="ops-label">AIS GAP</span>
              <span className="ops-value tabular-nums">{gapMin.toFixed(0)} min</span>
            </div>
          )}
          {pip.confidence_lower != null && (
            <div className="ops-row" style={{ borderBottom: 'none', padding: '2px 0' }}>
              <span className="ops-label">95% CI</span>
              <span className="ops-value tabular-nums">
                [{(pip.confidence_lower * 100).toFixed(1)}, {(pip.confidence_upper * 100).toFixed(1)}]
              </span>
            </div>
          )}

          {/* SHAP breakdown */}
          {pip.shap_breakdown && Object.keys(pip.shap_breakdown).length > 0 && (
            <div className="mt-2 border-t border-sentinel-border pt-2">
              <div className="ops-header-label mb-1">SHAP ATTRIBUTION</div>
              {Object.entries(pip.shap_breakdown).map(([k, v]) => (
                <ShapRow key={k} label={k.replace(/_/g, ' ').toUpperCase()} value={v as number} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
