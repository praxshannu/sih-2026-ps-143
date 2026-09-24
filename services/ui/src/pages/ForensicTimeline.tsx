import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Clock, ArrowLeft, Filter } from 'lucide-react';
import ForensicBar from '@/components/timeline/ForensicBar';
import AISGapMarker from '@/components/timeline/AISGapMarker';
import StatusPulse from '@/components/shared/StatusPulse';
import { useCasesStore } from '@/store/casesStore';
import { COLORS } from '@/lib/constants';
import type { TimelineEvent } from '@/lib/api';

const SEVERITY_COLORS: Record<string, string> = {
  low: COLORS.success,
  medium: COLORS.warning,
  high: '#F97316',
  critical: COLORS.danger,
};

const TYPE_LABELS: Record<string, string> = {
  SPILL_DETECTED: 'Spill Detected',
  AIS_GAP: 'AIS Gap',
  VESSEL_SIGHTING: 'Vessel Sighting',
  SAR_PASS: 'SAR Pass',
  WEATHER_CHANGE: 'Weather Change',
  SUSPECT_FLAGGED: 'Suspect Flagged',
};

export default function ForensicTimeline() {
  const { caseId } = useParams<{ caseId: string }>();
  const timeline = useCasesStore((s) => s.timeline);
  const fetchTimeline = useCasesStore((s) => s.fetchTimeline);
  const loading = useCasesStore((s) => s.loading);
  const [severityFilter, setSeverityFilter] = useState<string | null>(null);

  useEffect(() => {
    if (caseId) fetchTimeline(caseId);
  }, [caseId, fetchTimeline]);

  const filtered = severityFilter
    ? timeline.filter((e) => e.severity === severityFilter)
    : timeline;

  const aisGaps = timeline.filter((e) => e.type === 'AIS_GAP');

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <StatusPulse status="active" label="Loading timeline..." size="lg" />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-sentinel-border bg-sentinel-surface px-5 py-3" style={{ boxShadow: '0 1px 3px 0 rgba(0,0,0,0.04)' }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button className="rounded bg-sentinel-panel p-1.5 text-sentinel-muted hover:text-sentinel-text-hi border border-sentinel-border transition-colors">
              <ArrowLeft size={16} />
            </button>
            <Clock size={16} className="text-sentinel-data" />
            <div>
              <h1 className="font-display text-sm font-bold text-sentinel-text-hi">
                FORENSIC TIMELINE
              </h1>
              <p className="font-mono text-[10px] text-sentinel-muted">
                CASE-{caseId?.slice(0, 8).toUpperCase()} | {timeline.length} events
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Filter size={12} className="text-sentinel-muted" />
            {['critical', 'high', 'medium', 'low'].map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(severityFilter === sev ? null : sev)}
                className={`rounded px-2 py-1 text-[9px] font-mono uppercase tracking-wider transition-colors ${
                  severityFilter === sev
                    ? 'bg-sentinel-data text-white'
                    : 'border border-sentinel-border text-sentinel-muted hover:border-sentinel-data hover:text-sentinel-data'
                }`}
              >
                {sev}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-sentinel-bg">
        <ForensicBar events={filtered} />

        <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
          {filtered.map((event, i) => (
            <motion.div
              key={event.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
              className="ops-panel p-3"
            >
              <div className="flex items-start gap-3">
                <div
                  className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded text-[10px] font-mono font-bold"
                  style={{
                    backgroundColor: `${SEVERITY_COLORS[event.severity]}15`,
                    color: SEVERITY_COLORS[event.severity],
                  }}
                >
                  {event.type === 'SPILL_DETECTED' ? '!' : event.type === 'AIS_GAP' ? '>>' : event.type === 'VESSEL_SIGHTING' ? '^' : event.type === 'SAR_PASS' ? 'S' : event.type === 'WEATHER_CHANGE' ? '~' : '*'}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-display text-xs font-semibold text-sentinel-text">
                      {event.title}
                    </span>
                    <span
                      className="rounded px-1.5 py-0.5 text-[8px] font-mono uppercase"
                      style={{
                        backgroundColor: `${SEVERITY_COLORS[event.severity]}15`,
                        color: SEVERITY_COLORS[event.severity],
                      }}
                    >
                      {event.severity}
                    </span>
                  </div>
                  <p className="text-[11px] text-sentinel-muted leading-relaxed">
                    {event.description}
                  </p>
                  <div className="mt-2 flex items-center gap-3 font-mono text-[9px] text-sentinel-muted/60">
                    <span>{new Date(event.timestamp).toISOString().slice(0, 19)}Z</span>
                    {event.related_mmsi && (
                      <span>MMSI: {event.related_mmsi}</span>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {aisGaps.length > 0 && (
          <div>
            <h3 className="mb-2 font-display text-xs font-semibold text-sentinel-danger uppercase">
              AIS Dark Periods ({aisGaps.length})
            </h3>
            <div className="space-y-2">
              {aisGaps.map((gap, i) => (
                <AISGapMarker
                  key={gap.id}
                  index={i}
                  gap={{
                    start: gap.timestamp,
                    end: gap.timestamp,
                    start_pos: { longitude: 73, latitude: 15 },
                    end_pos: { longitude: 73.5, latitude: 15.2 },
                  }}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
