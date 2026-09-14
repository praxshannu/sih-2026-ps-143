import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Search, ArrowLeft } from 'lucide-react';
import { useCallback } from 'react';
import { useLiveCase } from '@/hooks/useLiveCase';
import { useSuspects } from '@/hooks/useSuspects';
import type { PipelineSuspect } from '@/lib/api';
import CaseStatusBanner from '@/components/shared/CaseStatusBanner';
import SuspectCard from '@/components/investigation/SuspectCard';
import EvidenceMap from '@/components/investigation/EvidenceMap';
import NarrativePane from '@/components/investigation/NarrativePane';
import StatusPulse from '@/components/shared/StatusPulse';
import { useMapStore } from '@/store/mapStore';

export default function CaseInvestigation() {
  const { caseId } = useParams<{ caseId: string }>();
  const { activeCase, loading, error, connected } = useLiveCase(caseId);
  const { suspects } = useSuspects(caseId);
  const flyTo = useMapStore((s) => s.flyTo);
  const selectVessel = useMapStore((s) => s.selectVessel);
  const selectedMmsi = useMapStore((s) => s.selectedMmsi);

  /**
   * Selecting a suspect always marks it. We only move the camera when the
   * payload actually carries a fix.
   *
   * The ranking pipeline reduces a vessel to `min_distance_nm` and never
   * retains its lat/lon, so today `latitude`/`longitude` are usually absent
   * and the camera correctly stays put. This used to call
   * `flyTo(73, 15)` — a hardcoded Arabian Sea point — for every suspect in
   * every case, which told the operator something false.
   */
  const focusSuspect = useCallback(
    (suspect: PipelineSuspect) => {
      selectVessel(suspect.mmsi);
      const { latitude, longitude } = suspect;
      if (typeof latitude === 'number' && typeof longitude === 'number') {
        flyTo(longitude, latitude);
      }
    },
    [flyTo, selectVessel],
  );

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <StatusPulse status="active" label="Loading case data..." size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="glass-panel p-6 text-center">
          <div className="font-mono text-sm text-sentinel-danger">{error}</div>
        </div>
      </div>
    );
  }

  if (!activeCase) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="glass-panel p-6 text-center">
          <Search size={32} className="mx-auto mb-2 text-sentinel-border" />
          <div className="font-mono text-sm text-sentinel-muted">No case data available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-sentinel-border bg-sentinel-surface/50 px-5 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <button className="rounded-lg bg-sentinel-bg/50 p-1.5 text-sentinel-muted hover:text-sentinel-text">
            <ArrowLeft size={16} />
          </button>
          <CaseStatusBanner
            status={activeCase.status}
            title={activeCase.title}
            caseId={activeCase.id}
            updatedAt={activeCase.updated_at}
          />
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <EvidenceMap spill={activeCase.spill} />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <NarrativePane narrative={activeCase.narrative} title="Case Summary" />
          </motion.div>

          {activeCase.drift_forecast && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="glass-panel p-4"
            >
              <h3 className="font-display text-sm font-semibold text-sentinel-text mb-2">
                Drift Forecast
              </h3>
              <div className="grid grid-cols-4 gap-3 font-mono text-[10px]">
                <div>
                  <div className="text-sentinel-muted">Hours</div>
                  <div className="text-sentinel-text">{activeCase.drift_forecast.hours}h</div>
                </div>
                <div>
                  <div className="text-sentinel-muted">Wind</div>
                  <div className="text-sentinel-text">
                    {activeCase.drift_forecast.wind_speed_ms.toFixed(1)} m/s
                  </div>
                </div>
                <div>
                  <div className="text-sentinel-muted">Current</div>
                  <div className="text-sentinel-text">
                    {activeCase.drift_forecast.current_speed_ms.toFixed(1)} m/s
                  </div>
                </div>
                <div>
                  <div className="text-sentinel-muted">Sea State</div>
                  <div className="text-sentinel-text">{activeCase.drift_forecast.sea_state}</div>
                </div>
              </div>
            </motion.div>
          )}
        </div>

        <aside className="w-96 border-l border-sentinel-border bg-sentinel-surface/30 overflow-y-auto">
          <div className="border-b border-sentinel-border p-4">
            <div className="flex items-center justify-between">
              <h2 className="font-display text-sm font-semibold text-sentinel-text">
                Suspects
              </h2>
              <span className="font-mono text-[10px] text-sentinel-muted">
                {suspects.length} ranked
              </span>
            </div>
          </div>
          <div className="p-3 space-y-2">
            {suspects.map((suspect) => (
              <SuspectCard
                key={suspect.mmsi}
                suspect={suspect}
                selected={selectedMmsi === suspect.mmsi}
                onClick={() => focusSuspect(suspect)}
              />
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}
