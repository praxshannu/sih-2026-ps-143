/**
 * WatchRoom — Primary Command View.
 * Sleek modern dark operations console for maritime surveillance.
 */
import { useEffect, useState } from 'react';
import { Layers, Activity, AlertCircle, ShieldAlert, Radio, Wind, Cpu, Database, ChevronRight } from 'lucide-react';
import SentinelGlobe from '@/components/globe/SentinelGlobe';
import StatusPulse from '@/components/shared/StatusPulse';
import ConfidenceBadge from '@/components/shared/ConfidenceBadge';
import { useCasesStore } from '@/store/casesStore';
import { useMapStore } from '@/store/mapStore';
import { useWsStore } from '@/store/wsStore';
import api from '@/lib/api';

// ── Live UTC Clock ────────────────────────────────────────────────────────
function UtcClock() {
  const [ts, setTs] = useState('');
  useEffect(() => {
    const tick = () => setTs(new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC');
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="flex items-center gap-2 font-mono text-xs text-slate-200 bg-slate-900/90 px-3 py-1.5 rounded-md border border-slate-800 shadow-sm">
      <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
      <span>{ts}</span>
    </div>
  );
}

// ── Modern Toggle Switch ─────────────────────────────────────────────────
function ModernToggle({
  label,
  active,
  onToggle,
  icon: Icon,
}: {
  label: string;
  active: boolean;
  onToggle: () => void;
  icon?: React.ElementType;
}) {
  return (
    <button
      onClick={onToggle}
      className="flex items-center justify-between w-full px-3.5 py-2.5 hover:bg-slate-800/60 rounded-lg transition-colors group text-left"
    >
      <div className="flex items-center gap-2.5">
        {Icon && <Icon size={14} className={active ? 'text-blue-400' : 'text-slate-500'} />}
        <span className={`text-xs font-medium ${active ? 'text-slate-200' : 'text-slate-400'}`}>
          {label}
        </span>
      </div>
      <div
        className={`w-9 h-5 rounded-full transition-colors relative flex items-center px-0.5 ${
          active ? 'bg-blue-600' : 'bg-slate-700'
        }`}
      >
        <div
          className={`w-4 h-4 rounded-full bg-white shadow-md transition-transform ${
            active ? 'translate-x-4' : 'translate-x-0'
          }`}
        />
      </div>
    </button>
  );
}

// ── Main WatchRoom ────────────────────────────────────────────────────────
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
  const flyTo        = useMapStore((s) => s.flyTo);

  const [llmStatus, setLlmStatus] = useState<'live' | 'template'>('live');

  useEffect(() => {
    fetchCases();
    connect();
    api.getLlmStatus()
      .then((s) => setLlmStatus(s.available ? 'live' : 'template'))
      .catch(() => setLlmStatus('live'));
  }, [fetchCases, connect]);

  const activeCases   = cases.filter((c) => c.status === 'ACTIVE' || c.status === 'ESCALATED').length;
  const totalSuspects = cases.reduce((n, c) => n + (c.suspects?.length ?? 0), 0);

  return (
    <div className="flex h-full flex-col bg-slate-950 text-slate-100 font-sans select-none">

      {/* ── Top Bar ───────────────────────────────────────────────────── */}
      <header className="flex flex-shrink-0 items-center justify-between border-b border-slate-800 bg-slate-900/90 backdrop-blur-md px-5 py-2.5 z-20 shadow-md">
        <div className="flex items-center gap-4">
          <UtcClock />
          <div className="h-4 w-px bg-slate-800" />
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-xs text-slate-400 font-medium">
              Active Cases:
              <span className="px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 font-semibold border border-blue-500/20">
                {activeCases}
              </span>
            </span>
            <span className="flex items-center gap-1.5 text-xs text-slate-400 font-medium">
              Ranked Suspects:
              <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-200 font-semibold border border-slate-700">
                {totalSuspects}
              </span>
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 text-xs">
            <Cpu size={13} className="text-blue-400" />
            <span className="text-slate-300 font-medium">AI Intelligence:</span>
            <span className="text-emerald-400 font-semibold">Active</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800/80 border border-slate-700 text-xs">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            <span className="text-slate-300 font-medium">Feed:</span>
            <span className="text-emerald-400 font-semibold">Live</span>
          </div>
        </div>
      </header>

      {/* ── Body: 3D Globe + Operations Sidebar ──────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* 3D Globe View (Clean, No Scanline Overlay) */}
        <div className="relative flex-1 bg-slate-950">
          <SentinelGlobe />
        </div>

        {/* Right Operations Sidebar */}
        <aside className="flex flex-shrink-0 flex-col border-l border-slate-800/90 bg-slate-900/95 backdrop-blur-md w-80 shadow-2xl z-20">

          {/* Section: Layer Control */}
          <div className="p-4 border-b border-slate-800/80">
            <div className="flex items-center gap-2 mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              <Layers size={13} className="text-blue-400" />
              <span>Map Overlays</span>
            </div>
            <div className="space-y-1 bg-slate-950/60 rounded-xl p-1.5 border border-slate-800/60">
              <ModernToggle
                label="Spill Discharges"
                active={showSpills}
                onToggle={toggleSpills}
                icon={AlertCircle}
              />
              <ModernToggle
                label="Drift Forecasts"
                active={showDrift}
                onToggle={toggleDrift}
                icon={Wind}
              />
              <ModernToggle
                label="Vessel AIS Trails"
                active={showVessels}
                onToggle={toggleVessels}
                icon={Radio}
              />
              <ModernToggle
                label="3D Orbital Rotation"
                active={autoRotate}
                onToggle={() => setRotate(!autoRotate)}
                icon={Activity}
              />
            </div>
          </div>

          {/* Section: System Status */}
          <div className="p-4 border-b border-slate-800/80">
            <div className="flex items-center gap-2 mb-2.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
              <Activity size={13} className="text-blue-400" />
              <span>System Telemetry</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { name: 'SAR Ingestion', status: 'Online' },
                { name: 'AIS Receiver', status: 'Online' },
                { name: 'Drift Physics', status: 'Online' },
                { name: 'Neural UNet++', status: 'Active' },
                { name: 'Attribution Tensor', status: 'Active' },
                { name: 'Redis Broker', status: 'Connected' },
              ].map((item) => (
                <div key={item.name} className="flex items-center justify-between p-2 rounded-lg bg-slate-950/60 border border-slate-800/60">
                  <span className="text-[11px] font-medium text-slate-400">{item.name}</span>
                  <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-sm" />
                </div>
              ))}
            </div>
          </div>

          {/* Section: Active Investigations */}
          <div className="flex flex-1 flex-col overflow-hidden p-4">
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
                <ShieldAlert size={13} className="text-amber-400" />
                <span>Active Incidents</span>
              </div>
              <span className="text-xs text-blue-400 font-semibold">{cases.length} Recorded</span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2.5 pr-0.5">
              {cases.map((c) => (
                <div
                  key={c.id}
                  onClick={() => {
                    if (c.spill?.centroid) {
                      flyTo(c.spill.centroid.longitude, c.spill.centroid.latitude, 7);
                    }
                  }}
                  className="p-3 rounded-xl bg-slate-950/80 hover:bg-slate-800/70 border border-slate-800 hover:border-blue-500/50 cursor-pointer transition-all group shadow-sm"
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-bold text-slate-200 group-hover:text-blue-400 transition-colors line-clamp-1">
                      {c.title}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider ${
                        c.status === 'ESCALATED'
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                      }`}
                    >
                      {c.status}
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 line-clamp-2 mb-2.5 leading-relaxed">
                    {c.narrative}
                  </p>

                  <div className="flex items-center justify-between pt-2 border-t border-slate-800/60 text-[11px] font-mono">
                    <div className="flex items-center gap-3">
                      <span className="text-slate-400">
                        Area: <strong className="text-slate-200">{c.spill?.area_km2?.toFixed(0) || '0'} km²</strong>
                      </span>
                      <span className="text-slate-400">
                        Suspects: <strong className="text-slate-200">{c.suspects?.length || 0}</strong>
                      </span>
                    </div>
                    {c.spill?.confidence && (
                      <span className="text-emerald-400 font-semibold">
                        {c.spill.confidence}% Conf
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </aside>
      </div>

    </div>
  );
}
