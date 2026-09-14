import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Ship, ArrowLeft, Anchor, MapPin, AlertTriangle, Shield, Clock } from 'lucide-react';
import StatusPulse from '@/components/shared/StatusPulse';
import ConfidenceBadge from '@/components/shared/ConfidenceBadge';
import { api, type VesselProfile as VesselProfileType } from '@/lib/api';
import { COLORS } from '@/lib/constants';

export default function VesselProfile() {
  const { mmsi } = useParams<{ mmsi: string }>();
  const [vessel, setVessel] = useState<VesselProfileType | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!mmsi) return;
    setLoading(true);
    api
      .getVessel(mmsi)
      .then(setVessel)
      .finally(() => setLoading(false));
  }, [mmsi]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <StatusPulse status="active" label="Loading vessel profile..." size="lg" />
      </div>
    );
  }

  if (!vessel) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="glass-panel p-6 text-center">
          <Ship size={32} className="mx-auto mb-2 text-sentinel-border" />
          <div className="font-mono text-sm text-sentinel-muted">No vessel data</div>
        </div>
      </div>
    );
  }

  const sections = [
    {
      title: 'IDENTITY',
      icon: Ship,
      items: [
        { label: 'Name', value: vessel.name },
        { label: 'MMSI', value: vessel.mmsi },
        { label: 'IMO', value: vessel.imo },
        { label: 'Flag', value: vessel.flag },
        { label: 'Type', value: vessel.vessel_type },
      ],
    },
    {
      title: 'DIMENSIONS',
      icon: Anchor,
      items: [
        { label: 'Gross Tonnage', value: vessel.gross_tonnage.toLocaleString() + ' GT' },
        { label: 'Length', value: vessel.length + ' m' },
        { label: 'Beam', value: vessel.beam + ' m' },
        { label: 'Built', value: vessel.built.toString() },
      ],
    },
    {
      title: 'OPERATOR',
      icon: Shield,
      items: [
        { label: 'Owner', value: vessel.owner },
        { label: 'Operator', value: vessel.operator },
        { label: 'Registered', value: vessel.registered_address },
      ],
    },
  ];

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-sentinel-border bg-sentinel-surface/50 px-5 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <button className="rounded-lg bg-sentinel-bg/50 p-1.5 text-sentinel-muted hover:text-sentinel-text">
            <ArrowLeft size={16} />
          </button>
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-sentinel-primary/10">
            <Ship size={16} className="text-sentinel-primary" />
          </div>
          <div>
            <h1 className="font-display text-sm font-bold text-sentinel-text">
              {vessel.name}
            </h1>
            <p className="font-mono text-[10px] text-sentinel-muted">
              MMSI {vessel.mmsi} | IMO {vessel.imo}
            </p>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <ConfidenceBadge value={100 - vessel.risk_score} size="md" />
            <StatusPulse
              status={vessel.risk_score > 70 ? 'danger' : vessel.risk_score > 40 ? 'warning' : 'success'}
              label={vessel.risk_score > 70 ? 'HIGH RISK' : vessel.risk_score > 40 ? 'MEDIUM RISK' : 'LOW RISK'}
            />
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {sections.map((section, i) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass-panel p-4"
            >
              <div className="mb-3 flex items-center gap-2">
                <section.icon size={14} className="text-sentinel-primary" />
                <h3 className="font-display text-xs font-semibold text-sentinel-text uppercase">
                  {section.title}
                </h3>
              </div>
              <div className="space-y-2">
                {section.items.map(({ label, value }) => (
                  <div key={label} className="flex justify-between font-mono text-[11px]">
                    <span className="text-sentinel-muted">{label}</span>
                    <span className="text-sentinel-text">{value}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="glass-panel p-4"
        >
          <h3 className="mb-3 flex items-center gap-2 font-display text-xs font-semibold text-sentinel-text uppercase">
            <AlertTriangle size={14} className="text-sentinel-danger" />
            Risk Assessment
          </h3>
          <div className="grid grid-cols-4 gap-3">
            <div className="rounded-lg bg-sentinel-bg/50 p-3 text-center">
              <div className="font-mono text-lg font-bold text-sentinel-danger">
                {vessel.risk_score}
              </div>
              <div className="text-[9px] text-sentinel-muted">RISK SCORE</div>
            </div>
            <div className="rounded-lg bg-sentinel-bg/50 p-3 text-center">
              <div className="font-mono text-lg font-bold text-sentinel-warning">
                {vessel.prior_incidents}
              </div>
              <div className="text-[9px] text-sentinel-muted">PRIOR INCIDENTS</div>
            </div>
            <div className="rounded-lg bg-sentinel-bg/50 p-3 text-center">
              <div className={`font-mono text-lg font-bold ${vessel.sanctions_history ? 'text-sentinel-danger' : 'text-sentinel-success'}`}>
                {vessel.sanctions_history ? 'YES' : 'NO'}
              </div>
              <div className="text-[9px] text-sentinel-muted">SANCTIONS</div>
            </div>
            <div className="rounded-lg bg-sentinel-bg/50 p-3 text-center">
              <div className="font-mono text-lg font-bold text-sentinel-primary">
                {vessel.flag}
              </div>
              <div className="text-[9px] text-sentinel-muted">FLAG STATE</div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-panel p-4"
        >
          <h3 className="mb-3 flex items-center gap-2 font-display text-xs font-semibold text-sentinel-text uppercase">
            <MapPin size={14} className="text-sentinel-primary" />
            Last Known Position
          </h3>
          <div className="flex items-center gap-4 font-mono text-[11px]">
            <span className="text-sentinel-muted">
              {vessel.last_known_position.latitude.toFixed(4)}N{' '}
              {vessel.last_known_position.longitude.toFixed(4)}E
            </span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
