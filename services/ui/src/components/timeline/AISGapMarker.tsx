import { motion } from 'framer-motion';
import { Radio } from 'lucide-react';
import { COLORS } from '@/lib/constants';

interface AISGap {
  start: string;
  end: string;
  start_pos: { longitude: number; latitude: number };
  end_pos: { longitude: number; latitude: number };
}

interface AISGapMarkerProps {
  gap: AISGap;
  index: number;
}

export default function AISGapMarker({ gap, index }: AISGapMarkerProps) {
  const start = new Date(gap.start);
  const end = new Date(gap.end);
  const durationH = ((end.getTime() - start.getTime()) / (1000 * 60 * 60)).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.05 }}
      className="glass-panel border-sentinel-danger/30 p-3"
      style={{ borderLeftColor: COLORS.danger, borderLeftWidth: 3 }}
    >
      <div className="flex items-center gap-2">
        <Radio size={12} className="text-sentinel-danger" />
        <span className="font-mono text-xs font-semibold text-sentinel-danger">
          AIS DARK PERIOD
        </span>
        <span className="ml-auto font-mono text-[10px] text-sentinel-muted">
          {durationH}h gap
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-[10px] text-sentinel-muted">
        <div>
          <span className="text-sentinel-text/50">FROM:</span>{' '}
          {start.toISOString().slice(0, 16)}
          <br />
          <span className="text-sentinel-text/30">
            {gap.start_pos.latitude.toFixed(4)}N {gap.start_pos.longitude.toFixed(4)}E
          </span>
        </div>
        <div>
          <span className="text-sentinel-text/50">TO:</span>{' '}
          {end.toISOString().slice(0, 16)}
          <br />
          <span className="text-sentinel-text/30">
            {gap.end_pos.latitude.toFixed(4)}N {gap.end_pos.longitude.toFixed(4)}E
          </span>
        </div>
      </div>
    </motion.div>
  );
}
