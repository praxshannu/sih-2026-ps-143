/**
 * ConfidenceBadge — Modern Numerical Confidence Display.
 */
import { CONFIDENCE } from '@/lib/constants';

interface ConfidenceBadgeProps {
  value: number;         // 0-100
  showBar?: boolean;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

function tier(v: number) {
  if (v >= CONFIDENCE.high) {
    return {
      color: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
      barHex: '#10B981',
      label: 'High',
    };
  }
  if (v >= CONFIDENCE.medium) {
    return {
      color: 'text-amber-400',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/20',
      barHex: '#F59E0B',
      label: 'Medium',
    };
  }
  return {
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/20',
    barHex: '#EF4444',
    label: 'Low',
  };
}

export default function ConfidenceBadge({
  value,
  showBar = true,
  showLabel = true,
  size = 'md',
}: ConfidenceBadgeProps) {
  const { color, bg, border, barHex, label } = tier(value);
  const pct = Math.max(0, Math.min(100, value));
  const textSize = size === 'sm' ? 'text-[11px]' : size === 'lg' ? 'text-sm' : 'text-xs';

  return (
    <span className="inline-flex items-center gap-2 font-mono">
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md ${textSize} font-semibold ${bg} ${color} border ${border}`}>
        <span>{pct.toFixed(0)}%</span>
        {showLabel && <span className="opacity-80 text-[10px] font-medium font-sans">({label})</span>}
      </span>
      {showBar && (
        <span className="h-1.5 w-10 bg-slate-800 rounded-full overflow-hidden inline-block">
          <span
            className="h-full rounded-full block transition-all duration-500"
            style={{ width: `${pct}%`, backgroundColor: barHex }}
          />
        </span>
      )}
    </span>
  );
}
