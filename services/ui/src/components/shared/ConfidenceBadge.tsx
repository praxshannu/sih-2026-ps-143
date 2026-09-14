/**
 * ConfidenceBadge — numerical confidence display.
 * No rounded pill. No glow. Plain text with a thin bar beneath.
 * High ≥75% → nominal green
 * Med 45-74% → caution amber
 * Low <45% → danger red
 */
import { CONFIDENCE } from '@/lib/constants';

interface ConfidenceBadgeProps {
  value: number;         // 0-100
  showBar?: boolean;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

function tier(v: number) {
  if (v >= CONFIDENCE.high) return { cls: 'ops-nominal', label: 'HIGH', bar: '#15803d' };
  if (v >= CONFIDENCE.medium) return { cls: 'ops-caution', label: 'MED', bar: '#ca8a04' };
  return { cls: 'ops-danger', label: 'LOW', bar: '#dc2626' };
}

export default function ConfidenceBadge({
  value,
  showBar = true,
  showLabel = true,
  size = 'md',
}: ConfidenceBadgeProps) {
  const { cls, label, bar } = tier(value);
  const pct = Math.max(0, Math.min(100, value));
  const textSize = size === 'sm' ? 'text-2xs' : size === 'lg' ? 'text-sm' : 'text-xs';

  return (
    <span className="inline-flex flex-col gap-0.5 font-mono tabular-nums">
      <span className={`${textSize} font-semibold ${cls}`}>
        {pct.toFixed(1)}%{showLabel && <span className="ml-1 opacity-60">{label}</span>}
      </span>
      {showBar && (
        <span className="conf-bar-track w-12">
          <span
            className="conf-bar-fill block"
            style={{ width: `${pct}%`, background: bar }}
          />
        </span>
      )}
    </span>
  );
}
