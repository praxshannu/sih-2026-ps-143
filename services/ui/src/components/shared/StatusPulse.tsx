/**
 * StatusPulse — Modern Glowing Status Indicator.
 */
interface StatusPulseProps {
  status: 'active' | 'warning' | 'danger' | 'success' | 'idle';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

const DOT_COLORS: Record<StatusPulseProps['status'], { dot: string; text: string }> = {
  active:  { dot: 'bg-blue-400',    text: 'text-blue-400' },
  warning: { dot: 'bg-amber-400',   text: 'text-amber-400' },
  danger:  { dot: 'bg-rose-500',    text: 'text-rose-400' },
  success: { dot: 'bg-emerald-400', text: 'text-emerald-400' },
  idle:    { dot: 'bg-slate-500',   text: 'text-slate-400' },
};

export default function StatusPulse({ status, label, size = 'md' }: StatusPulseProps) {
  const conf = DOT_COLORS[status] || DOT_COLORS.idle;
  const textSize = size === 'sm' ? 'text-[11px]' : size === 'lg' ? 'text-sm' : 'text-xs';

  return (
    <span className="inline-flex items-center gap-1.5 font-sans">
      <span className="relative flex h-2 w-2">
        {status !== 'idle' && (
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${conf.dot} opacity-60`} />
        )}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${conf.dot} shadow-sm`} />
      </span>
      {label && (
        <span className={`${textSize} font-semibold ${conf.text}`}>
          {label}
        </span>
      )}
    </span>
  );
}
