/**
 * StatusPulse — static square status indicator.
 * No animation rings. No glowing circles.
 * Real ops interfaces use fixed-color squares.
 */
interface StatusPulseProps {
  status: 'active' | 'warning' | 'danger' | 'success' | 'idle';
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

const STATUS_CLASS: Record<StatusPulseProps['status'], string> = {
  active: 'status-sq active',
  warning: 'status-sq caution',
  danger: 'status-sq danger',
  success: 'status-sq nominal',
  idle: 'status-sq offline',
};

const STATUS_LABEL_CLASS: Record<StatusPulseProps['status'], string> = {
  active: 'ops-amber',
  warning: 'ops-caution',
  danger: 'ops-danger',
  success: 'ops-nominal',
  idle: 'text-sentinel-muted',
};

export default function StatusPulse({ status, label, size = 'md' }: StatusPulseProps) {
  const textSize = size === 'sm' ? 'text-2xs' : size === 'lg' ? 'text-sm' : 'text-xs';
  return (
    <span className="inline-flex items-center gap-1.5 font-mono">
      <span className={STATUS_CLASS[status]} />
      {label && (
        <span className={`${textSize} uppercase tracking-wider ${STATUS_LABEL_CLASS[status]}`}>
          {label}
        </span>
      )}
    </span>
  );
}
