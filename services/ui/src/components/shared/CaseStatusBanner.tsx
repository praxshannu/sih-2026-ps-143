/**
 * CaseStatusBanner — header strip for the case investigation page.
 * No glassmorphism, no animated slide-in, no rounded icon containers.
 */
import type { CaseStatus } from '@/lib/constants';

interface CaseStatusBannerProps {
  status: CaseStatus;
  title: string;
  caseId: string;
  updatedAt: string;
}

const STATUS_CLS: Record<CaseStatus, string> = {
  ACTIVE:    'ops-amber',
  CLOSED:    'ops-nominal',
  PENDING:   'ops-caution',
  ESCALATED: 'ops-danger',
};

export default function CaseStatusBanner({
  status, title, caseId, updatedAt,
}: CaseStatusBannerProps) {
  const cls = STATUS_CLS[status];

  return (
    <div className="ops-header justify-between border-b border-sentinel-border">
      <div className="flex items-center gap-3 min-w-0">
        <span className={`font-mono text-2xs font-semibold uppercase tracking-wider ${cls}`}>
          [{status}]
        </span>
        <span className="font-mono text-xs text-sentinel-text-hi truncate">{title}</span>
        <span className="font-mono text-2xs text-sentinel-muted">
          #{caseId.slice(0, 8).toUpperCase()}
        </span>
      </div>
      <div className="font-mono text-2xs text-sentinel-muted flex-shrink-0">
        UPD {new Date(updatedAt).toISOString().slice(0, 19)}Z
      </div>
    </div>
  );
}
