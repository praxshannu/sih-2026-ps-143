import { Link, useLocation } from 'react-router-dom';
import { ChevronRight, Globe } from 'lucide-react';

const ROUTE_LABELS: Record<string, string> = {
  investigate: 'Investigation',
  timeline: 'Forensic Timeline',
  vessel: 'Vessel Profile',
  casefile: 'Case File',
};

export default function Breadcrumbs() {
  const { pathname } = useLocation();

  if (pathname === '/') return null;

  const segments = pathname.split('/').filter(Boolean);
  const crumbs: Array<{ label: string; to: string }> = [
    { label: 'Watch Room', to: '/' },
  ];

  if (segments[0] && ROUTE_LABELS[segments[0]]) {
    crumbs.push({
      label: ROUTE_LABELS[segments[0]],
      to: segments[1] ? `/${segments[0]}/${segments[1]}` : `/${segments[0]}`,
    });
  }

  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 px-5 py-2 font-mono text-[11px] bg-sentinel-surface border-b border-sentinel-border">
      <Globe size={12} className="text-sentinel-muted" />
      {crumbs.map((crumb, i) => (
        <span key={crumb.to} className="flex items-center gap-1.5">
          {i > 0 && <ChevronRight size={10} className="text-sentinel-border-hi" />}
          {i === crumbs.length - 1 ? (
            <span className="text-sentinel-text-hi font-semibold" aria-current="page">{crumb.label}</span>
          ) : (
            <a href={crumb.to} className="text-sentinel-muted hover:text-sentinel-data transition-colors">
              {crumb.label}
            </a>
          )}
        </span>
      ))}
    </nav>
  );
}
