import { useEffect, useState } from 'react';
import { NavLink, Route, Routes, useLocation } from 'react-router-dom';
import {
  ChevronLeft,
  Database,
  FileText,
  Globe,
  Search,
  Ship,
  Clock,
  ScanLine,
  Waves,
} from 'lucide-react';

import Archive from './pages/Archive';
import CaseFile from './pages/CaseFile';
import CaseInvestigation from './pages/CaseInvestigation';
import ForensicTimeline from './pages/ForensicTimeline';
import NotFound from './pages/NotFound';
import SceneIntake from './pages/SceneIntake';
import VesselProfile from './pages/VesselProfile';
import WatchRoom from './pages/WatchRoom';
import Breadcrumbs from './components/shared/Breadcrumbs';
import SiteSchema from './components/shared/SiteSchema';

/**
 * Shell.
 *
 * Dark slate nav rail (#1E293B) on a light (#F8FAFC) page.
 * The globe is the content hero — chrome stays out of its way.
 * The rail carries words; icon-only rails are guessing games.
 */

type NavEntry = {
  path: string;
  label: string;
  icon: typeof Globe;
  depends: string;
  parameterised?: boolean;
};

const NAV_ITEMS: NavEntry[] = [
  { path: '/',                    label: 'Watch Room',   icon: Globe,     depends: 'api · drift' },
  { path: '/intake',             label: 'Scene Intake', icon: ScanLine,  depends: 'detect · ingest' },
  { path: '/archive',            label: 'Archive',      icon: Database,  depends: 'ingest · CDSE' },
  { path: '/investigate/:caseId', label: 'Investigate',  icon: Search,    depends: 'attribute · drift', parameterised: true },
  { path: '/timeline/:caseId',   label: 'Timeline',     icon: Clock,     depends: 'api · store',        parameterised: true },
  { path: '/vessel/:mmsi',       label: 'Vessel',        icon: Ship,      depends: 'ingest · AIS',       parameterised: true },
  { path: '/casefile/:caseId',   label: 'Case File',    icon: FileText,  depends: 'intel · store',      parameterised: true },
];

function Rail({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { pathname } = useLocation();
  const active = NAV_ITEMS.find((item) =>
    item.path === '/'
      ? pathname === '/'
      : pathname.startsWith(item.path.split('/:')[0] ?? '\u0000'),
  );

  return (
    <aside
      className="relative z-30 flex h-full flex-col bg-sentinel-nav-bg border-r border-sentinel-nav-border"
      style={{
        width: collapsed ? '3.25rem' : '13.5rem',
      }}
    >
      {/* Wordmark — tactical cyan badge on dark rail */}
      <div className="flex h-14 items-center gap-2.5 px-3 border-b border-sentinel-nav-border">
        {/* S badge — glowing tactical cyan */}
        <span
          className="flex h-6 w-6 flex-shrink-0 items-center justify-center font-display text-xs font-black text-sentinel-bg animate-badge-pulse"
          style={{
            background: '#00D2FF',
            borderRadius: '2px',
            boxShadow: '0 0 10px rgba(0, 210, 255, 0.5)',
          }}
        >
          S
        </span>
        {!collapsed && (
          <span className="flex flex-col leading-none">
            <span className="font-display text-sm font-bold tracking-tight text-sentinel-text-hi">
              SENTINEL
            </span>
            <span className="mt-0.5 font-mono text-3xs uppercase tracking-instrument text-sentinel-data font-semibold">
              SIH26143 · NTRO
            </span>
          </span>
        )}
      </div>

      <nav className="flex flex-1 flex-col pt-2">
        {NAV_ITEMS.filter((item) => !item.parameterised).map(({ path, label, icon: Icon }) => {
          const isActive =
            path === '/' ? pathname === '/' : pathname.startsWith(path.split('/:')[0] ?? '');
          return (
            <NavLink
              key={path}
              to={path}
              title={collapsed ? label : undefined}
              className={`nav-item ${isActive ? 'active' : ''} ${collapsed ? 'justify-center px-0' : ''}`}
            >
              <Icon size={14} className="flex-shrink-0" strokeWidth={1.75} />
              {!collapsed && <span className="nav-item-label truncate">{label}</span>}
            </NavLink>
          );
        })}

        {/* Case-scoped routes — shown as unavailable when no case is open */}
        {!collapsed && (
          <>
            <div className="mt-3 px-3.5 pb-1">
              <span className="font-mono text-3xs uppercase tracking-instrument" style={{ color: '#475569' }}>
                Requires a case
              </span>
            </div>
            {NAV_ITEMS.filter((item) => item.parameterised).map(({ label, icon: Icon }) => (
              <span
                key={label}
                className="flex h-9 w-full cursor-not-allowed items-center gap-2.5 border-l-2 border-transparent px-3.5"
                style={{ color: '#475569', opacity: 0.55 }}
                title={`Open a case to reach ${label}`}
              >
                <Icon size={14} className="flex-shrink-0" strokeWidth={1.75} />
                <span className="nav-item-label truncate">{label}</span>
              </span>
            ))}
          </>
        )}
      </nav>

      {/* Footer: depends-on readout + collapse toggle */}
      <div className="border-t border-sentinel-nav-border">
        {!collapsed && active && (
          <div className="px-3.5 py-2">
            <div className="font-mono text-3xs uppercase tracking-instrument text-sentinel-muted">
              Depends on
            </div>
            <div className="mt-0.5 font-mono text-2xs text-sentinel-data font-semibold">
              {active.depends}
            </div>
          </div>
        )}
        <button
          type="button"
          onClick={onToggle}
          className="flex h-8 w-full items-center gap-2.5 px-3.5 transition-colors border-t border-sentinel-nav-border text-sentinel-muted hover:text-sentinel-text-hi"
          aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
        >
          <ChevronLeft
            size={13}
            strokeWidth={1.75}
            className="flex-shrink-0 transition-transform duration-600 ease-instrument"
            style={{ transform: collapsed ? 'rotate(180deg)' : 'none' }}
          />
          {!collapsed && (
            <span className="font-mono text-3xs uppercase tracking-instrument">Collapse</span>
          )}
        </button>
      </div>
    </aside>
  );
}

function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (pathname === '/' || pathname === '/archive') setCollapsed(true);
  }, [pathname]);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-sentinel-bg text-sentinel-text">
      <SiteSchema />
      <Rail collapsed={collapsed} onToggle={() => setCollapsed((value) => !value)} />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Breadcrumbs />
        <div className="relative min-h-0 flex-1 overflow-hidden">{children}</div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/"                     element={<WatchRoom />} />
        <Route path="/intake"              element={<SceneIntake />} />
        <Route path="/archive"             element={<Archive />} />
        <Route path="/investigate/:caseId" element={<CaseInvestigation />} />
        <Route path="/timeline/:caseId"    element={<ForensicTimeline />} />
        <Route path="/vessel/:mmsi"        element={<VesselProfile />} />
        <Route path="/casefile/:caseId"    element={<CaseFile />} />
        <Route path="*"                    element={<NotFound />} />
      </Routes>
    </Layout>
  );
}
