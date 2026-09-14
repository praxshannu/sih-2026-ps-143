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
 * Three deliberate choices, each a reaction to how this screen previously read:
 *
 * 1. **The rail carries words.** An icon-only rail is a guessing game: a ship
 *    outline, a clock and a magnifier are all "the thing I click to look at a
 *    case". Labels cost 150px and remove the guess. The rail collapses, so the
 *    cost is the analyst's to accept.
 * 2. **No blur, no radius, no spring.** The map is the hero; chrome that
 *    floats, glows and bounces competes with it. Chrome here is flat, square
 *    and still.
 * 3. **The footer is a real status line.** It reports which service each route
 *    depends on, so "the page is empty" is never a mystery. It is not a
 *    decorative health dot.
 */

type NavEntry = {
  path: string;
  label: string;
  icon: typeof Globe;
  /** Rendered as the footer's `depends on` readout when the route is active. */
  depends: string;
  /** Routes that need a live case id are shown but not linked from the rail. */
  parameterised?: boolean;
};

const NAV_ITEMS: NavEntry[] = [
  { path: '/', label: 'Watch Room', icon: Globe, depends: 'api · drift' },
  { path: '/intake', label: 'Scene Intake', icon: ScanLine, depends: 'detect · ingest' },
  { path: '/archive', label: 'Archive', icon: Database, depends: 'ingest · CDSE' },
  { path: '/investigate/:caseId', label: 'Investigate', icon: Search, depends: 'attribute · drift', parameterised: true },
  { path: '/timeline/:caseId', label: 'Timeline', icon: Clock, depends: 'api · store', parameterised: true },
  { path: '/vessel/:mmsi', label: 'Vessel', icon: Ship, depends: 'ingest · AIS', parameterised: true },
  { path: '/casefile/:caseId', label: 'Case File', icon: FileText, depends: 'intel · store', parameterised: true },
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
      className="relative z-30 flex h-full flex-col border-r border-sentinel-border bg-sentinel-surface"
      style={{ width: collapsed ? '3rem' : '13.5rem' }}
    >
      {/* Wordmark. The mark is a stencil `S` in a hairline box — a plate on
          equipment, not a logo lockup. */}
      <div className="flex h-14 items-center gap-2.5 border-b border-sentinel-border px-3">
        <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center border border-sentinel-border-hi font-display text-xs font-bold text-sentinel-oil">
          S
        </span>
        {!collapsed && (
          <span className="flex flex-col leading-none">
            <span className="font-display text-sm font-medium tracking-tight text-sentinel-text-hi">
              SENTINEL
            </span>
            <span className="mt-0.5 font-mono text-3xs uppercase tracking-instrument text-sentinel-muted">
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

        {/* Case-scoped routes need an id, so the rail lists them as
            unavailable rather than dead-linking to a placeholder case. */}
        {!collapsed && (
          <>
            <div className="mt-3 px-3.5 pb-1">
              <span className="font-mono text-3xs uppercase tracking-instrument text-sentinel-muted">
                Requires a case
              </span>
            </div>
            {NAV_ITEMS.filter((item) => item.parameterised).map(({ label, icon: Icon }) => (
              <span
                key={label}
                className="flex h-9 w-full cursor-not-allowed items-center gap-2.5 border-l-2 border-transparent px-3.5 text-sentinel-muted/60"
                title={`Open a case to reach ${label}`}
              >
                <Icon size={14} className="flex-shrink-0" strokeWidth={1.75} />
                <span className="nav-item-label truncate">{label}</span>
              </span>
            ))}
          </>
        )}
      </nav>

      <div className="border-t border-sentinel-border">
        {!collapsed && active && (
          <div className="px-3.5 py-2">
            <div className="font-mono text-3xs uppercase tracking-instrument text-sentinel-muted">
              Depends on
            </div>
            <div className="mt-0.5 font-mono text-2xs text-sentinel-data">{active.depends}</div>
          </div>
        )}
        <button
          type="button"
          onClick={onToggle}
          className="flex h-8 w-full items-center gap-2.5 border-t border-sentinel-border px-3.5 text-sentinel-muted transition-colors hover:text-sentinel-text-hi"
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

  // The rail collapses automatically on the two screens where horizontal
  // room is the whole point. It is a default, not a lock — the analyst can
  // expand it again and the choice sticks for the session.
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
        <Route path="/" element={<WatchRoom />} />
        <Route path="/intake" element={<SceneIntake />} />
        <Route path="/archive" element={<Archive />} />
        <Route path="/investigate/:caseId" element={<CaseInvestigation />} />
        <Route path="/timeline/:caseId" element={<ForensicTimeline />} />
        <Route path="/vessel/:mmsi" element={<VesselProfile />} />
        <Route path="/casefile/:caseId" element={<CaseFile />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Layout>
  );
}
