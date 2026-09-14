import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Globe, Database, Search, Clock, Ship, FileText } from 'lucide-react';
import WatchRoom from './pages/WatchRoom';
import Archive from './pages/Archive';
import CaseInvestigation from './pages/CaseInvestigation';
import ForensicTimeline from './pages/ForensicTimeline';
import VesselProfile from './pages/VesselProfile';
import CaseFile from './pages/CaseFile';
import NotFound from './pages/NotFound';
import SiteSchema from './components/shared/SiteSchema';
import Breadcrumbs from './components/shared/Breadcrumbs';

const NAV_ITEMS = [
  { path: '/', label: 'WATCH ROOM', icon: Globe },
  { path: '/archive', label: 'ARCHIVE', icon: Database },
  { path: '/investigate/:caseId', label: 'INVESTIGATE', icon: Search },
  { path: '/timeline/:caseId', label: 'TIMELINE', icon: Clock },
  { path: '/vessel/:mmsi', label: 'VESSEL', icon: Ship },
  { path: '/casefile/:caseId', label: 'CASE FILE', icon: FileText },
] as const;

function Sidebar() {
  const location = useLocation();

  return (
    <aside className="fixed left-0 top-0 z-50 flex h-full w-16 flex-col items-center border-r border-sentinel-border bg-sentinel-surface/80 backdrop-blur-md">
      <div className="flex h-16 w-full items-center justify-center border-b border-sentinel-border">
        <span className="font-display text-sm font-bold tracking-widest text-sentinel-primary">
          S
        </span>
      </div>
      <nav className="mt-4 flex flex-1 flex-col gap-1">
        {NAV_ITEMS.slice(0, 4).map(({ path, icon: Icon }) => {
          const isActive =
            path === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(path.split('/:')[0] ?? '');
          return (
            <NavLink
              key={path}
              to={path.replace(':caseId', 'demo').replace(':mmsi', 'demo')}
              className="group relative flex h-10 w-10 items-center justify-center rounded-lg transition-colors"
            >
              {isActive && (
                <motion.div
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-lg bg-sentinel-primary/10"
                  transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                />
              )}
              <Icon
                size={20}
                className={
                  isActive
                    ? 'relative z-10 text-sentinel-primary'
                    : 'relative z-10 text-sentinel-muted group-hover:text-sentinel-text'
                }
              />
            </NavLink>
          );
        })}
      </nav>
    </aside>
  );
}

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-sentinel-bg font-sans text-sentinel-text">
      <SiteSchema />
      <Sidebar />
      <main className="ml-16 flex flex-1 flex-col overflow-hidden">
        <Breadcrumbs />
        <div className="flex-1 overflow-hidden relative">
          <AnimatePresence mode="wait">
            <motion.div
              key={useLocation().pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<WatchRoom />} />
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
