import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Globe, ArrowLeft, AlertTriangle } from 'lucide-react';
import PageMeta from '@/components/shared/PageMeta';

export default function NotFound() {
  return (
    <div className="flex h-full items-center justify-center">
      <PageMeta 
        title="404 — Not Found" 
        description="The requested page was not found in the SENTINEL system." 
        path="/404" 
      />
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="ops-panel max-w-md p-8 text-center"
      >
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full" style={{ background: 'rgba(217,119,6,0.10)' }}>
          <AlertTriangle size={32} className="text-sentinel-caution" />
        </div>
        <h1 className="mb-2 font-display text-3xl font-bold text-sentinel-text-hi">
          404
        </h1>
        <h2 className="mb-1 font-display text-lg text-sentinel-text-hi">
          Sector Not Found
        </h2>
        <p className="mb-6 text-sm text-sentinel-muted">
          The coordinates you entered do not match any monitored maritime sector.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 rounded bg-sentinel-data px-4 py-2 font-display text-sm font-semibold text-white transition-colors hover:bg-sentinel-data-hi"
        >
          <ArrowLeft size={16} />
          Return to Watch Room
        </Link>
        <div className="mt-6 flex items-center justify-center gap-2 font-mono text-[10px] text-sentinel-muted">
          <Globe size={12} />
          SENTINEL Maritime Intelligence
        </div>
      </motion.div>
    </div>
  );
}
