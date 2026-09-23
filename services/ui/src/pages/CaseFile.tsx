import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileText, ArrowLeft, Download, Printer, Eye } from 'lucide-react';
import StatusPulse from '@/components/shared/StatusPulse';
import ConfidenceBadge from '@/components/shared/ConfidenceBadge';
import NarrativePane from '@/components/investigation/NarrativePane';
import { useLiveCase } from '@/hooks/useLiveCase';
import type { Evidence } from '@/lib/api';
import { COLORS } from '@/lib/constants';

const EVIDENCE_TYPE_ICONS: Record<string, string> = {
  SAR_IMAGE: 'SAR',
  AIS_DATA: 'AIS',
  WEATHER: 'WMO',
  SATELLITE: 'SAT',
  MANIFEST: 'DOC',
};

export default function CaseFile() {
  const { caseId } = useParams<{ caseId: string }>();
  const { activeCase, loading } = useLiveCase(caseId);
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <StatusPulse status="active" label="Loading case file..." size="lg" />
      </div>
    );
  }

  if (!activeCase) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="glass-panel p-6 text-center">
          <FileText size={32} className="mx-auto mb-2 text-sentinel-border" />
          <div className="font-mono text-sm text-sentinel-muted">No case file available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="border-b border-sentinel-border bg-sentinel-surface px-5 py-3" style={{ boxShadow: '0 1px 3px 0 rgba(0,0,0,0.04)' }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button className="rounded bg-sentinel-panel p-1.5 text-sentinel-muted hover:text-sentinel-text-hi border border-sentinel-border transition-colors">
              <ArrowLeft size={16} />
            </button>
            <FileText size={16} className="text-sentinel-data" />
            <div>
              <h1 className="font-display text-sm font-bold text-sentinel-text-hi">
                CASE FILE
              </h1>
              <p className="font-mono text-[10px] text-sentinel-muted">
                CASE-{caseId?.slice(0, 8).toUpperCase()} | Legal Evidence Viewer
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-ghost flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono">
              <Printer size={12} />
              Print
            </button>
            <button className="btn-ghost flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono">
              <Download size={12} />
              Export
            </button>
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-sentinel-bg">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <NarrativePane
              narrative={activeCase.narrative}
              title="Executive Summary"
            />
          </motion.div>

          <div className="ops-panel p-4">
            <h3 className="mb-3 font-display text-xs font-semibold text-sentinel-text-hi uppercase">
              Evidence Chain of Custody
            </h3>
            <div className="space-y-2">
              {activeCase.evidence.map((ev, i) => (
                <motion.div
                  key={ev.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className={`ops-panel cursor-pointer p-3 transition-all ${
                    selectedEvidence?.id === ev.id
                      ? 'ring-2 ring-sentinel-data border-sentinel-data'
                      : 'hover:shadow-card-hover'
                  }`}
                  onClick={() => setSelectedEvidence(ev)}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded font-mono text-[9px] font-bold"
                      style={{
                        backgroundColor: 'rgba(64, 150, 255, 0.10)',
                        color: '#4096FF',
                      }}>

                      {EVIDENCE_TYPE_ICONS[ev.type] ?? '??'}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-display text-xs font-semibold text-sentinel-text">
                          {ev.title}
                        </span>
                        <ConfidenceBadge value={ev.confidence} size="sm" />
                      </div>
                      <p className="mt-0.5 text-[10px] text-sentinel-muted truncate">
                        {ev.description}
                      </p>
                    </div>
                    <Eye size={14} className="text-sentinel-muted" />
                  </div>
                  <div className="mt-2 flex items-center gap-3 font-mono text-[9px] text-sentinel-muted/60">
                    <span>{new Date(ev.timestamp).toISOString().slice(0, 19)}Z</span>
                    <span>Source: {ev.source}</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {selectedEvidence && (
          <aside className="w-96 border-l border-sentinel-border bg-sentinel-surface overflow-y-auto">
            <div className="border-b border-sentinel-border p-4 bg-sentinel-panel">
              <h3 className="font-display text-sm font-semibold text-sentinel-text-hi">
                {selectedEvidence.title}
              </h3>
              <p className="mt-1 text-[10px] text-sentinel-muted-hi">
                {selectedEvidence.type.replace('_', ' ')}
              </p>
            </div>
            <div className="p-4 space-y-4">
              <div className="ops-inset p-3">
                <div className="text-[10px] text-sentinel-muted font-medium mb-1 uppercase tracking-wider">Description</div>
                <p className="text-xs text-sentinel-text-hi leading-relaxed">
                  {selectedEvidence.description}
                </p>
              </div>

              <div className="ops-inset p-3">
                <div className="text-[10px] text-sentinel-muted font-medium mb-1 uppercase tracking-wider">Source</div>
                <div className="font-mono text-xs text-sentinel-text-hi">
                  {selectedEvidence.source}
                </div>
              </div>

              <div className="ops-inset p-3">
                <div className="text-[10px] text-sentinel-muted font-medium mb-1 uppercase tracking-wider">Timestamp</div>
                <div className="font-mono text-xs text-sentinel-text-hi">
                  {new Date(selectedEvidence.timestamp).toISOString()}
                </div>
              </div>

              <div className="ops-inset p-3">
                <div className="text-[10px] text-sentinel-muted font-medium mb-1 uppercase tracking-wider">Confidence</div>
                <ConfidenceBadge value={selectedEvidence.confidence} size="lg" />
              </div>

              <div className="ops-inset p-3">
                <div className="text-[10px] text-sentinel-muted font-medium mb-2 uppercase tracking-wider">Evidence Preview</div>
                <div className="flex h-48 items-center justify-center rounded bg-sentinel-surface border border-dashed border-sentinel-border">
                  <div className="text-center">
                    <FileText size={24} className="mx-auto mb-2 text-sentinel-border-hi" />
                    <div className="font-mono text-[10px] text-sentinel-muted">
                      {selectedEvidence.url ? 'Evidence file' : 'No preview available'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
