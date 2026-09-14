import { useState } from 'react';
import { motion } from 'framer-motion';
import { Map, Layers } from 'lucide-react';
import type { SpillEvent } from '@/lib/api';
import { MAP_STYLE } from '@/lib/constants';

interface EvidenceMapProps {
  spill: SpillEvent;
}

export default function EvidenceMap({ spill }: EvidenceMapProps) {
  const [activeLayer, setActiveLayer] = useState<'sar' | 'attribution'>('sar');

  return (
    <div className="glass-panel overflow-hidden">
      <div className="flex items-center justify-between border-b border-sentinel-border px-4 py-2">
        <div className="flex items-center gap-2">
          <Map size={14} className="text-sentinel-primary" />
          <span className="font-display text-sm font-semibold text-sentinel-text">
            Evidence Map
          </span>
        </div>
        <div className="flex gap-1">
          {(['sar', 'attribution'] as const).map((layer) => (
            <button
              key={layer}
              onClick={() => setActiveLayer(layer)}
              className={`rounded px-2 py-1 text-[10px] font-mono uppercase transition-colors ${
                activeLayer === layer
                  ? 'bg-sentinel-primary/20 text-sentinel-primary'
                  : 'text-sentinel-muted hover:text-sentinel-text'
              }`}
            >
              {layer === 'sar' ? 'SAR' : 'Attribution'}
            </button>
          ))}
        </div>
      </div>

      <div className="relative h-64 w-full bg-sentinel-bg">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <Layers size={32} className="mx-auto mb-2 text-sentinel-border" />
            <div className="font-mono text-xs text-sentinel-muted">
              {activeLayer === 'sar'
                ? 'SAR Composite Layer'
                : 'Attribution Analysis Layer'}
            </div>
            <div className="mt-1 font-mono text-[10px] text-sentinel-muted/60">
              Centroid: {spill.centroid.latitude.toFixed(4)}N{' '}
              {spill.centroid.longitude.toFixed(4)}E
            </div>
            <div className="mt-1 font-mono text-[10px] text-sentinel-warning">
              Area: {spill.area_km2.toFixed(1)} km&sup2; | Confidence: {spill.confidence}%
            </div>
          </div>
        </div>

        <div className="absolute bottom-2 left-2 rounded bg-sentinel-bg/80 px-2 py-1 font-mono text-[9px] text-sentinel-muted">
          MAP_STYLE: {activeLayer.toUpperCase()}
        </div>
      </div>
    </div>
  );
}
