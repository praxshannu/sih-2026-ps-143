/**
 * SentinelGlobe — Deck.gl GlobeView with live case layers.
 * Map style: dark-matter-nolabels (Carto, free, no Mapbox token needed)
 */
import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { _GlobeView as GlobeView } from '@deck.gl/core';
import type { Layer, ViewStateChangeParameters } from '@deck.gl/core';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useMapStore } from '@/store/mapStore';
import { useCasesStore } from '@/store/casesStore';
import { createSpillLayer } from './SpillPolygon';
import { createDriftConeLayer } from './DriftCone';
import { createOriginEllipseLayer } from './OriginEllipse';
import { createVesselTrailLayers } from './VesselTrails';
import { GLOBE_INITIAL_VIEW, MAP_STYLE, SYSTEM_ID, BUILD_VER } from '@/lib/constants';

const GLOBE_VIEW = new GlobeView({ id: 'globe', resolution: 2 });

interface TooltipInfo {
  x: number;
  y: number;
  object?: {
    vessel_name?: string;
    mmsi?: string;
    speed?: number;
    confidence?: number;
    area_km2?: number;
  };
}

export default function SentinelGlobe() {
  const viewState     = useMapStore((s) => s.viewState);
  const setViewState  = useMapStore((s) => s.setViewState);
  const isAutoRotating= useMapStore((s) => s.isAutoRotating);
  const showDrift     = useMapStore((s) => s.showDrift);
  const showVessels   = useMapStore((s) => s.showVesselTrails);
  const showSpills    = useMapStore((s) => s.showSpillPolygons);
  const cases         = useCasesStore((s) => s.cases);
  const fetchCases    = useCasesStore((s) => s.fetchCases);

  const [tooltip, setTooltip] = useState<TooltipInfo | null>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => { fetchCases(); }, [fetchCases]);

  // Auto-rotate
  useEffect(() => {
    if (!isAutoRotating) { cancelAnimationFrame(rafRef.current); return; }
    const rotate = () => {
      setViewState({ bearing: (useMapStore.getState().viewState.bearing + 0.04) % 360 });
      rafRef.current = requestAnimationFrame(rotate);
    };
    rafRef.current = requestAnimationFrame(rotate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isAutoRotating, setViewState]);

  const onHover = useCallback((info: { x: number; y: number; object?: TooltipInfo['object'] }) => {
    setTooltip(info.object ? { x: info.x, y: info.y, object: info.object } : null);
  }, []);

  const onViewStateChange = useCallback(
    (params: ViewStateChangeParameters) =>
      setViewState(params.viewState as Partial<typeof viewState>),
    [setViewState],
  );

  const layers = useMemo(() => {
    const all: Layer[] = [];
    for (const c of cases) {
      if (showSpills && c.spill?.polygon) {
        const l = createSpillLayer(c.spill.polygon, c.spill.confidence, c.id);
        if (l) all.push(l);
      }
      if (showDrift && c.drift_forecast) {
        const l = createDriftConeLayer(c.drift_forecast);
        if (l) all.push(l);
      }
      if (showVessels && c.suspects?.length) {
        all.push(...createVesselTrailLayers(c.suspects.map((s) => s.vessel).filter(Boolean)));
      }
      if (c.spill?.centroid) {
        all.push(...createOriginEllipseLayer(c.spill.centroid, 50, c.spill.confidence));
      }
    }
    return all;
  }, [cases, showDrift, showVessels, showSpills]);

  return (
    <div className="relative h-full w-full">
      <DeckGL
        views={GLOBE_VIEW}
        viewState={viewState}
        onViewStateChange={onViewStateChange}
        layers={layers}
        controller
        onHover={onHover}
        style={{ backgroundColor: '#080a0e' }}
      />

      {/* Tooltip — no glass, just hard border */}
      {tooltip?.object && (
        <div
          className="ops-tooltip"
          style={{ left: tooltip.x + 14, top: tooltip.y - 8 }}
        >
          {tooltip.object.vessel_name && (
            <div className="font-semibold text-sentinel-amber">{tooltip.object.vessel_name}</div>
          )}
          {tooltip.object.mmsi && (
            <div className="text-sentinel-muted">MMSI {tooltip.object.mmsi}</div>
          )}
          {tooltip.object.speed != null && (
            <div>{tooltip.object.speed.toFixed(1)} kn</div>
          )}
          {tooltip.object.confidence != null && (
            <div className="ops-caution">CONF {tooltip.object.confidence}%</div>
          )}
          {tooltip.object.area_km2 != null && (
            <div>{tooltip.object.area_km2.toFixed(2)} km²</div>
          )}
        </div>
      )}

      {/* Bottom-left system watermark */}
      <div className="absolute bottom-3 left-3 font-mono text-2xs text-sentinel-muted pointer-events-none">
        {SYSTEM_ID} / {BUILD_VER}
      </div>
    </div>
  );
}
