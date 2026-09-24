/**
 * SentinelGlobe — High-Performance 3D Satellite & Nautical Globe.
 * Uses Esri World Dark Gray Canvas & Esri World Imagery (Zero Watermark, 100% Free, High Resolution).
 */
import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { _GlobeView as GlobeView } from '@deck.gl/core';
import { TileLayer } from 'deck.gl';
import { BitmapLayer } from 'deck.gl';
import type { Layer, ViewStateChangeParameters } from '@deck.gl/core';
import { Compass, Eye, Layers, Minus, Plus, RotateCcw } from 'lucide-react';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useMapStore } from '@/store/mapStore';
import { useCasesStore } from '@/store/casesStore';
import { createSpillLayer } from './SpillPolygon';
import { createDriftConeLayer } from './DriftCone';
import { createOriginEllipseLayer } from './OriginEllipse';
import { createVesselTrailLayers } from './VesselTrails';
import { SYSTEM_ID, BUILD_VER } from '@/lib/constants';

const GLOBE_VIEW = new GlobeView({ id: 'globe', resolution: 2 });

const TILE_SOURCES = {
  dark: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
  satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
};

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
  const viewState      = useMapStore((s) => s.viewState);
  const setViewState   = useMapStore((s) => s.setViewState);
  const isAutoRotating = useMapStore((s) => s.isAutoRotating);
  const showDrift      = useMapStore((s) => s.showDrift);
  const showVessels    = useMapStore((s) => s.showVesselTrails);
  const showSpills     = useMapStore((s) => s.showSpillPolygons);
  const cases          = useCasesStore((s) => s.cases);
  const fetchCases     = useCasesStore((s) => s.fetchCases);
  const setAutoRotating = useMapStore((s) => s.setAutoRotating);
  const flyTo          = useMapStore((s) => s.flyTo);

  const [mapMode, setMapMode] = useState<'dark' | 'satellite'>('dark');
  const [tooltip, setTooltip] = useState<TooltipInfo | null>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    fetchCases();
  }, [fetchCases]);

  // Smooth auto-rotation
  useEffect(() => {
    if (!isAutoRotating) {
      cancelAnimationFrame(rafRef.current);
      return;
    }
    const rotate = () => {
      setViewState({ bearing: (useMapStore.getState().viewState.bearing + 0.05) % 360 });
      rafRef.current = requestAnimationFrame(rotate);
    };
    rafRef.current = requestAnimationFrame(rotate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isAutoRotating, setViewState]);

  const onHover = useCallback((info: { x: number; y: number; object?: TooltipInfo['object'] }) => {
    setTooltip(info.object ? { x: info.x, y: info.y, object: info.object } : null);
  }, []);

  const onViewStateChange = useCallback(
    (params: ViewStateChangeParameters) => {
      setViewState(params.viewState as Partial<typeof viewState>);
    },
    [setViewState],
  );

  const layers = useMemo(() => {
    // Crisp, watermark-free Esri basemap
    const baseTileLayer = new TileLayer({
      id: `globe-base-${mapMode}`,
      data: TILE_SOURCES[mapMode],
      minZoom: 0,
      maxZoom: 16,
      tileSize: 256,
      renderSubLayers: (props) => {
        const {
          bbox: { west, south, east, north },
        } = props.tile;
        return new BitmapLayer(props, {
          data: null,
          image: props.data,
          bounds: [west, south, east, north],
        });
      },
    });

    const all: Layer[] = [baseTileLayer];

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
  }, [cases, showDrift, showVessels, showSpills, mapMode]);

  return (
    <div
      className="relative h-full w-full overflow-hidden select-none"
      style={{
        background: 'radial-gradient(ellipse at center, #0F172A 0%, #090D16 70%, #04060A 100%)',
      }}
    >
      <DeckGL
        views={GLOBE_VIEW}
        viewState={viewState}
        onViewStateChange={onViewStateChange}
        layers={layers}
        controller
        onHover={onHover}
        style={{ backgroundColor: 'transparent' }}
      />

      {/* Floating Modern Header Badge */}
      <div className="absolute top-4 left-5 z-20 flex items-center gap-3 pointer-events-none">
        <div className="flex items-center gap-2.5 rounded-lg bg-slate-900/90 backdrop-blur-md px-3.5 py-2 border border-slate-800 shadow-xl">
          <span className="h-2.5 w-2.5 rounded-full bg-blue-500 animate-pulse" />
          <span className="text-xs font-semibold text-slate-100 tracking-wide">
            3D Maritime Globe
          </span>
          <span className="text-slate-600">·</span>
          <span className="text-xs text-blue-400 font-medium">
            Indian Ocean Theatre
          </span>
        </div>
      </div>

      {/* Modern Floating Map Controls */}
      <div className="absolute top-4 right-5 z-20 flex flex-col gap-2">
        <div className="flex flex-col rounded-lg bg-slate-900/90 backdrop-blur-md border border-slate-800 shadow-xl overflow-hidden">
          <button
            onClick={() => setViewState({ zoom: Math.min(viewState.zoom + 0.7, 12) })}
            title="Zoom In"
            className="flex h-8 w-8 items-center justify-center hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
          >
            <Plus size={15} />
          </button>
          <div className="h-px bg-slate-800" />
          <button
            onClick={() => setViewState({ zoom: Math.max(viewState.zoom - 0.7, 2) })}
            title="Zoom Out"
            className="flex h-8 w-8 items-center justify-center hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
          >
            <Minus size={15} />
          </button>
        </div>

        <button
          onClick={() => flyTo(73.0, 15.0, 4.5)}
          title="Reset to Indian Ocean"
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-900/90 backdrop-blur-md hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 shadow-xl transition-colors"
        >
          <RotateCcw size={14} />
        </button>

        <button
          onClick={() => setAutoRotating(!isAutoRotating)}
          title={isAutoRotating ? 'Pause 3D Orbit' : 'Resume 3D Orbit'}
          className={`flex h-8 w-8 items-center justify-center rounded-lg backdrop-blur-md border shadow-xl transition-colors ${
            isAutoRotating
              ? 'bg-blue-600/30 border-blue-500 text-blue-400'
              : 'bg-slate-900/90 border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <Compass size={15} className={isAutoRotating ? 'animate-spin' : ''} style={{ animationDuration: '9s' }} />
        </button>

        <button
          onClick={() => setMapMode(mapMode === 'dark' ? 'satellite' : 'dark')}
          title={mapMode === 'dark' ? 'Switch to Satellite View' : 'Switch to Dark Canvas'}
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-900/90 backdrop-blur-md hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 shadow-xl transition-colors"
        >
          <Layers size={14} className={mapMode === 'satellite' ? 'text-blue-400' : ''} />
        </button>
      </div>

      {/* Floating Coordinates Telemetry */}
      <div className="absolute bottom-4 right-5 z-20 pointer-events-none">
        <div className="flex items-center gap-3 rounded-lg bg-slate-900/85 backdrop-blur-md px-3.5 py-1.5 border border-slate-800 shadow-lg font-mono text-xs text-slate-400">
          <span>LAT: <strong className="text-slate-100">{viewState.latitude.toFixed(2)}°</strong></span>
          <span>LON: <strong className="text-slate-100">{viewState.longitude.toFixed(2)}°</strong></span>
          <span>ZOOM: <strong className="text-blue-400">{viewState.zoom.toFixed(1)}</strong></span>
        </div>
      </div>

      {/* Modern Tooltip */}
      {tooltip?.object && (
        <div
          className="fixed z-50 rounded-lg bg-slate-900/95 backdrop-blur-md border border-slate-700 px-3.5 py-2.5 shadow-2xl pointer-events-none font-sans text-xs"
          style={{ left: tooltip.x + 16, top: tooltip.y - 10 }}
        >
          {tooltip.object.vessel_name && (
            <div className="font-semibold text-slate-100 text-sm">{tooltip.object.vessel_name}</div>
          )}
          {tooltip.object.mmsi && (
            <div className="text-slate-400 font-mono text-[11px] mt-0.5">MMSI: {tooltip.object.mmsi}</div>
          )}
          {tooltip.object.speed != null && (
            <div className="text-slate-300 font-mono text-[11px] mt-1">Speed: {tooltip.object.speed.toFixed(1)} knots</div>
          )}
          {tooltip.object.confidence != null && (
            <div className="mt-1 text-emerald-400 font-medium">Confidence: {tooltip.object.confidence}%</div>
          )}
          {tooltip.object.area_km2 != null && (
            <div className="mt-1 text-amber-400 font-medium font-mono">Area: {tooltip.object.area_km2.toFixed(1)} km²</div>
          )}
        </div>
      )}

      {/* System Watermark */}
      <div className="absolute bottom-4 left-5 text-xs text-slate-500 font-mono pointer-events-none">
        {SYSTEM_ID} · {BUILD_VER} · Satellite Stream Online
      </div>
    </div>
  );
}
