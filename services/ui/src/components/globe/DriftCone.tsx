import { PolygonLayer } from '@deck.gl/layers';
import type { GeoPoint, DriftForecast } from '@/lib/api';
import { COLORS } from '@/lib/constants';

interface DriftConeProps {
  forecast: DriftForecast;
}

export function createDriftConeLayer(forecast: DriftConeProps['forecast'] | null) {
  if (!forecast || forecast.polygon.length < 3) return null;

  const coords = [...forecast.polygon, forecast.polygon[0]].map((p) => [
    p.longitude,
    p.latitude,
  ]);

  const quantiles = forecast.probability_quantiles;
  const innerOpacity = quantiles[2] ?? 0.5;
  const outerOpacity = quantiles[0] ?? 0.15;

  return new PolygonLayer({
    id: `drift-cone-${forecast.hours}h`,
    data: [
      {
        polygon: coords,
        innerOpacity,
        outerOpacity,
      },
    ],
    getPolygon: (d: { polygon: number[][] }) => d.polygon,
    getFillColor: (d: { innerOpacity: number }) => {
      const r = parseInt(COLORS.primary.slice(1, 3), 16);
      const g = parseInt(COLORS.primary.slice(3, 5), 16);
      const b = parseInt(COLORS.primary.slice(5, 7), 16);
      return [r, g, b, Math.round(d.innerOpacity * 255)];
    },
    getLineColor: (d: { outerOpacity: number }) => {
      const r = parseInt(COLORS.primary.slice(1, 3), 16);
      const g = parseInt(COLORS.primary.slice(3, 5), 16);
      const b = parseInt(COLORS.primary.slice(5, 7), 16);
      return [r, g, b, Math.round(d.outerOpacity * 255)];
    },
    lineWidthMinPixels: 1,
    lineWidthMaxPixels: 2,
    pickable: true,
    parameters: { depthTest: false },
  });
}

export default function DriftCone(_props: DriftConeProps) {
  return null;
}
