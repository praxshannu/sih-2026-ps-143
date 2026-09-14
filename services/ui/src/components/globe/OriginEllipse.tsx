import { ScatterplotLayer } from '@deck.gl/layers';
import type { GeoPoint } from '@/lib/api';
import { COLORS } from '@/lib/constants';

interface OriginEllipseProps {
  center: GeoPoint;
  radiusKm?: number;
  confidence?: number;
}

export function createOriginEllipseLayer(
  center: GeoPoint,
  radiusKm = 50,
  confidence = 95,
) {
  const r = parseInt(COLORS.danger.slice(1, 3), 16);
  const g = parseInt(COLORS.danger.slice(3, 5), 16);
  const b = parseInt(COLORS.danger.slice(5, 7), 16);

  const outerPoints = generateEllipsePoints(center, radiusKm, 64);
  const innerPoints = generateEllipsePoints(center, radiusKm * 0.3, 32);

  const layers = [
    new ScatterplotLayer({
      id: `origin-ellipse-outer-${center.longitude}-${center.latitude}`,
      data: outerPoints.map((p) => ({ position: [p.longitude, p.latitude] })),
      getPosition: (d: { position: number[] }) => d.position as [number, number],
      getRadius: 40000,
      getFillColor: [r, g, b, 0],
      getLineColor: [r, g, b, Math.round((confidence / 100) * 200)],
      lineWidthMinPixels: 2,
      lineWidthMaxPixels: 3,
      pickable: false,
      stroked: true,
      filled: false,
      radiusUnits: 'meters' as const,
      parameters: { depthTest: false },
    }),
    new ScatterplotLayer({
      id: `origin-crosshair-h-${center.longitude}-${center.latitude}`,
      data: [
        {
          from: [center.longitude - radiusKm / 111, center.latitude],
          to: [center.longitude + radiusKm / 111, center.latitude],
        },
      ],
      getRadius: 15000,
      getPosition: (d: { from: number[] }) => d.from as [number, number],
      getFillColor: [r, g, b, Math.round((confidence / 100) * 180)],
      getLineColor: [r, g, b, Math.round((confidence / 100) * 180)],
      radiusMinPixels: 1,
      radiusMaxPixels: 2,
      pickable: false,
      parameters: { depthTest: false },
    }),
    new ScatterplotLayer({
      id: `origin-crosshair-v-${center.longitude}-${center.latitude}`,
      data: [
        {
          position: [center.longitude, center.latitude - radiusKm / 111],
        },
        {
          position: [center.longitude, center.latitude + radiusKm / 111],
        },
      ],
      getPosition: (d: { position: number[] }) => d.position as [number, number],
      getRadius: 15000,
      getFillColor: [r, g, b, Math.round((confidence / 100) * 180)],
      getLineColor: [r, g, b, Math.round((confidence / 100) * 180)],
      radiusMinPixels: 1,
      radiusMaxPixels: 2,
      pickable: false,
      parameters: { depthTest: false },
    }),
  ];

  return layers;
}

function generateEllipsePoints(
  center: GeoPoint,
  radiusKm: number,
  segments: number,
): GeoPoint[] {
  const points: GeoPoint[] = [];
  const latRad = (center.latitude * Math.PI) / 180;
  const degLat = radiusKm / 111.32;
  const degLon = radiusKm / (111.32 * Math.cos(latRad));

  for (let i = 0; i < segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    points.push({
      longitude: center.longitude + degLon * Math.cos(angle),
      latitude: center.latitude + degLat * Math.sin(angle),
    });
  }
  return points;
}

export default function OriginEllipse(_props: OriginEllipseProps) {
  return null;
}
