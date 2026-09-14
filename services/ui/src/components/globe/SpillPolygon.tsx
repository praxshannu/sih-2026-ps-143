import { PolygonLayer } from '@deck.gl/layers';
import type { GeoPoint } from '@/lib/api';
import { LAYER_COLORS } from '@/lib/constants';

export function createSpillLayer(
  polygon: GeoPoint[],
  confidence: number,
  id: string,
): PolygonLayer | null {
  if (polygon.length < 3) return null;
  const coords = [...polygon, polygon[0]].map((p) => [p.longitude, p.latitude]);
  // Opacity scales with confidence but stays readable
  const alpha = Math.round((0.35 + (confidence / 100) * 0.5) * 255);
  const fill: [number, number, number, number] = [...LAYER_COLORS.spillFill.slice(0, 3), alpha] as [number, number, number, number];
  return new PolygonLayer({
    id: `spill-${id}`,
    data: [{ polygon: coords }],
    getPolygon: (d: { polygon: number[][] }) => d.polygon,
    getFillColor: fill,
    getLineColor: LAYER_COLORS.spillLine,
    lineWidthMinPixels: 1.5,
    lineWidthMaxPixels: 3,
    pickable: true,
    parameters: { depthTest: false },
  });
}
