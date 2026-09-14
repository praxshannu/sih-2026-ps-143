import { PathLayer, ScatterplotLayer } from '@deck.gl/layers';
import type { AisTrack } from '@/lib/api';
import { LAYER_COLORS } from '@/lib/constants';

// Risk-tier colours: red for high-risk (tanker), amber for cargo, green for low
function vesselColor(type: string, selected: boolean): [number, number, number, number] {
  if (selected) return [243, 244, 246, 255];
  switch (type?.toUpperCase()) {
    case 'TANKER':  return LAYER_COLORS.vesselHigh;
    case 'CARGO':   return LAYER_COLORS.vesselMed;
    default:        return LAYER_COLORS.vesselLow;
  }
}

export function createVesselTrailLayers(
  tracks: AisTrack[],
  selectedMmsi?: string | null,
) {
  return tracks.flatMap((track) => {
    const sel = track.mmsi === selectedMmsi;
    const color = vesselColor(track.vessel_type, sel);

    // Build trail as a single path. Typed as a tuple, not `number[][]` —
    // deck.gl's PathGeometry wants fixed-length positions.
    const path: [number, number][] = track.waypoints.map((wp) => [
      wp.longitude,
      wp.latitude,
    ]);
    const trailLayer = new PathLayer({
      id: `trail-${track.mmsi}`,
      data: [{ path, vessel_name: track.vessel_name, mmsi: track.mmsi }],
      getPath: (d: { path: [number, number][] }) => d.path,
      getColor: color,
      getWidth: sel ? 2.5 : 1.5,
      widthMinPixels: sel ? 2 : 1,
      widthMaxPixels: sel ? 4 : 2,
      pickable: true,
      parameters: { depthTest: false },
    });

    const lastWp = track.waypoints.at(-1);
    if (!lastWp) return [trailLayer];

    const headLayer = new ScatterplotLayer({
      id: `head-${track.mmsi}`,
      data: [{
        position: [lastWp.longitude, lastWp.latitude] as [number, number],
        vessel_name: track.vessel_name,
        mmsi: track.mmsi,
        speed: lastWp.speed,
      }],
      getPosition: (d: { position: [number, number] }) => d.position,
      getRadius: sel ? 55000 : 35000,
      getFillColor: color,
      getLineColor: [243, 244, 246, 180],
      lineWidthMinPixels: 1,
      stroked: true,
      radiusMinPixels: sel ? 5 : 3,
      radiusMaxPixels: sel ? 9 : 5,
      pickable: true,
      parameters: { depthTest: false },
    });

    return [trailLayer, headLayer];
  });
}

// Component form (unused in globe, but keeps the export for any direct use)
export default function VesselTrails({ tracks, selectedMmsi }: { tracks: AisTrack[]; selectedMmsi?: string | null }) {
  void createVesselTrailLayers(tracks, selectedMmsi);
  return null;
}
