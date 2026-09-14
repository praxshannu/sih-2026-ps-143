import { create } from 'zustand';
import { GLOBE_INITIAL_VIEW } from '@/lib/constants';

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch: number;
  bearing: number;
}

interface MapState {
  viewState: ViewState;
  selectedSpillId: string | null;
  selectedMmsi: string | null;
  showDrift: boolean;
  showVesselTrails: boolean;
  showSpillPolygons: boolean;
  isAutoRotating: boolean;
  setViewState: (vs: Partial<ViewState>) => void;
  selectSpill: (id: string | null) => void;
  selectVessel: (mmsi: string | null) => void;
  toggleDrift: () => void;
  toggleVesselTrails: () => void;
  toggleSpillPolygons: () => void;
  setAutoRotating: (v: boolean) => void;
  flyTo: (lon: number, lat: number, zoom?: number) => void;
}

export const useMapStore = create<MapState>((set) => ({
  viewState: { ...GLOBE_INITIAL_VIEW },
  selectedSpillId: null,
  selectedMmsi: null,
  showDrift: true,
  showVesselTrails: true,
  showSpillPolygons: true,
  isAutoRotating: true,

  setViewState: (vs) =>
    set((s) => ({ viewState: { ...s.viewState, ...vs }, isAutoRotating: false })),

  selectSpill: (id) => set({ selectedSpillId: id }),

  selectVessel: (mmsi) => set({ selectedMmsi: mmsi }),

  toggleDrift: () => set((s) => ({ showDrift: !s.showDrift })),

  toggleVesselTrails: () => set((s) => ({ showVesselTrails: !s.showVesselTrails })),

  toggleSpillPolygons: () => set((s) => ({ showSpillPolygons: !s.showSpillPolygons })),

  setAutoRotating: (v) => set({ isAutoRotating: v }),

  flyTo: (lon, lat, zoom = 8) =>
    set({ viewState: { longitude: lon, latitude: lat, zoom, pitch: 45, bearing: -15 }, isAutoRotating: false }),
}));
