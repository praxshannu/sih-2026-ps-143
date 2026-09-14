import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          maps: ['deck.gl', '@deck.gl/core', '@deck.gl/layers', '@deck.gl/react', '@deck.gl/maplibre', 'maplibre-gl', 'react-map-gl'],
          charts: ['recharts', 'd3', 'd3-scale'],
          ui: ['framer-motion', 'lucide-react'],
        },
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    // Dev only: allow docker hostnames (sentinel-ui) + colima forwards.
    allowedHosts: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://sentinel-api:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: process.env.VITE_WS_URL || 'ws://sentinel-api:8000',
        ws: true,
      },
    },
  },
  // `vite preview` serves the built bundle. It needs the same /api proxy as
  // dev, otherwise the production build silently loses its backend and every
  // panel renders empty — which looks like a data bug, not a config one.
  preview: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://sentinel-api:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: process.env.VITE_WS_URL || 'ws://sentinel-api:8000',
        ws: true,
      },
    },
  },
}));
