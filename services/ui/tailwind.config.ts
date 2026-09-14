import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        sentinel: {
          // Near-black base — not navy, not midnight blue
          bg: '#080a0e',
          surface: '#0e1117',
          panel: '#121820',
          border: '#1e2530',
          'border-hi': '#2a3545',
          // Amber as the operational accent — radar/FLIR aesthetic
          amber: '#d97706',
          'amber-dim': '#92400e',
          'amber-glow': '#fbbf24',
          // Status
          danger: '#dc2626',
          'danger-dim': '#7f1d1d',
          caution: '#ca8a04',
          nominal: '#15803d',
          'nominal-dim': '#14532d',
          // Text
          text: '#d1d5db',
          'text-hi': '#f3f4f6',
          muted: '#4b5563',
          'muted-hi': '#6b7280',
          // Interactive blue — only for links and selected states
          link: '#3b82f6',
        },
      },
      fontFamily: {
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
        sans: ['"IBM Plex Sans"', 'ui-sans-serif', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '1rem' }],
      },
      borderWidth: {
        DEFAULT: '1px',
      },
      animation: {
        // Removed glow/pulse rings — use static indicators
        blink: 'blink 1.2s step-end infinite',
        scanline: 'scanline 6s linear infinite',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
