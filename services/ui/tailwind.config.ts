import type { Config } from 'tailwindcss';

/**
 * SENTINEL design tokens — Light Professional Theme.
 *
 * The direction is a **maritime intelligence platform**, not a SaaS dashboard.
 * Key decisions:
 *
 * - Background: #F8FAFC (cool off-white). Clean, neutral, never warm cream.
 * - Primary accent: #4096FF. Used for interactive elements, live data values,
 *   and active navigation. ONE accent colour, applied consistently.
 * - Oil/amber stays amber. It is a domain-semantic colour: oil slick colour
 *   on SAR imagery. An analyst cannot afford oil and UI sharing a colour.
 * - Nav rail is dark slate (#1E293B). Dark sidebar + light content area is
 *   the gold standard for professional tools (VS Code, Linear, Vercel).
 * - Globe and map tiles remain on dark-matter substrate. Geospatial data
 *   layers (orange oil, red vessels, green origin) are tuned for dark water.
 *
 * Type families unchanged:
 * - `display` (Space Grotesk) — headings and labels
 * - `mono` (IBM Plex Mono) — every number, ID and coordinate
 * - `serif` (Newsreader) — analyst narrative only
 */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        sentinel: {
          // ── Deep Maritime Command Surface Hierarchy ─────────────────────────
          bg:         '#060A11',  // abyssal deep ocean base
          surface:    '#0B111D',  // tactical console surface
          panel:      '#0F1726',  // inset telemetry panel
          raised:     '#152033',  // elevated cards / focus state
          border:     '#1A273D',  // hairline instrument border
          'border-hi':'#2A3E5E',  // active / prominent divider

          // ── Primary Tactical Accent: Electric Cyan ─────────────────────────
          data:       '#00D2FF',  // HUD telemetry cyan
          'data-dim': '#083344',  // muted cyan tint
          'data-hi':  '#38BDF8',  // hover state
          primary:    '#00D2FF',
          'primary-hover': '#38BDF8',
          link:       '#38BDF8',

          // ── Signal Radar Amber: Oil & SAR Slicks ────────────────────────────
          oil:        '#F59E0B',
          'oil-dim':  '#451A03',
          'oil-hi':   '#FBBF24',
          amber:      '#F59E0B',
          'amber-dim':'#451A03',
          'amber-glow':'#FBBF24',

          // ── Tactical Operational Status ────────────────────────────────────
          danger:      '#EF4444',
          'danger-dim':'#450A0A',
          caution:     '#F59E0B',
          nominal:     '#10B981',
          'nominal-dim':'#064E3B',

          // ── High-Contrast Monospace / Display Text ─────────────────────────
          text:       '#CBD5E1',  // slate-200 legible running text
          'text-hi':  '#F8FAFC',  // crisp white numbers, coordinates, titles
          muted:      '#64748B',  // slate-500 secondary metrics
          'muted-hi': '#94A3B8',  // slate-400 uppercase field headers

          // ── Compatibility aliases ──────────────────────────────────────────
          warning:    '#F59E0B',
          success:    '#10B981',
        },

        // ── Navigation Rail ──────────────────────────────────────────────────
        nav: {
          bg:       '#05080E',
          surface:  '#0B111D',
          hover:    '#111A29',
          active:   '#0C1E33',
          border:   '#162338',
          text:     '#64748B',
          'text-hi':'#F8FAFC',
          accent:   '#00D2FF',
          muted:    '#475569',
        },
      },
      fontFamily: {
        display: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"IBM Plex Mono"', 'ui-monospace', 'monospace'],
        serif: ['Newsreader', 'Georgia', 'serif'],
      },
      fontSize: {
        // Instrument scale — gaps are intentionally uneven.
        '3xs': ['0.5625rem', { lineHeight: '0.875rem' }],  // 9px  — micro labels
        '2xs': ['0.625rem',  { lineHeight: '0.9375rem' }], // 10px — instrument labels
        xs:   ['0.6875rem', { lineHeight: '1.0625rem' }],  // 11px — table body
        sm:   ['0.75rem',   { lineHeight: '1.125rem' }],   // 12px — body
        base: ['0.8125rem', { lineHeight: '1.25rem' }],    // 13px — comfortable body
        lg:   ['0.9375rem', { lineHeight: '1.375rem' }],   // 15px — section heads
        xl:   ['1.125rem',  { lineHeight: '1.5rem' }],     // 18px — page titles
        lede: ['1.0625rem', { lineHeight: '1.75rem' }],    // 17px — serif narrative
      },
      letterSpacing: {
        instrument: '0.14em',
        stencil: '0.22em',
      },
      borderRadius: {
        none: '0',
        hair: '1px',
        instrument: '2px',
        sm: '4px',
        DEFAULT: '6px',
        md: '8px',
      },
      spacing: {
        1.75: '0.4375rem',
        3.5:  '0.875rem',
        4.5:  '1.125rem',
        row:  '1.75rem',
        rail: '13.5rem',
        gauge: '3.5rem',
      },
      boxShadow: {
        // Panels: subtle lift — not glassmorphism, just material.
        card: '0 1px 3px 0 rgba(0,0,0,0.06), 0 1px 2px -1px rgba(0,0,0,0.04)',
        'card-md': '0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.04)',
        'card-hover': '0 4px 12px 0 rgba(64,150,255,0.10), 0 1px 3px 0 rgba(0,0,0,0.06)',
        // Input focus ring
        focus: '0 0 0 3px rgba(64,150,255,0.18)',
      },
      transitionTimingFunction: {
        instrument: 'cubic-bezier(0.22, 0.61, 0.36, 1)',
      },
      transitionDuration: {
        600: '600ms',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        heartbeat: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.35' },
        },
        'badge-pulse': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        blink: 'blink 1.2s step-end infinite',
        heartbeat: 'heartbeat 2.4s ease-in-out infinite',
        'badge-pulse': 'badge-pulse 3.6s ease-in-out infinite',
        'fade-up': 'fade-up 0.25s ease-out',
      },
    },
  },
  plugins: [],
} satisfies Config;
