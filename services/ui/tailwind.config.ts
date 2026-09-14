import type { Config } from 'tailwindcss';

/**
 * SENTINEL design tokens.
 *
 * The direction is an **oceanographic instrument**, not a dashboard. That
 * distinction drives every value here:
 *
 * - Colour is semantic, not decorative. `data` (cyan) means "this is a
 *   measurement". `oil` (amber) is reserved *exclusively* for oil — the moment
 *   amber also means "hover" or "accent", an analyst loses the one colour cue
 *   that carries meaning on a dark SAR image. Nothing else is coloured.
 * - Separation comes from 1px rules and negative space, never from shadows or
 *   blur. `shadow` and `backdrop-blur` are deliberately absent: glassmorphism
 *   reads as decoration, and this is equipment.
 * - Radii stop at 2px. A card with a 12px radius is the single clearest tell
 *   of a template.
 *
 * Type does three distinct jobs, so there are three families:
 * - `display` (Space Grotesk) — headings and nav. Slightly condensed, a little
 *   mechanical, reads as labelling rather than prose.
 * - `mono` (IBM Plex Mono) — every number, coordinate, MMSI and timestamp.
 *   Tabular figures, because a column of coordinates that jitters is unusable.
 * - `serif` (Newsreader) — the intelligence narrative only. A serif paragraph
 *   is the fastest way to stop a screen reading as generated, and it is
 *   genuinely the right voice for an analyst's written assessment.
 */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        sentinel: {
          // Near-black base. Not navy, not midnight — this sits under a map
          // and must not compete with it.
          bg: '#070b10',
          surface: '#0c1116',
          panel: '#111821',
          raised: '#151d27',
          border: '#1b2430',
          'border-hi': '#2a3644',

          // Chart cyan: measurements, tracks, geometry, live data.
          data: '#5ec8d8',
          'data-dim': '#2b6d78',
          'data-hi': '#8fe0ec',

          // Signal amber: oil. Nothing else. See note above.
          oil: '#e8a33d',
          'oil-dim': '#8a5f1c',
          'oil-hi': '#f6c477',
          // Legacy alias kept so the older pages keep rendering while they are
          // migrated to `oil`; do not use in new code.
          amber: '#d97706',
          'amber-dim': '#92400e',
          'amber-glow': '#fbbf24',

          // Status. Deliberately desaturated — status should read as a glyph
          // and a letter code first, colour second (colour-blind safe).
          danger: '#c8453a',
          'danger-dim': '#6d241d',
          caution: '#c08a2e',
          nominal: '#4f9d69',
          'nominal-dim': '#24513a',

          // Text ramp.
          text: '#c9d1d9',
          'text-hi': '#eef2f6',
          muted: '#5b6673',
          'muted-hi': '#7d8794',

          link: '#5ec8d8',
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
        sans: ['"IBM Plex Sans"', 'ui-sans-serif', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
        serif: ['Newsreader', 'Georgia', 'serif'],
      },
      fontSize: {
        // A real scale, not five arbitrary sizes. The gaps are uneven on
        // purpose: 2xs/3xs carry dense instrument labels, then there is a
        // jump to `lede` for the narrative voice.
        '3xs': ['0.5625rem', { lineHeight: '0.875rem' }], // 9px — micro labels
        '2xs': ['0.625rem', { lineHeight: '0.9375rem' }], // 10px — instrument labels
        xs: ['0.6875rem', { lineHeight: '1.0625rem' }], // 11px — table body
        sm: ['0.75rem', { lineHeight: '1.125rem' }], // 12px — body
        base: ['0.8125rem', { lineHeight: '1.25rem' }], // 13px — comfortable body
        lg: ['0.9375rem', { lineHeight: '1.375rem' }], // 15px — section heads
        xl: ['1.125rem', { lineHeight: '1.5rem' }], // 18px — page titles
        lede: ['1.0625rem', { lineHeight: '1.75rem' }], // 17px — serif narrative
      },
      letterSpacing: {
        instrument: '0.14em',
        stencil: '0.22em',
      },
      borderRadius: {
        none: '0',
        hair: '1px',
        instrument: '2px',
      },
      spacing: {
        // Optical rhythm: instrument rows are 28px, panel padding is uneven
        // (14px vertical, 16px horizontal) because equal padding reads as
        // machine-generated on a dense panel.
        1.75: '0.4375rem',
        3.5: '0.875rem',
        4.5: '1.125rem',
        row: '1.75rem',
        rail: '13.5rem',
        gauge: '3.5rem',
      },
      transitionTimingFunction: {
        // Slow and expensive-feeling. No bounce, ever.
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
        // One deliberate motion: a live-feed heartbeat. Nothing else pulses.
        heartbeat: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.35' },
        },
      },
      animation: {
        blink: 'blink 1.2s step-end infinite',
        heartbeat: 'heartbeat 2.4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
} satisfies Config;
