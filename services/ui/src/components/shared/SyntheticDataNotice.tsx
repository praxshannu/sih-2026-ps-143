/**
 * SyntheticDataNotice — the mandatory warning that accompanies simulated AIS.
 *
 * RATIONALE
 * ---------
 * There is no free AIS source covering the open Indian Ocean (AISStream is a
 * terrestrial network and returned zero messages over an Indian Ocean bbox in
 * a 40-second live test). Rather than show nothing — or worse, pass simulated
 * traffic off as real — SENTINEL generates labelled synthetic tracks and puts
 * this notice in the operator's face.
 *
 * DESIGN RULES (matches the instrument-panel language, not a marketing site):
 *   - Hard 1px borders, zero radius, no blur, no glow.
 *   - Diagonal hazard hatching on the leading edge, drawn with CSS gradients.
 *   - Amber = caution, NOT red. The data is usable for demonstration; it is
 *     not an error. Red would cry wolf.
 *   - `variant="blocking"` adds a dismiss gate with an explicit "I understand"
 *     checkbox so the acknowledgement is deliberate, not a reflex click.
 */

import { useId, useState } from 'react';

export type SyntheticVariant = 'inline' | 'banner' | 'blocking';

interface SyntheticDataNoticeProps {
  /** Short provenance reason — e.g. why no real coverage exists here. */
  reason?: string | null;
  /** Number of synthetic vessels on screen, if known. */
  count?: number;
  /** ISO window the synthetic tracks cover. */
  window?: [string, string] | null;
  variant?: SyntheticVariant;
  /** For `blocking`: called once the operator ticks "I understand". */
  onAcknowledge?: () => void;
  /** Dismiss handler for `banner`. Omit to make it non-dismissible. */
  onDismiss?: () => void;
  className?: string;
}

const HATCH =
  'repeating-linear-gradient(45deg, #d97706 0 4px, #121820 4px 8px)';

function formatStamp(iso?: string | null) {
  if (!iso) return '—';
  return iso.replace('T', ' ').replace(':00Z', 'Z');
}

export default function SyntheticDataNotice({
  reason,
  count,
  window: timeWindow,
  variant = 'banner',
  onAcknowledge,
  onDismiss,
  className = '',
}: SyntheticDataNoticeProps) {
  const [ack, setAck] = useState(false);
  const cbId = useId();

  const shell = 'border-l-2 border-sentinel-amber bg-sentinel-surface';

  const header = (
    <div className="flex items-start gap-2">
      {/* Hazard hatch block — reads as a physical warning label */}
      <span
        aria-hidden
        className="mt-0.5 h-3 w-3 flex-shrink-0"
        style={{ background: HATCH }}
      />
      <div className="min-w-0 flex-1">
        <p className="font-mono text-2xs font-semibold uppercase tracking-[0.15em] text-sentinel-amber">
          Simulated vessel data — not a live feed
        </p>
        <p className="mt-1 font-mono text-2xs leading-relaxed text-sentinel-text">
          No AIS receiver network covers this area. Every vessel shown is
          synthetic and must not be used as evidence in any investigation or
          enforcement action.
        </p>
      </div>
    </div>
  );

  const facts =
    reason || count !== undefined || timeWindow ? (
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-2xs">
        {count !== undefined && (
          <>
            <dt className="uppercase tracking-wider text-sentinel-muted">
              Tracks
            </dt>
            <dd className="tabular-nums text-sentinel-text-hi">
              {count} synthetic
            </dd>
          </>
        )}
        {timeWindow && (
          <>
            <dt className="uppercase tracking-wider text-sentinel-muted">
              Window
            </dt>
            <dd className="tabular-nums text-sentinel-text-hi">
              {formatStamp(timeWindow[0])} → {formatStamp(timeWindow[1])}
            </dd>
          </>
        )}
        {reason && (
          <>
            <dt className="uppercase tracking-wider text-sentinel-muted">
              Basis
            </dt>
            <dd className="text-sentinel-text">{reason}</dd>
          </>
        )}
      </dl>
    ) : null;

  if (variant === 'inline') {
    return (
      <div
        role="status"
        className={`${shell} px-3 py-2 ${className}`}
        aria-label="Synthetic data warning"
      >
        {header}
        {facts}
      </div>
    );
  }

  if (variant === 'blocking') {
    return (
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={`${cbId}-title`}
        className={`${shell} border-y border-sentinel-amber px-4 py-3 ${className}`}
      >
        <p id={`${cbId}-title`} className="sr-only">
          Synthetic AIS acknowledgement required
        </p>
        {header}
        {facts}
        <label
          htmlFor={cbId}
          className="mt-3 flex cursor-pointer items-start gap-2 font-mono text-2xs text-sentinel-text-hi"
        >
          <input
            id={cbId}
            type="checkbox"
            checked={ack}
            onChange={(e) => setAck(e.target.checked)}
            className="mt-0.5 h-3 w-3 flex-shrink-0 accent-[#d97706]"
          />
          <span>
            I understand these tracks are simulated and carry no evidential
            weight.
          </span>
        </label>
        <button
          type="button"
          disabled={!ack}
          onClick={onAcknowledge}
          className="mt-2 border border-sentinel-amber bg-transparent px-3 py-1 font-mono text-2xs font-semibold uppercase tracking-[0.15em] text-sentinel-amber transition-colors hover:bg-sentinel-amber hover:text-sentinel-bg disabled:cursor-not-allowed disabled:border-sentinel-border disabled:bg-transparent disabled:text-sentinel-muted"
        >
          Proceed with synthetic data
        </button>
      </div>
    );
  }

  // variant === 'banner'
  return (
    <div
      role="status"
      className={`${shell} flex items-start gap-3 border-y border-sentinel-amber px-4 py-2.5 ${className}`}
      aria-label="Synthetic data warning"
    >
      <div className="min-w-0 flex-1">
        {header}
        {facts}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss synthetic data warning"
          className="flex-shrink-0 border border-sentinel-border px-2 py-0.5 font-mono text-2xs uppercase tracking-wider text-sentinel-muted transition-colors hover:border-sentinel-amber hover:text-sentinel-amber"
        >
          ✕
        </button>
      )}
    </div>
  );
}
