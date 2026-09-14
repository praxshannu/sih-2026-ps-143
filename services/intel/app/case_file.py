"""WeasyPrint PDF case file generator for SENTINEL evidence packages.

Generates legal-grade case files containing detection imagery, drift analysis,
suspect rankings, AIS data, chain of custody, and MARPOL citations.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import qrcode
from weasyprint import HTML
from jinja2 import Template
from loguru import logger

CASE_FILE_OUTPUT_DIR = os.getenv("CASE_FILE_OUTPUT_DIR", "/app/data/case_files")

# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

CASE_FILE_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  @page { size: A4; margin: 2cm 2.5cm; }
  @page :first { margin-top: 1.5cm; }
  body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 10pt; color: #1a1a1a; line-height: 1.5; }
  h1 { font-size: 18pt; color: #0d1b2a; margin: 0 0 4pt; }
  h2 { font-size: 13pt; color: #1b263b; border-bottom: 2px solid #415a77; padding-bottom: 4pt; margin-top: 18pt; margin-bottom: 8pt; }
  h3 { font-size: 11pt; color: #415a77; margin-top: 12pt; margin-bottom: 4pt; }
  .header { text-align: center; border-bottom: 3px solid #0d1b2a; padding-bottom: 12pt; margin-bottom: 16pt; }
  .header .org { font-size: 9pt; color: #778da9; text-transform: uppercase; letter-spacing: 2pt; margin-bottom: 4pt; }
  .header .case-num { font-size: 14pt; font-weight: bold; color: #0d1b2a; margin-top: 8pt; }
  .meta-table { width: 100%; border-collapse: collapse; margin-bottom: 12pt; }
  .meta-table td { padding: 4pt 8pt; font-size: 9pt; border: 1px solid #dee2e6; }
  .meta-table td.label { background: #f1f3f5; color: #495057; width: 30%; font-weight: 600; }
  table.data { width: 100%; border-collapse: collapse; margin: 8pt 0 12pt; font-size: 9pt; }
  table.data th { background: #0d1b2a; color: #fff; padding: 5pt 6pt; text-align: left; font-weight: 600; }
  table.data td { padding: 4pt 6pt; border-bottom: 1px solid #dee2e6; }
  table.data tr:nth-child(even) { background: #f8f9fa; }
  .highlight { background: #fff3cd !important; font-weight: bold; }
  .hash-box { background: #f1f3f5; border: 1px solid #dee2e6; padding: 8pt; font-family: monospace; font-size: 8pt; word-break: break-all; margin: 8pt 0; border-radius: 4pt; }
  .priority-critical { color: #8e44ad; font-weight: bold; }
  .priority-high { color: #e74c3c; font-weight: bold; }
  .priority-medium { color: #f39c12; font-weight: bold; }
  .priority-low { color: #2ecc71; font-weight: bold; }
  .gap-list { margin: 4pt 0; padding-left: 16pt; }
  .gap-list li { color: #e74c3c; font-size: 9pt; margin-bottom: 2pt; }
  .qr-container { text-align: center; margin: 12pt 0; }
  .qr-container img { width: 120px; height: 120px; }
  .qr-label { font-size: 8pt; color: #6c757d; margin-top: 4pt; }
  .footer { position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 7pt; color: #adb5bd; border-top: 1px solid #dee2e6; padding-top: 4pt; }
  .page-break { page-break-before: always; }
  .timeline { margin: 8pt 0; }
  .timeline-entry { display: flex; align-items: flex-start; margin-bottom: 6pt; }
  .timeline-dot { width: 10px; height: 10px; border-radius: 50%; background: #415a77; margin-right: 8pt; margin-top: 3pt; flex-shrink: 0; }
  .timeline-content { font-size: 9pt; }
  .timeline-time { font-family: monospace; color: #6c757d; font-size: 8pt; }
  .marpol-box { background: #e8f4f8; border-left: 4px solid #0077b6; padding: 10pt 12pt; margin: 12pt 0; font-size: 9pt; }
  .marpol-box .article { font-weight: bold; color: #0077b6; }
</style>
</head>
<body>

<div class="header">
  <div class="org">National Technical Research Organisation</div>
  <h1>SENTINEL - Maritime Pollution Intelligence Report</h1>
  <div class="case-num">CASE {{ case_id }}</div>
  <div style="font-size: 9pt; color: #6c757d; margin-top: 4pt;">
    Classification: RESTRICTED | Generated: {{ generated_utc }}
  </div>
</div>

<table class="meta-table">
  <tr>
    <td class="label">Evidence Hash (SHA-256)</td>
    <td class="hash-box">{{ evidence_hash }}</td>
  </tr>
  <tr>
    <td class="label">Spill ID</td>
    <td>{{ spill_id }}</td>
  </tr>
  <tr>
    <td class="label">Detection Timestamp (UTC)</td>
    <td>{{ detection_time }}</td>
  </tr>
  <tr>
    <td class="label">Location</td>
    <td>{{ "%.6f"|format(centroid_lat) }}N, {{ "%.6f"|format(centroid_lon) }}E</td>
  </tr>
  <tr>
    <td class="label">Spill Area</td>
    <td>{{ "%.1f"|format(area_m2) }} m&sup2;</td>
  </tr>
  <tr>
    <td class="label">Detection Confidence</td>
    <td>{{ "%.1f"|format(confidence_pct) }}%</td>
  </tr>
  <tr>
    <td class="label">Estimated Spill Age</td>
    <td>{{ "%.1f"|format(age_hours) }} hours</td>
  </tr>
</table>

<h2>1. Incident Narrative</h2>
<div style="background: #f8f9fa; padding: 10pt; border-left: 4px solid #415a77; margin-bottom: 8pt;">
  {{ narrative_text }}
</div>
{% if key_finding %}
<h3>Key Finding</h3>
<p>{{ key_finding }}</p>
{% endif %}
{% if evidentiary_gaps %}
<h3>Evidentiary Gaps</h3>
<ul class="gap-list">
{% for gap in evidentiary_gaps %}
  <li>{{ gap }}</li>
{% endfor %}
</ul>
{% endif %}

<h2>2. SAR Detection Image</h2>
<p style="font-size: 9pt; color: #6c757d;">Spill polygon overlay on Sentinel-1 SAR imagery.</p>
{% if sar_image_path %}
<div style="text-align: center; margin: 8pt 0;">
  <img src="file://{{ sar_image_path }}" style="max-width: 100%; max-height: 400px; border: 1px solid #dee2e6;">
</div>
{% else %}
<div style="background: #f8f9fa; padding: 20pt; text-align: center; color: #6c757d; border: 1px dashed #dee2e6;">
  SAR image not available - provide image_path in detection data
</div>
{% endif %}

<div class="page-break"></div>

<h2>3. Geometric Properties</h2>
<table class="data">
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Spill Area</td><td>{{ "%.1f"|format(area_m2) }} m&sup2;</td></tr>
  <tr><td>Estimated Age</td><td>{{ "%.1f"|format(age_hours) }} hours</td></tr>
  <tr><td>Detection Confidence</td><td>{{ "%.1f"|format(confidence_pct) }}%</td></tr>
  <tr><td>Centroid Lon</td><td>{{ "%.6f"|format(centroid_lon) }}</td></tr>
  <tr><td>Centroid Lat</td><td>{{ "%.6f"|format(centroid_lat) }}</td></tr>
  {% if spill_polygon_wkt %}
  <tr><td>Polygon WKT</td><td style="font-family: monospace; font-size: 8pt; word-break: break-all;">{{ spill_polygon_wkt }}</td></tr>
  {% endif %}
</table>

<h2>4. Drift Analysis - Probable Origin</h2>
{% if drift %}
<table class="data">
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Origin Lon</td><td>{{ "%.6f"|format(drift.origin_lon) }}</td></tr>
  <tr><td>Origin Lat</td><td>{{ "%.6f"|format(drift.origin_lat) }}</td></tr>
  <tr><td>95% CI Semi-Major</td><td>{{ "%.1f"|format(drift.semi_major_km) }} km</td></tr>
  <tr><td>95% CI Semi-Minor</td><td>{{ "%.1f"|format(drift.semi_minor_km) }} km</td></tr>
  <tr><td>Orientation</td><td>{{ "%.1f"|format(drift.orientation_deg) }}&deg;</td></tr>
  <tr><td>Drift Regime</td><td>{{ drift.regime }}</td></tr>
</table>
{% else %}
<p style="color: #6c757d; font-style: italic;">Drift analysis data not provided.</p>
{% endif %}

<h2>5. Suspect Vessel Ranking</h2>
{% if suspects %}
<table class="data">
  <tr><th>#</th><th>Vessel Name</th><th>MMSI</th><th>Composite</th><th>Proximity</th><th>Temporal</th><th>Trajectory</th><th>Anomaly</th><th>Gap (min)</th><th>Dark</th></tr>
{% for s in suspects %}
  <tr>
    <td>{{ s.rank }}</td>
    <td>{{ s.vessel_name }}</td>
    <td style="font-family: monospace;">{{ s.mmsi }}</td>
    <td>{{ "%.3f"|format(s.composite_score) }}</td>
    <td>{{ "%.3f"|format(s.score_proximity) }}</td>
    <td>{{ "%.3f"|format(s.score_temporal) }}</td>
    <td>{{ "%.3f"|format(s.score_trajectory) }}</td>
    <td>{{ "%.3f"|format(s.score_anomaly) }}</td>
    <td>{{ "%.0f"|format(s.ais_gap_minutes) }}</td>
    <td>{{ "YES" if s.is_dark_vessel else "" }}</td>
  </tr>
{% endfor %}
</table>
{% else %}
<p style="color: #6c757d; font-style: italic;">No suspect vessels identified.</p>
{% endif %}

<div class="page-break"></div>

<h2>6. AIS Data Excerpt - Primary Suspect</h2>
{% if ais_excerpt %}
<table class="data">
  <tr><th colspan="5">Vessel: {{ ais_excerpt.vessel_name }} (MMSI: {{ ais_excerpt.mmsi }})</th></tr>
  <tr><th>Timestamp (UTC)</th><th>Lon</th><th>Lat</th><th>SOG (kn)</th><th>COG (&deg;)</th></tr>
{% for pos in ais_excerpt.positions %}
  <tr{% if ais_excerpt.gap_start and ais_excerpt.gap_end and pos.timestamp >= ais_excerpt.gap_start and pos.timestamp <= ais_excerpt.gap_end %} class="highlight"{% endif %}>
    <td>{{ pos.timestamp.strftime('%Y-%m-%d %H:%M') if pos.timestamp is string else pos.timestamp }}</td>
    <td>{{ "%.6f"|format(pos.lon) }}</td>
    <td>{{ "%.6f"|format(pos.lat) }}</td>
    <td>{{ "%.1f"|format(pos.sog) }}</td>
    <td>{{ "%.1f"|format(pos.cog) }}</td>
  </tr>
{% endfor %}
</table>
{% if ais_excerpt.gap_start and ais_excerpt.gap_end %}
<p style="font-size: 9pt; color: #e74c3c; font-weight: bold;">
  AIS gap detected: {{ ais_excerpt.gap_start }} to {{ ais_excerpt.gap_end }}
  (highlighted rows indicate positions during gap period).
</p>
{% endif %}
{% else %}
<p style="color: #6c757d; font-style: italic;">AIS data excerpt not provided.</p>
{% endif %}

<h2>7. Forensic Timeline</h2>
<div class="timeline">
{% for entry in chain_of_custody %}
  <div class="timeline-entry">
    <div class="timeline-dot"></div>
    <div class="timeline-content">
      <strong>{{ entry.stage }}</strong> &mdash; {{ entry.service }}<br>
      <span class="timeline-time">{{ entry.timestamp }}</span><br>
      <span style="font-size: 8pt; color: #6c757d; font-family: monospace;">in: {{ entry.input_hash[:16] }}... | out: {{ entry.output_hash[:16] }}...</span>
    </div>
  </div>
{% endfor %}
</div>

<div class="page-break"></div>

<h2>8. Chain of Custody</h2>
<table class="data">
  <tr><th>Stage</th><th>Service</th><th>Timestamp (UTC)</th><th>Input Hash</th><th>Output Hash</th><th>Chain Hash</th></tr>
{% for entry in chain_of_custody %}
  <tr>
    <td>{{ entry.stage }}</td>
    <td>{{ entry.service }}</td>
    <td>{{ entry.timestamp }}</td>
    <td style="font-family: monospace; font-size: 7pt;">{{ entry.input_hash }}</td>
    <td style="font-family: monospace; font-size: 7pt;">{{ entry.output_hash }}</td>
    <td style="font-family: monospace; font-size: 7pt;">{{ entry.chain_hash }}</td>
  </tr>
{% endfor %}
</table>

{% if marpol_citation %}
<h2>9. Applicable Legal Framework</h2>
<div class="marpol-box">
  <div class="article">MARPOL Annex I, Regulation 4 - Avoidance of Oil Discharge</div>
  <p style="margin: 6pt 0 0;">
    {{ marpol_citation }}
  </p>
</div>
{% endif %}

<h2>10. Dashboard Access</h2>
<div class="qr-container">
  <img src="data:image/png;base64,{{ qr_code_b64 }}" alt="QR Code">
  <div class="qr-label">Scan to access live dashboard for case {{ case_id }}</div>
</div>

<div class="footer">
  SENTINEL Maritime Intelligence System | NTRO | Case {{ case_id }} | Evidence: {{ evidence_hash[:16] }}... | Page generated {{ generated_utc }}
</div>

</body>
</html>
"""
)

MARPOL_CITATION = (
    "Under MARPOL Annex I, Regulation 4, any discharge of oil or oily mixtures "
    "into the sea from ships is prohibited except when all the following conditions "
    "are met: (a) the oil content of the discharge does not exceed 15 parts per "
    "million; (b) the ship is proceeding en route; (c) the discharge is made as "
    "far as practicable, but not less than 50 nautical miles from the nearest "
    "land. The detection of a confirmed oil spill in the absence of permissible "
    "conditions constitutes a violation of this regulation."
)


def _generate_qr_code(data: str) -> str:
    """Generate a QR code and return it as a base64 string."""
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0d1b2a", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_case_file(
    case_id: str,
    detection: dict[str, Any],
    drift: dict[str, Any] | None = None,
    suspects: list[dict[str, Any]] | None = None,
    ais_excerpt: dict[str, Any] | None = None,
    narrative: dict[str, Any] | None = None,
    chain_of_custody: list[dict[str, Any]] | None = None,
    evidence_hash: str = "",
) -> str:
    """Generate a PDF evidence case file.

    Returns the path to the generated PDF.
    """
    output_dir = Path(CASE_FILE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{case_id}.pdf"

    # Build template context
    narrative_text = ""
    key_finding = ""
    evidentiary_gaps: list[str] = []
    alert_priority = "MEDIUM"
    if narrative:
        narrative_text = narrative.get("summary", "")
        key_finding = narrative.get("key_finding", "")
        evidentiary_gaps = narrative.get("evidentiary_gaps", [])
        alert_priority = narrative.get("alert_priority", "MEDIUM")

    # Check if MARPOL applies (discharge confirmed)
    marpol_citation = ""
    if alert_priority in ("HIGH", "CRITICAL"):
        marpol_citation = MARPOL_CITATION

    # QR code for live dashboard
    dashboard_url = f"https://dashboard.sentinel.gov.in/case/{case_id}"
    qr_b64 = _generate_qr_code(dashboard_url)

    # Prepare suspect dicts for template
    suspect_dicts = []
    for s in (suspects or []):
        if hasattr(s, "model_dump"):
            suspect_dicts.append(s.model_dump())
        elif isinstance(s, dict):
            suspect_dicts.append(s)
        else:
            suspect_dicts.append(s.__dict__)

    # Prepare AIS excerpt
    ais_dict = None
    if ais_excerpt:
        if hasattr(ais_excerpt, "model_dump"):
            ais_dict = ais_excerpt.model_dump()
        elif isinstance(ais_excerpt, dict):
            ais_dict = ais_excerpt
        else:
            ais_dict = ais_excerpt.__dict__

    # Prepare chain of custody
    custody_dicts = []
    for c in (chain_of_custody or []):
        if hasattr(c, "model_dump"):
            custody_dicts.append(c.model_dump())
        elif isinstance(c, dict):
            custody_dicts.append(c)
        else:
            custody_dicts.append(c.__dict__)

    # SAR image path
    sar_image_path = detection.get("image_path", "")

    ctx = {
        "case_id": case_id,
        "spill_id": detection.get("spill_id", ""),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "evidence_hash": evidence_hash,
        "centroid_lat": detection.get("centroid_lat", 0),
        "centroid_lon": detection.get("centroid_lon", 0),
        "area_m2": detection.get("area_m2", 0),
        "confidence_pct": detection.get("confidence", 0) * 100,
        "age_hours": detection.get("age_hours", 0),
        "detection_time": str(detection.get("detected_at", "")),
        "spill_polygon_wkt": detection.get("spill_polygon_wkt"),
        "narrative_text": narrative_text,
        "key_finding": key_finding,
        "evidentiary_gaps": evidentiary_gaps,
        "sar_image_path": sar_image_path,
        "drift": drift,
        "suspects": suspect_dicts,
        "ais_excerpt": ais_dict,
        "chain_of_custody": custody_dicts,
        "marpol_citation": marpol_citation,
        "qr_code_b64": qr_b64,
    }

    html_content = CASE_FILE_TEMPLATE.render(**ctx)

    logger.info("Generating PDF case file for case {}", case_id)
    try:
        HTML(string=html_content).write_pdf(str(pdf_path))
    except Exception as e:
        # Never let one bad image kill a legal package: retry text-only.
        logger.warning("PDF with images failed ({}), retrying text-only", e)
        import re as _re

        text_only = _re.sub(r"<img\b[^>]*>", "", html_content)
        HTML(string=text_only).write_pdf(str(pdf_path))

    # Count pages (approximate by counting page-breaks + 1)
    page_count = html_content.count("page-break") + 1

    logger.info("Case file generated: {} ({} pages)", pdf_path, page_count)
    return str(pdf_path)
