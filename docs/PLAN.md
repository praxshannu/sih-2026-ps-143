# SENTINEL — Execution Plan

Companion to [`VERIFICATION.md`](./VERIFICATION.md). Every substitution below is
driven by something that was observed failing on this machine, not by preference.

---

## 0. The one decision that matters

The prompt specifies a **2 GB-per-scene, SNAP-based, CUDA-accelerated, 11-container**
pipeline. On an M2 with 16 GB and 39 GB free that pipeline cannot start.

The same capability is reachable by moving SAR preprocessing **server-side** into
Copernicus. This is not a downgrade — it is what the Copernicus Browser itself does,
and it turns a 4-minute 2 GB download into a 20-second 44 MB calibrated GeoTIFF:

```
spec:    CDSE OData download → 2 GB .SAFE → SNAP (Java, 8 GB RAM) → GeoTIFF
actual:  CDSE OData metadata + sh.dataspace Process API → 44 MB GeoTIFF  ✅ proven
```

Everything else in the spec survives. Only the toolchain changes.

---

## 1. Runtime topology

Keep all seven service boundaries (they already exist and match `AGENTS.md`), but
**run them as native processes, not containers.** Seven uvicorn workers cost
~450 MB total; the *containers* are what would have exhausted 16 GB once
Postgres, Redis, Celery, Prometheus, Grafana and Flower are added.

```make
make dev     # 7 × uvicorn (native) + Postgres + Redis (docker) + Vite
make prod    # full docker-compose, for deployment off this laptop
```

Ports unchanged: ingest 8001, detect 8002, drift 8003, attribute 8004,
intel 8005, api 8000, ui 5173.

**Docker is used for exactly two things on this machine:** `db` (PostGIS +
TimescaleDB) and `redis`. Grafana/Prometheus/Flower move to `make prod` only.

---

## 2. Data sources — final wiring

| Layer | Source | Status | Client |
|---|---|---|---|
| SAR imagery | CDSE Sentinel Hub Process API | ✅ proven | `services/ingest/app/sources/sentinel1_cdse.py` |
| Scene metadata | CDSE OData catalogue | ✅ proven (bare ISO dates) | same module |
| Ocean currents | CMEMS GLORYS12 `cmems_mod_glo_phy_my_0.083deg_P1D-m` | ✅ proven | `copernicusmarine.subset()` |
| Winds | ERA5 via CDS OGC `/api/retrieve/v1/` | ⚠️ needs licence accept | `services/ingest/app/sources/era5.py` |
| Winds (fallback) | CMEMS wave/WRF or GFS via AWS Open Data | to verify | `gfs_provider.py` (exists) |
| Vessels, live | AISStream WebSocket | ✅ proven, coastal only | `services/ingest/app/sources/ais_aishub.py` |
| Vessels, historical | **no free source for 2020 Indian Ocean** | ❌ | handle honestly (§7) |

**Two manual actions block full real-data operation:**
1. Accept the ERA5 licence at
   `https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download`
2. `ollama pull llama3.2:3b` (≈2 GB) or accept the deterministic narrative fallback.

---

## 3. Detection: ship physics first, neural net second

There are **no labelled oil-spill masks** on this machine, and 100 epochs of
UNet++/efficientnet-b4 on an M2 CPU is many hours plus ~10 GB of dataset. Shipping a
network that has never converged would produce numbers that look authoritative and
mean nothing — exactly what the "no synthetic data" rule exists to prevent.

So the detector ships in two tiers behind one interface:

**Tier A — deterministic operator (default, ships first).** This is genuinely how
operational services screen SAR, and it is defensible in front of judges:
1. Lee-sigma speckle filter (5×5)
2. σ⁰ → dB, land-masked via `dataMask` + coastline
3. Adaptive local threshold: `mean − k·σ` over sliding windows (k tuned per scene)
4. **Wind viability gate** — slicks are only separable in ~2–10 m/s; outside that
   band the detection is flagged `LOW_CONFIDENCE`, not silently emitted
5. Morphological open/close + minimum-area + minimum-contrast filter
6. Feature vector: area, perimeter, compactness, elongation, dB contrast vs
   local sea, GLCM homogeneity/entropy
7. Look-alike scoring: rain cells, low-wind patches, biogenic films, ship wakes,
   grease ice → each writes a named penalty, so the UI can *explain* a score
8. Fay spreading inverse → age estimate with the regime stated

**Tier B — UNet++ (optional, same interface).** `unetpp_scse.py` already exists.
Populate it only if Phase 3 time allows, by training on the Zenodo DOI
`10.5281/zenodo.8320179` set at 512×512 with MPS. If untrained, the API returns
`model_status: "untrained"` rather than fake probabilities.

Every detection carries `{ score, wilson_ci_low, wilson_ci_high, method }`. The
Wilson interval is computed from the pixel-level evidence, never invented.

---

## 4. Drift engine

`AGENTS.md` mandates OpenDrift/OpenOil (not PyGNOME) and ≥3 ensemble members —
both are correct and both are kept.

**OpenDrift 1.14.11 is installed and verified running** (see VERIFICATION §2.10).
It needs a one-time shim — `awesome-slugify`'s sdist is unpackable by modern pip —
but once installed it runs a real backward hindcast on CMEMS GLORYS12 forcing.
Keep this in a `scripts/install_opendrift.sh` so it is reproducible.

Configuration: `OpenOil` with CMEMS `uo/vo` + ERA5 `u10/v10` readers, 3 ensemble
members varying `wind_drift_factor` (0.02 / 0.03 / 0.04) and
`drift:current_uncertainty`; backward run for the origin cloud, forward run for
p10/p50/p95 contours. Note the 1.14 API renames in VERIFICATION §2.10 — the
prompt's `set_config('drift:horizontal_diffusivity')` / `oiltype=` calls will
raise `ValueError` unchanged.

**Wind is mandatory**, not optional: OpenOil refuses to run on currents alone.
Accepting the ERA5 licence is therefore a hard blocker for the drift engine, and
a GFS fallback (NOAA AWS Open Data, keyless) should be built as insurance since
NOMADS OPeNDAP is retired.

If OpenDrift ever breaks again, the fallback is a 2D Lagrangian SDE with the
**Well-Mixed Criterion ∇·K correction on the backward pass** (per `AGENTS.md`, the
one thing that must never be skipped) — ~150 lines, same output schema.

Either way: 95 % covariance ellipse from the backward particle cloud, Mahalanobis
not axis-aligned, and the ensemble spread reported as a distance in km so the UI
can say "model spread 2.3 km (p50–p95)".

---

## 5. ⭐ New feature: in-platform Copernicus archive browser

*The user's explicit ask: stop making anyone leave the platform to find data.*

A first-class **Archive** route that reproduces the Copernicus Browser workflow
inside SENTINEL, backed by the same two APIs already proven:

**Layout** — three zones, no modals:
- **Left rail (280px):** AOI (draw on map / paste bbox / pick a saved AOI), date
  range, platform `S1A/S1B/S1C/S1D`, product type `GRD/SLC/OCN`, polarisation,
  orbit direction. A live "next pass over this AOI" readout.
- **Centre:** MapLibre map with real footprint polygons (GeoJSON, from OData) —
  selected scene outlined, others at 20 % opacity. SAR quicklook rendered as a
  raster layer via the Process API.
- **Right (320px):** scene metadata (product ID, acquisition UTC, orbit, mode,
  size) and the primary action: **Send to Detection**.

**Endpoints** (`services/api/app/routers/archive.py`, all proxying CDSE):

| Route | Upstream | Returns |
|---|---|---|
| `GET /api/archive/search` | CDSE OData `$filter` | scenes + footprints |
| `GET /api/archive/footprint/{id}` | OData `GeoFootprint` | GeoJSON polygon |
| `GET /api/archive/quicklook` | Process API `image/png` | 512px SAR preview |
| `GET /api/archive/nextpass` | computed from orbit phase | ETA |
| `POST /api/archive/ingest` | Process API `image/tiff` | GeoTIFF → `data/sar/` → detection job |

**Rendering note:** the quicklook evalscript returns a false-colour
(R,G,B) = (VV, VH, VV−VH) composite. Slicks read as dark with a cyan cast against
grey sea, which is the convention SAR analysts actually read. Provide a
`[VV] [VH] [ratio] [wind]` layer toggle, not a decorative basemap switcher.

This is the highest-value-per-line feature in the build: it reuses proven APIs,
it is visibly "real data", and it is the thing a judge can operate themselves.

---

## 6. UI direction — how to avoid looking AI-generated

The failure mode is specific and nameable: **unearned cards**. Symmetric rounded
rectangles floating on a gradient, each holding one number, arranged in a 3-up grid.
That reads as generated because no decision was made about hierarchy.

**SENTINEL's direction: an oceanographic instrument, not a dashboard.**

- **The map is the page.** Everything else is instrumentation bolted to its edges.
  No panel floats over the map; panels are dockable chrome.
- **Type does the work.** `Space Grotesk` for headings, `IBM Plex Mono` for every
  number/coordinate/ID (more distinctive than JetBrains Mono and slightly wider,
  which suits tabular data), and a **serif — `Newsreader`** — for the intelligence
  narrative. A serif paragraph is the single fastest way to stop a screen reading
  as a template.
- **Palette is instrument-grade, not gradient-grade.** Ink `#070B10`, hairlines at
  8 % white, chart-cyan `#5EC8D8` for data, **signal amber `#E8A33D` reserved
  exclusively for oil**. One accent per semantic meaning; nothing else is coloured.
- **Geometry:** 1–2 px radii max. Separation comes from 1px rules and negative
  space, never from shadows or blur. No glassmorphism.
- **Status is a glyph + letter code, not a coloured pill** — `■ ACT`, `□ IDLE`,
  `▲ WARN`. Colour-blind-safe and it looks like equipment.
- **Signature details:** graticule tick ruler along the map edge, crosshair reticle
  that follows the cursor with live lat/lon, subtle bathymetric contour texture in
  the background, stencil numerals for case IDs.
- **Motion:** slow and expensive-feeling — 600 ms map easing, numbers counting up
  once on load, no bouncing, no pulsing dots except the live-data heartbeat.

**Anti-patterns explicitly banned** (add to `.cursorrules`):
purple/indigo gradients · 3-up feature card rows · centered hero + CTA ·
rounded-xl everything · decorative glassmorphism · emoji as iconography ·
"Trusted by" logo rows · spinners for data loads (skeleton shimmer instead) ·
placeholder Latin copy anywhere.

---

## 7. The AIS honesty problem (and its fix)

Two facts that cannot be engineered around:

1. AISStream is **terrestrial** — verified 0 messages over the Indian Ocean AOI in
   40 s, while a global bbox yielded 48 messages in 13 s.
2. **No free historical AIS covers 2020 Mauritius.** MarineCadastre is US-waters
   only; the Danish archive is unreachable; satellite-AIS (Spire/ORBCOMM) is paid.

Rather than fabricate a Wakashio track, the UI states the situation plainly —
which is more credible to an evaluator than a suspiciously perfect one:

- Live AIS panel shows **real** vessels where coverage exists (verified working),
  with a per-region coverage indicator: `COVERED / SPARSE / NO RECEIVERS`.
- The historical case ships with **SAR + CMEMS + ERA5 evidence only** (all real),
  and the attribution panel renders a `HISTORICAL AIS UNAVAILABLE — satellite-AIS
  subscription required` state card listing exactly which source would fill it.
- `data/ais_demo.csv` is deleted. It is synthetic — 8 MMSIs × exactly 200 rows at a
  perfect 600.0 s cadence (`std = 0.0`). Shipping it would undermine every other
  claim in the product.

---

## 8. Phases

| # | Phase | Outcome | Depends on |
|---|---|---|---|
| **1** | Foundation | `.env` consolidated (dead keys removed, `OLLAMA_BASE_URL` fixed), DB init SQL verified against PostGIS, `make dev` boots 7 services | — |
| **2** | Archive browser | §5 shipped end-to-end: search → footprint → quicklook → ingest | 1 |
| **3** | Detection | Tier A detector on the 5 real GeoTIFFs; polygons + confidence + CI into PostGIS | 1 |
| **4** | Drift | OpenOil 3-member ensemble (or SDE fallback), 95 % origin ellipse, p10/p50/p95 | 1, 3 |
| **5** | Attribution | PostGIS `ST_DWithin` spatiotemporal slicing, fuzzy Cauchy scorer, Wilson CI, dark-vessel injection | 3, 4 |
| **6** | UI shell | Design system from §6, WatchRoom + CaseInvestigation + Archive + DataHealth | 2–5 |
| **7** | Intel | Narrative (Ollama or deterministic), WeasyPrint dossier, SHA-256 chain of custody | 5 |
| **8** | Hardening | `make demo` clean, pytest ≥80 %, README with architecture diagram | all |

Phase 2 is deliberately early: it is the user's requested feature, it is
independently demoable, and it validates the SAR pipeline that Phases 3–5 consume.

**Disk budget** (from 39 GB free):

| Item | Size |
|---|---|
| SAR GeoTIFFs (5 done, allow 20 scenes) | 180 MB → 0.7 GB |
| CMEMS/ERA5 forcing per case | ~0.5 GB |
| Ollama model (optional) | 2 GB |
| Docker images + volumes | ~4 GB |
| `node_modules` (exists) | 0.9 GB |
| **Total** | **≈ 8 GB** — comfortable |

---

## 9. Model & effort recommendation

You asked for a model that completes this without eating the credit budget. The
trap is running all 8 phases in one session with the strongest model: context
bloat makes late phases slower *and* worse.

**Recommended split:**

| Work | Model tier | Why |
|---|---|---|
| Architecture, design system, detector tuning, drift physics, gnarly API/debug failures | **Top tier** (this session's class) | These are judgement calls where a wrong turn costs a whole phase |
| Page scaffolding, CRUD routers, tests, docs, SQL migrations, repetitive components | **Fast/mid tier** | Mechanical, verifiable, and cheap to redo |
| Independent workstreams (drift vs UI vs archive) | **Parallel sub-agents** | Wall-clock down 3×, each with a clean context |

**Sequencing that keeps costs down:**
1. Finish the design system + one exemplar page at top tier.
2. Hand the exemplar + tokens to a mid-tier pass to replicate the remaining pages.
   One exemplar is worth more than a thousand words of instruction.
3. Reserve top tier for the three genuinely hard parts: detector threshold tuning,
   drift/WMC correctness, and the archive browser's API contract.

**Realistic estimate:** 5–7 focused sessions, not one. Roughly 60 % of the code
already exists; the work is *correction and wiring*, not greenfield — the audit
found a working 7-service skeleton and a 35-file React app.

**Do first, before any more code:** the two manual actions in §2 (ERA5 licence,
Ollama model). Both gate real-data claims and neither needs me.
