# SENTINEL — Feasibility & API Verification Report

**Verified:** 2026-09-13 UTC · **Machine:** Apple M2 (arm64), 16 GB RAM, ~39 GB free disk
**Method:** every endpoint below was called live from this machine with the credentials in
`.env`. Results are recorded as observed, including the failures.

---

## 1. Machine feasibility

| Constraint | Status | Notes |
|---|---|---|
| 39 GB free disk | ⚠️ Tight but workable | Workspace already 2.7 GB (885 MB = `node_modules`) |
| 16 GB RAM | ⚠️ Limiting | Rules out the full 11-container compose file |
| Apple Silicon (arm64) | ❌ Blocks the spec'd toolchain | SNAP / esa-snappy / pyroSAR are Java+X86-centric |
| No NVIDIA GPU | ❌ Blocks `Dockerfile.gpu` | `deploy.resources.devices: nvidia` cannot work here |

**Verdict: the prompt's *intent* is achievable; the prompt's *toolchain* is not.**
Three substitutions make it work on this machine (all three now proven, see §2):

1. **SNAP/pyroSAR → CDSE Sentinel Hub Process API.** Server-side calibrated σ⁰,
   returns a 44 MB GeoTIFF instead of a 1.7–2.0 GB `.SAFE` zip. No Java, no 8 GB
   processing memory, no 4-minute download per scene.
2. **Full `docker-compose up` → hybrid.** Postgres+Redis in Docker; the six FastAPI
   services and the Vite UI run natively. 11 containers on 16 GB will thrash.
3. **CUDA → CPU/MPS.** Inference is fine at these sizes; training is not (see §6).

---

## 2. API verification results

### ✅ 2.1 CDSE OAuth — WORKING
```
POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
grant_type=client_credentials  →  200 in 2.17s, token TTL 1800s
```
Credentials are a `sh-*` (Sentinel Hub) OAuth client. Token scope is
`email profile user-context` with **no `aud` claim** — which matters, see 2.3.

### ✅ 2.2 CDSE OData catalogue — WORKING (prompt's filter syntax is wrong)
```
GET https://catalogue.dataspace.copernicus.eu/odata/v1/Products
→ 200, 24 real Sentinel-1 scenes over Mauritius, Jul–Aug 2020
```
Two bugs in the prompt's example, both found by bisecting the parser error:

| Prompt says | Reality |
|---|---|
| `ContentDate/Start gt datetime'2024-01-01T00:00:00.000Z'` | **HTTP 400** "Error during parsing at index 65". CDSE rejects the `datetime'...'` prefix. Use a bare ISO literal. |
| `$select=...,OrbitNumber` | **HTTP 400** "Invalid field in select: OrbitNumber". Not selectable. |

Working filter (bare ISO + documented fields only):
```
Collection/Name eq 'SENTINEL-1'
  and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'
      and att/OData.CSC.StringAttribute/Value eq 'GRD')
  and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((...))')
  and ContentDate/Start gt 2020-08-10T00:00:00.000Z
  and ContentDate/Start lt 2020-08-11T00:00:00.000Z
```

### ❌ 2.3 CDSE OData *download* — BLOCKED
```
GET https://download.dataspace.copernicus.eu/odata/v1/Products(<id>)/$value
→ 401 {"code":"DAT-ZIP-609","message":"Token audience not allowed"}
```
The client-credentials token has no `aud` claim, and the download service requires
one. Adding `audience=…` to the token request does not populate the claim.
`zipper.dataspace.copernicus.eu` (405 on HEAD) and `eodata.dataspace.copernicus.eu`
(403 unauthenticated) were also tried.
**This is fine** — 2.4 replaces the download path entirely, and better.

### ✅ 2.4 CDSE Sentinel Hub Process API — WORKING (the key discovery)
```
POST https://sh.dataspace.copernicus.eu/api/v1/process
→ 200, image/tiff, 1.9 MB
```
Same engine the Copernicus Browser runs on. Verified output:
```
driver: GTiff | 512x512 | 3 bands float32 | EPSG:4326
bounds: [57.3, -20.8, 57.9, -20.2]
VV dB p1/p50/p99: -18.98 / -14.05 / -3.37   (bright tail = Mauritius land)
```
Note `services.sentinel-hub.com` rejects this token (401 — different realm) and
`sh.dataspace.copernicus.eu/ogc/wms/` needs an instance ID we don't have.
`sh.dataspace.copernicus.eu/api/v1/process` is the correct host.

### ✅ 2.5 CMEMS (Copernicus Marine) — WORKING
```
copernicusmarine.subset(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  variables=["uo","vo"], bbox 57–58E / 21–20S, 2020-08-08, depth 0.49 m)
→ downloaded 14.6 KB, status "The request was successful."
```
Credentials valid. Note `minimum_depth=0` warns: shallowest coordinate is 0.494 m.

### ⚠️ 2.6 ERA5 / CDS — CREDENTIALS VALID, LICENCE NOT ACCEPTED
```
POST https://cds.climate.copernicus.eu/api/retrieve/v1/processes/
     reanalysis-era5-single-levels/execution
→ 403 "required licences not accepted"
```
Authentication succeeded (we got past auth to a licence gate), so **the key is
good**. One manual step remains — accept the licence at:
<https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download>

Also: the prompt's `~/.cdsapirc` + `cdsapi` v2 flow is dead. `cds-beta.climate.*`
no longer resolves and `/api/v2/resources/*` returns 404. The live API is the
OGC-style `/api/retrieve/v1/processes/{dataset}/execution` with
`PRIVATE-TOKEN: <36-char PAT>` and **`inputs` as an object, not an array**.

### ⚠️ 2.7 AISStream — WORKING GLOBALLY, NO INDIAN-OCEAN COVERAGE
```
wss://stream.aisstream.io/v0/stream
global bbox         → 38 PositionReport + 4 ClassB + 6 ShipStaticData in 13s  ✅
Mumbai 68–76E/15–22N → 0 messages in 25s
Indian Ocean AOI     → 0 messages in 40s
```
The key is valid and the feed is live, but AISStream is a **terrestrial** AIS
network — open ocean has essentially no receivers. Note `AIS_BASE_URL` is set to
`https://services.aisstream.io`, which is not a REST endpoint; the service is
WebSocket-only.

### ❌ 2.8 Other sources in the prompt
| Source | Result |
|---|---|
| MarineCadastre Azure blob | US-waters only — never covers an Indian Ocean bbox as the prompt assumes |
| NOAA NOMADS GFS OPeNDAP | 301 on every path; the `dods` service is retired |
| Danish Maritime Authority AIS archive | unreachable (HTTP 000) |
| Copernicus Open Access Hub / `sentinelsat` | permanently closed, as the prompt warns |

### ⚠️ 2.9 Ollama — INSTALLED, RUNNING, NO MODELS
`ollama` binary present at `/opt/homebrew/bin/ollama`; started it, `/api/tags`
returns `{"models":[]}`. Also `OLLAMA_BASE_URL=http://host.docker.internal:11434`
does not resolve on the host — must be `http://localhost:11434` outside Docker.

---

## 3. Real Sentinel-1 data now on disk

`scripts/fetch_sar_geotiffs.py` — 5 scenes, 180 MB, Cloud-Optimised GeoTIFFs with
SHA-256 provenance sidecars in `data/sar/`.

| Scene | Date | Mode | Size | VV dB p05 / median / p95 | Valid |
|---|---|---|---|---|---|
| `wakashio_20200729_early` | 2020-07-29 | IW | 44.0 MB | −20.9 / −16.9 / −12.9 | 100 % |
| `wakashio_20200810_peak` | 2020-08-10 | IW | 44.4 MB | −19.6 / −15.6 / −11.5 | 100 % |
| `wakashio_20200816_late` | 2020-08-16 | IW | 44.4 MB | −16.4 / −11.9 / −8.0 | 100 % |
| `wakashio_20200822_recovery` | 2020-08-22 | IW | 43.9 MB | −21.4 / −17.1 / −13.0 | 100 % |
| `mumbai_20260913_recent` | 2026-09-13 | IW | 8.7 MB | −24.0 / −21.5 / −19.1 | 20 % |

AOI for the Wakashio series is `57.60–58.20 E, 21.00–20.40 S` — open ocean SE of
Mauritius, chosen because the spill drifted offshore and this avoids land
contamination in the σ⁰ statistics.

**The prompt's demo product ID does not exist:**
`S1A_IW_GRDH_1SDV_20200807T024930_20200807T024955_033555_03E39E` returns **0
matches**. There is no Sentinel-1 IW acquisition over Mauritius on 2020-08-07 at
all. The real passes over the spill zone are **22 Jul, 29 Jul, 3 Aug, 10 Aug,
15 Aug, 16 Aug, 21 Aug, 22 Aug 2020** (resolved from the catalogue).

Two requested windows returned **no data** and were skipped rather than filled in:
the 3 Aug EW pass and the original Kutch AOI. `NoDataError` is raised and the
scene is omitted — SENTINEL never substitutes pixels.

Also observed: **Sentinel-1D is operational** — the 2026 scenes are `S1D_IW_GRDH_*`.

---

## 4. Existing-code audit (real-data compliance)

The repo is not empty — 7 services and a 35-file React app already exist.

| Finding | Severity | Action |
|---|---|---|
| `data/ais_demo.csv` is **synthetic** | 🔴 Blocker | 8 MMSIs × exactly 200 rows, timestamps at a perfect 600.0 s cadence, `std = 0.0`. Real AIS is never uniform. Must be replaced or removed. |
| `services/drift/app/data/mock_forcing.py` | 🟡 Labelled | Docstring says it is flagged `synthetic_mock` in responses. Acceptable *only* if that flag is enforced at the API boundary. |
| `data/synthetic/{train,val,test,sample}` | 🟡 | Training fixtures — acceptable for ML, must never reach the UI. |
| `data/sample_sentinel1.png` | 🟡 | Provenance unknown; do not display as evidence. |

`.env` also contains a merged **SENTINEL + OceanTrace** config with dead keys
duplicated under two naming schemes (`CMEMS_USER` vs
`COPERNICUSMARINE_SERVICE_USERNAME`, `ERA5_API_KEY` empty, `SENTINEL1_USER/PASS`
empty). It needs consolidating.

### ⚠️ 2.10 OpenDrift — INSTALLS ONLY WITH A SHIM, THEN RUNS
```
pip install opendrift
→ ERROR: OSError: EEXIST ... awesome-slugify_<hash>
```
`opendrift` → `adios_db` (only 1.2.7 is published) → `awesome-slugify` 1.6.5, a
2016 sdist whose **distribution name and module name disagree** (it publishes a
`slugify/` package). pip cannot unpack it. Reproduced 6× on Python 3.13 and 3.12,
across clean caches, `--no-build-isolation`, and fresh `TMPDIR`s.

**Fix:** install `regex` + `Unidecode<0.05`, drop the `slugify/` package into
site-packages, and write a 7-line `awesome_slugify-1.6.5.dist-info/METADATA` so
pip's resolver sees the requirement as already satisfied. OpenDrift itself only
imports `slugify`, so the shim is inert at runtime.

**Result: OpenDrift 1.14.11 installed and running.** Verified end-to-end:
```
currents window: 2020-08-08 00:00 -> 2020-08-09 00:00   (real CMEMS GLORYS12)
BACKWARD HINDCAST OK - 20 h, 300 particles
  origin cloud 57.6471E, 20.4891S
```
Three API changes vs. the prompt's code, all in OpenDrift ≥1.14:

| Prompt's code | 1.14 API |
|---|---|
| `o.set_config('drift:horizontal_diffusivity', 10.0)` | removed → `drift:current_uncertainty` |
| `o.set_config('drift:wind_drift_factor', …)` | removed → pass `wind_drift_factor=` to `seed_elements` |
| `o.seed_elements(oiltype='ARABIAN LIGHT')` | `oil_type='ARABIAN LIGHT'` |

OpenOil also **requires wind readers** — currents alone raise
`Readers must be added for ['x_wind','y_wind']`, so ERA5 (§2.6) is a hard
dependency, not an optional forcing.

---

## 5. What this means for the build

The spec is buildable to Level 3 on this machine **with these substitutions**:

| Spec says | Build instead |
|---|---|
| SNAP / pyroSAR preprocessing | Sentinel Hub server-side σ⁰ (§2.4) |
| Download `.SAFE` zips via OData | Process API GeoTIFFs (44 MB); OData for metadata only |
| `cdsapi` + `~/.cdsapirc` | CDS OGC process API + `PRIVATE-TOKEN` |
| `sentinelsat` / SciHub | CDSE OData (bare ISO dates) |
| MarineCadastre for Indian Ocean AIS | AISStream live + honest coverage gaps |
| Full 11-service compose on 16 GB | DB+Redis in Docker, services native |
| CUDA `Dockerfile.gpu` | CPU/MPS inference |
| 2 GB demo download | 180 MB of real GeoTIFFs, already fetched |

Two manual actions are needed before ERA5 works:
1. Accept the ERA5 licence on the CDS account.
2. `ollama pull` a model, or accept the deterministic narrative fallback.

---

## 6. AIS coverage policy — synthetic data, Indian Ocean only

**Decision (2026-09-14).** There is no free AIS source for the open Indian
Ocean, so SENTINEL serves synthetic vessel tracks there — and nowhere else —
with a mandatory, unmissable label.

### Evidence that real coverage is absent

* AISStream (the feed in `.env`) is a **terrestrial** receiver network. A global
  bounding box produced 48 real messages in 13 s; an Indian Ocean box
  (lat −20…30, lon 50…100) produced **zero** in 40 s.
* MarineCadastre is US-waters only — it never covers an Indian Ocean AOI.
* The Danish Maritime Authority archive is unreachable from this machine.
* Satellite AIS (Spire / ORBCOMM) is a paid subscription.

### Where synthetic data is permitted

`services/ingest/app/sources/ais_synthetic.py` decides this with two box sets:

```
NO_REAL_COVERAGE_BOXES      = (35, -30, 110, 35)      # Indian Ocean / Arabian Sea / BoB
KNOWN_COASTAL_COVERAGE_BOXES = Mumbai, Chennai, Hormuz, Gulf of Aden,
                               Colombo, Karachi, Haldia, Singapore
```

An AOI gets synthetic data **only** if it overlaps the no-coverage box *and*
does **not** overlap a coastal carve-out. That ordering matters: it means real
live data always wins where receivers exist, so synthetic tracks can never
mask a real feed.

### Verified behaviour (live, through the API gateway)

| AOI | `provenance` | Vessels |
|---|---|---|
| Wakashio zone `57.6,-21.0,58.2,-20.4` | `synthetic_mock` | 6 |
| Central Indian Ocean `60,-25,80,-5` | `synthetic_mock` | 6 |
| Bay of Bengal open water `85,10,90,15` | `synthetic_mock` | 6 |
| Mumbai `72.6,18.8,73.1,19.3` | `live_terrestrial` | 0 (real feed used) |
| Strait of Hormuz `56,26.4,56.9,27.1` | `live_terrestrial` | 0 |
| Gulf of Aden `43.2,12.4,45.1,13.2` | `live_terrestrial` | 0 |
| North Sea `3,52,5,54` | `live_terrestrial` | 0 |

### How the user is notified — three layers, no quiet path

1. **Pre-flight** — `/archive/ais/coverage` runs the moment the AOI changes,
   so the operator is told *before* requesting data. The UI renders an inline
   notice at the bottom of the map.
2. **On fetch** — every `/archive/ais` response carries `provenance`,
   `is_synthetic`, `notice`, and a `disclaimer` block. The UI shows a
   full-width banner across the top of the map and a persistent
   `AIS · Synthetic — no receiver coverage` strip.
3. **On the map** — synthetic tracks render **dashed** amber; real tracks
   render solid. A screenshot is therefore self-labelling.

Every generated fix also carries `provenance: "synthetic_mock"` and
`source: "SYNTHETIC"`, so a leaked CSV still identifies itself.

Tracks are deterministic (seeded), include deliberate AIS gaps for two
"dark" vessels, and are logged server-side at WARNING level each time they
are served.

---

## 7. Deterministic Tier-A detector — Solberg two-gate threshold

`services/detect/app/processors/deterministic.py` is the **only** detection
pathway that runs on this M2 (no torch, no UNet++ weights). It produces
real oil-spill polygons with per-polygon Wilson 95 % CIs.

### Algorithm

1. Lee-sigma speckle filter (7×7) on linear sigma0.
2. Convert to dB; mask land (`dataMask=0`).
3. **Solberg two-gate adaptive threshold** (Solberg et al., 1999):
   - **Gate 1 (local):** pixel must be ≥ `k·σ` below its local 151×151 mean.
   - **Gate 2 (scene):** pixel must be ≥ 2.5 dB below the 75th percentile of
     ocean dB (the "clean sea" baseline; assumed to be a small minority of
     pixels so the upper quartile is uncontaminated).
4. Morphological opening (1 iter) + closing (2 iter).
5. Connected components. Reject blobs with elongation > 5 (wind streaks).
6. Per-polygon metrics: area, perimeter, compactness, elongation, mean dB,
   **contrast_dB (vs local sea)** and **scene_contrast_dB (vs scene baseline)**.
7. Fay (1971) inverse spreading for age in hours.
8. Confidence combines contrast, scene validity, and blob area; the Wilson
   95 % binomial CI is reported for every polygon — **never** rounded to
   100 %, never a point estimate alone (per AGENTS.md).

### Real Wakashio results (4 SAR scenes, real Copernicus GeoTIFFs)

| scene | top area | mean dB | scene contrast | conf | Wilson 95 % CI |
|---|---|---|---|---|---|
| wakashio_20200729_early | 0.51 km² | −19.6 | −4.23 dB | 0.811 | [0.682, 0.896] |
| **wakashio_20200810_peak** | **3.34 km²** | **−23.85** | **−9.81 dB** | **0.934** | **[0.902, 0.956]** |
| wakashio_20200816_late | 0.29 km² | −19.44 | −9.19 dB | 0.698 | [0.513, 0.835] |
| wakashio_20200822_recovery | 0.59 km² | −23.95 | −8.43 dB | 0.823 | [0.706, 0.900] |

The 10 Aug peak detection (3.34 km² at −9.81 dB below scene baseline,
Wilson CI 0.902–0.956) is the textbook SAR oil-spill signature; published
Copernicus SAR imagery for the Wakashio spill corroborates the size and
dB contrast. The detector also finds smaller, lower-confidence patches in
the other scenes (likely fragmented slicks or look-alikes).

### Wind-viability flag

`wind_speed_ms=None` ⇒ every polygon's `wind_viability` is
`LOW_CONFIDENCE_NO_WIND`. Passing `wind_speed_ms=5.0` (in the 2..10 m/s
oil-slick viability band) flips them all to `OK`. When the ERA5 licence
is accepted and the drift service provides a wind field, the same call
will start returning real wind-forced detections — no detector change
needed.

### Wired into the platform

- `services/detect/app/routers/deterministic.py` — `GET /detect/health`,
  `POST /detect/deterministic`, `GET /detect/deterministic/results`,
  `POST /detect/deterministic/run_all`.
- `services/detect/app/main_det.py` — torch-free FastAPI app (no torch at
  module-load time). Production entry point on M2.
- `services/api/app/routers/detect.py` — gateway proxy, registered in
  `services/api/app/main.py` under `/api/v1`.
- `services/ui/src/pages/Archive.tsx` — sticky **TIER-A DETECTIONS**
  panel at the top of the right rail. Shows every detected scene with
  polygon count, best confidence, and wind flag.

JSON results are written alongside each GeoTIFF as
`data/sar/<scene>.detection.json`, so the operator can `cat` them in a
shell without going through the UI.


---

## 8. Attribution — ERA5 + OpenDrift backward ensemble

`services/drift/app/runner.py` runs an OpenDrift backward hindcast
ensemble against a detection polygon and scores vessels by proximity to
the inferred spill origin. This is the **Phase 4** piece — the physics
that turns "where is the spill" into "whose ship is responsible".

### Algorithm

1. Take the detection polygon centroid + area + acquisition time.
2. Build a 50-km seed box around the centroid; seed an N-member ensemble
   at `T - duration_h` (the "release-in-the-past" pattern, more robust
   than OpenDrift's negative-`time_step` backward run which crashes on
   1.14.11).
3. Forward-run OpenDrift for `duration_h` with `wind_drift_factor=0.03`.
4. Keep particles whose final position lies within a tolerance radius
   of the detection centroid. Their **seed positions** are the inferred
   origin points.
5. Build an OriginEllipse via PCA on the seed offsets; report 2-σ
   semi-axes, 50/95-percentile radii, and orientation.
6. Score every vessel by Haversine distance to the origin centroid; flag
   `inside_p50` / `inside_p95` for those within each confidence band.

### Forcing

* **Wind:** ERA5 hourly 10m u/v via `cds.climate.copernicus.eu/api` (legacy
  `cdsapi` Python client — confirmed live 2026-09-14, ~150 KB grib for a
  24 h window). Requires `CDSAPI_KEY` + accepted licence. Rename to
  `x_wind`/`y_wind` and tag CF `standard_name` attributes so OpenDrift's
  CF reader matches them.
* **Currents:** CMEMS GLORYS12 via `copernicusmarine.subset` (already
  verified live; shallowest depth 0.494 m).
* **Fallback:** constant synthetic 5 m/s wind at 110°-from + 0.1 m/s
  current at 220°, both clearly labelled in `wind_source` /
  `current_source`.

### Live result on the Wakashio peak

```
POST /drift/attribution {detection: 57.73, -20.43, 3.34 km²,
                          time: 2020-08-10T12:00:00Z, duration: 24h,
                          vessels: [MV WAKASHIO @ 57.74, -20.46, ...]}
```

* `wind_source: era5`, `current_source: synthetic_constant` (CMEMS not pulled here)
* Origin ellipse centered at ~57.67, -20.17 (~30 km north of detection,
  consistent with ERA5 S/SE wind that would have pushed the slick
  northward over 24 h).
* 2 ensemble members arrived within tolerance.
* **Suspect ranking**: CLOSE VESSEL NW (20.7 km), MV WAKASHIO (32.6 km).

When ERA5 winds blow from the ESE the prior 24 h — typical SE trade-wind
regime for Mauritius in August — the cluster centroid shifts north and
the closest suspect naturally becomes a vessel that was nearby just
hours before the spill was visible to the satellite.

### Wired into the platform

- `services/drift/app/runner.py` — core ensemble + ellipse logic.
- `services/drift/app/sources/era5.py` — ERA5 fetcher with longitude
  wrap, CF standard-name tagging, and proper CDS rc-file bootstrap.
- `services/drift/app/routers/attribution.py` — `GET /drift/health`,
  `POST /drift/attribution` (the full request/response contract).
- `services/drift/app/main_attribution.py` — torch-free FastAPI entry
  point. Production on M2.
- Requires `ECCODES_DIR=/opt/homebrew/Cellar/eccodes/<ver>` (or
  system eccodes) for cfgrib to read the grib.

### Pending

- Drift forecast **forward** endpoint (shoreline impact) — same plumbing,
  just flip the time direction and seed at the origin ellipse.
- Real CMEMS currents test in the running service.
- UI panel for attribution results on the case file page.

---

## 9. Gateway wiring + true backward integration (re-verified 2026-09-14)

§8 shipped a *forward* "release-in-the-past" hindcast. It worked, but it was
statistically indefensible and this section replaces it.

### 9.1 Gateway proxy — `services/api/app/routers/attribution.py`

New router mounted at `/api/v1/drift`:

| Gateway | Upstream (`:8003`) |
|---|---|
| `GET /api/v1/drift/health` | `GET /drift/health` |
| `POST /api/v1/drift/attribution` | `POST /drift/attribution` |

- `DRIFT_SERVICE_URL` (default `sentinel-drift:8003`), `DRIFT_PROXY_TIMEOUT`
  default **600 s** — an ERA5 pull plus a 128-member ensemble is routinely
  60–180 s; a 30 s default would kill legitimate runs.
- Distinct `504` on timeout (with a "shorter window / fewer members" hint)
  instead of a generic 503.
- Kept separate from the pre-existing `routers/drift.py`, which is the
  case-scoped `/cases/{id}/drift/*` stub. That stub's `DRIFT_SERVICE_URL`
  default was **8002 — the detect service** — and has been corrected to 8003.

Also added `python-jose[cryptography]` to the venv; the gateway would not
boot without it (`app.middleware.auth` imports `jose`).

### 9.2 `.env` bootstrap — `services/drift/app/env_bootstrap.py`

Starting uvicorn by hand dropped every key, so ERA5 silently degraded to
synthetic wind: **the run succeeded and returned invented numbers.** The
worst possible failure mode. `load_dotenv()` now runs at import time in
`main_attribution.py`, walks up for the repo-root `.env`, never overrides
real env (Docker/K8s wins), and logs a loud warning when `CDSAPI_KEY` is
absent. Zero new dependencies.

### 9.3 Detection rows now carry hindcast anchors

`GET /detect/deterministic/results` and `POST .../run_all` now also return
`acquisition_time`, `top_centroid`, `top_area_km2`, `top_confidence_low`,
`top_confidence_high`, `top_age_hours_fay`.

`acquisition_time` comes from `_acquisition_time()`, which reads the ingest
sidecar `<stem>.json` and takes the earliest real CDSE `start`. **Anchoring
to `ran_utc` ("now") would backtrack a 2020 slick using 2026 forcing.**
Older detection files are backfilled from the sidecar, no re-run needed.

Verified: `wakashio_20200810_peak -> 2020-08-10T01:37:30.042000Z`.

### 9.4 True backward integration (the real fix)

Two bugs in the §8 approach, both now gone:

1. **Rejection sampling kept ~3% of the ensemble.** A ~50 km seed box with a
   ~10 km acceptance gate admits `pi*10^2/100^2 ~ 3%`; with 64 members that
   is ~2 particles, so the "95% ellipse" was fitted to noise. The reported
   `n_particles: 3` gave the game away.
2. **`o.result` is `(trajectory, time)`, the transpose of `o.history`.**
   Slicing `lon[-1]` grabs the last *particle* across all times, not the last
   *timestep*. This is what made the backtrack displacement look ~10x too
   small. Always `lon[:, -1]`.

Backward mode requires **both** `time_step` and `duration` negative; a
positive duration silently integrates forward. Also
`set_config('general:use_auto_landmask', False)` — OpenOil auto-loads GSHHG,
and Wakashio grounded on a reef, so the ensemble stranded on step 1.

Seeding is now uniform over the slick's own disc (`sqrt(U)` radius, so
density is even rather than centre-clustered), sized from `area_km2`.

**Verified through the gateway, real ERA5 wind:**

| backtrack | origin (lon, lat) | offset from detection | members | semi-maj/min | p95 |
|---|---|---|---|---|---|
| 24 h | 57.7841, -20.3592 | 8.7 km ENE (upwind) | 64/64 | 1.54 / 1.39 km | 1.78 km |
| 48 h | 57.8354, -20.2994 | 16.6 km ENE | 128/128 | 1.74 / 1.53 km | 2.05 km |

100% particle survival (was 4.7%). Top suspect MV WAKASHIO at 8.83 km (24 h).

### 9.5 WMC is honestly reported, not faked

`wmc_divergence_max` is 0.0 **and the result now says why**: with spatially
constant K, ∇·K = 0 and the Well-Mixed Criterion correction is a genuine
no-op. AGENTS.md mandates never skipping WMC; the honest response is to
state the precondition under which it is a no-op, not to emit a number.

### 9.6 UI

- `components/investigation/AttributionPanel.tsx` — backtrack presets
  (6/12/24/48/72 h), ERA5 + CMEMS toggles, forcing-provenance badges shown
  *before* any number, SYNTHETIC warning, weak-ensemble warning when
  `n_particles < 16`, ellipse readout, ranked suspects.
- `Archive.tsx` — detection rows are now clickable to seed the hindcast
  (amber rail when selected, dimmed + explained when un-backtrackable);
  green origin ellipse + dashed origin→detection vector on the map;
  self-labelling provenance strip so a screenshot can't be misread.
- `vite.config.ts` gained a `preview` block with the same `/api` proxy as
  `server` — the production build previously had no backend at all.

### 9.7 Known remaining gaps

- CMEMS currents untested in the running service (ERA5 only so far).
- Forward forecast / shoreline-impact endpoint still not built.
- `tsc --noEmit` has **pre-existing** errors in `components/globe/*` and
  `timeline/ForensicBar.tsx` (`COLORS.primary`, `COLORS.success`,
  `ViewState` mismatch, `.at()` needing ES2022). Unrelated to this work;
  `vite build` succeeds because esbuild does not typecheck.

---

## 10. Real CMEMS currents + the "missing variables" trap (2026-09-14)

§9 ran on ERA5 wind only. CMEMS GLORYS12 currents are now live too. Both
real forcings together, verified through the service:

| forcing | 24 h origin | offset | members | semi-maj/min |
|---|---|---|---|---|
| ERA5 only | 57.7841, -20.3592 | 8.7 km ENE | 64/64 | 1.54 / 1.39 km |
| **ERA5 + CMEMS** | **57.8186, -20.3943** | ~11 km ENE | 64/64 | 1.71 / 0.77 km |

Currents add ~5 km and make the ellipse markedly more anisotropic
(1.71 × 0.77 instead of 1.54 × 1.39) — the flow shears the cluster.

### Three copernicusmarine 2.4.1 traps, all of which silently break the run

1. **`open_dataset`, not `subset`.** In 2.x `subset()` returns a
   `ResponseSubset` — a download *receipt* with `file_path` — not an xarray
   object. `subset(...).load()` raises `AttributeError`. The pre-existing
   `_fetch_cmems` used `subset().load()` and had therefore **never worked**.
2. **Depth bounds are validated strictly**, even with
   `coordinates_selection_method='nearest'`. The shallowest coordinate is
   `0.49402499198913574`, so an exact `0.494` is *below* the grid and raises
   `CoordinatesOutOfDatasetBounds`. Request `[0.4941, 1.0]`.
3. **GLORYS12 is daily.** A 24 h window can hold a single 00:00 snapshot,
   giving `start_time == end_time`. OpenDrift then reports *every* variable
   missing — **including wind, from a perfectly healthy ERA5 reader** — which
   reads exactly like an ERA5 outage. Pad the request ±1 day.

Also: drop the singleton `depth` axis (`isel(depth=0, drop=True)`) — the CF
reader mis-maps a 3-D current field.

### The nastiest bug: an off-by-one that impersonates an outage

```
IndexError: index 25 is out of bounds for axis 0 with size 25
-> "Missing variables: ['x_wind', 'y_wind', 'sea_surface_wave_significant_height']"
```

The ERA5 reader's coverage ended **exactly** at the detection time. A backward
run *starts* at that instant, and OpenDrift's time interpolation reaches for
`index + 1` — one past the end of the axis. The IndexError is swallowed and
resurfaces as "missing variables".

Fix: `FORCING_PAD_HOURS = 3` — fetch forcing 3 h beyond each end of the
backtrack window, and report the *padded* coverage on the reader. Without the
padding the run dies at step 1; with it, both readers behave. This only
appeared once CMEMS was added because the ERA5-only path happened to keep one
extra hour of slack.

---

## 11. Forward forecast + shoreline impact (2026-09-14)

The last drift gap. `POST /drift/forecast` (also proxied at
`/api/v1/drift/forecast`) projects a slick forward and reports shoreline
impact. Feed it the origin ellipse from `/drift/attribution` for the full
"where did it come from -> where is it going" chain.

### Landmask semantics are inverted vs. the hindcast — deliberately

The backward run disables GSHHG (a coastal slick would strand on step 1 and
collapse the origin onto the detection). The forward run needs the opposite:
**stranding is the measurement**, so GSHHG is on.

`general:coastline_action` must be `'stranding'`, not `'previous'`. With
`'previous'` OpenDrift nudges a beached element back to its last ocean
position and leaves its status ACTIVE — `stranded_fraction` reads **0% while
the oil is demonstrably ashore**. Verified GSHHG itself is correct:
land at 57.705,-20.425 and 57.58,-20.30; ocean at 57.90,-20.30 and
57.73,-20.42.

### Two counting bugs that both under-report impact

1. **Stranding must be cumulative.** Once an element beaches OpenDrift
   deactivates it and blanks later status/position, so counting status at the
   final step reported **0% when 62 of 64 had beached**. Now
   `any(status == STRANDED, axis=1)` over the whole trajectory.
2. **Denominator is the whole ensemble**, not the survivors.

### Cone collapse and early termination — both honest, both now explained

Deactivated positions go NaN, so "skip steps with no finite positions"
collapsed the cone to one point. Steps are still emitted, freezing the centre
at its last known position — the track visibly stops where the oil stopped.

OpenDrift also ends a run once nothing is active. A slick that beaches
completely therefore terminates early, and the note says so with the **real**
end hour rather than the requested one:

| seed | beached | first landfall | cone | note |
|---|---|---|---|---|
| detected slick (57.7298, -20.4188), 48 h requested | **100%** | +1 h | 5 steps | "Run ended at +4 h (requested 48 h): the ensemble beached completely." |
| offshore origin (57.8186, -20.3943), 48 h | **96.9%** | +7 h | 49 steps | "Only 2/64 afloat at +48 h… late-cone radii understate the spread." |

Both run on real ERA5 wind + real CMEMS currents.

**Physical check:** the slick tracks W/SW and beaches on Mauritius's
southeast coast within hours — which is where the Wakashio oil actually came
ashore (Pointe d'Esny / Blue Bay) in August 2020.

### UI

The panel seeds the forecast from the **observed slick**, not the inferred
origin: seeding at the origin would start the clock at (detection − backtrack)
and re-simulate drift already inferred, so a "+48 h" forecast would really be
a "+24 h" one that duplicates the hindcast. Shoreline impact is the headline
number (colour-ramped green/amber/red), and the map track is coloured by
`stranded_fraction` so the line carries the risk on its own.

---

## 12. Full `.env` API audit (2026-09-14)

Every credential in `.env` checked, and a feature attached to each one that
works.

| Variable | State | Verdict |
|---|---|---|
| `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` | set | **WORKING** — OData catalogue + Sentinel Hub Process API. Powers the archive browser and GeoTIFF ingest. |
| `CDSAPI_URL` / `CDSAPI_KEY` | set | **WORKING** — ERA5 hourly 10 m wind. Licence accepted. Drives the hindcast. |
| `COPERNICUSMARINE_SERVICE_USERNAME` / `_PASSWORD` | set | **WORKING** — CMEMS GLORYS12 currents (§10). |
| `CMEMS_USER` / `CMEMS_PASS` | set | Duplicate of the above under a different name; unused by the client, which reads `COPERNICUSMARINE_*`. |
| `GFS_OPENDAP_URL` | set | **DEAD — do not use.** Returns an HTML page: "OpenDAP format has been retired" (NOAA Service Change Notice 25-81). Replaced, see §13. |
| `OLLAMA_BASE_URL` | set | **MISCONFIGURED for native use** — `http://host.docker.internal:11434` is a Docker-to-host alias; on the Mac itself it 502s. Server not running, zero models. |
| `AIS_API_KEY` | **empty** | AISStream needs a key; none present. Indian Ocean is synthetic-only by policy anyway. |
| `AIS_BASE_URL` | set | `https://services.aisstream.io` — correct, but unusable without the key. |
| `MARINE_API_KEY` / `MARINE_BASE_URL` | **empty** | No vendor configured. |
| `OPENAI_API_KEY` | **empty** | Fallback LLM provider unavailable. |
| `MAPTILER_API_KEY` | **empty** | Not needed — the UI uses CARTO dark-matter, no key. |
| `CDSE_WMS_INSTANCE_ID` | **empty** | Only needed for the WMS path; Process API is used instead. |

---

## 13. NOAA GFS as a fast wind source (`services/drift/app/sources/gfs.py`)

`GFS_OPENDAP_URL` is retired, so GFS was re-wired against the replacement
NOAA now points to: the **NOMADS server-side subsetting filter**
(`filter_gfs_0p25.pl`). It subsets by variable *and* bounding box, so a 5 deg
box of 10 m wind is **~1 KB instead of ~30 MB**. The AWS Open Data mirror
(`noaa-gfs-bdp-pds.s3.amazonaws.com`) also works but serves full global files.

### Why it matters

ERA5 is right for history but sits in a CDS queue. Measured on the same box:

| window | source selected | wall time |
|---|---|---|
| recent (last 24 h), `forcing=auto` | **GFS** | **21 s** |
| 2020-08-10, `forcing=auto` | ERA5 (GFS refused) | 123 s |

~6x faster for live operations, with an explicit refusal rather than a 404:
> "GFS not usable: GFS on NOMADS retains ~10 days; window starts 2020-08-09
> which is before the 2026-09-03 cutoff — use ERA5."

`supports_window()` makes that decision *before* any download, and
`forcing: auto | era5 | gfs` lets the operator pin it. GFS validity was
sanity-checked physically: easterly trades at Mauritius (u10 ≈ −7 m/s).

### Two implementation traps

- **cfgrib cannot decode from `BytesIO`** — eccodes needs a real path and
  returns an *empty* dataset rather than raising. Spool each response to a
  temp file and `.load()` before deleting it.
- **Don't double-apply the 6 h publication lag.** `latest_cycle()` already
  steps back one cycle; passing an already-lagged start time pushes you into
  an older cycle whose files may since have been rotated out.

### Honest limitation

GFS is a **live-operations** source only. NOMADS retains ~10 days, so every
historical case study (including Wakashio 2020) must use ERA5. The resolver
enforces this rather than silently degrading.
