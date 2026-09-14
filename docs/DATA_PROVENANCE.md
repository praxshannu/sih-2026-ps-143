# SENTINEL — Data Provenance

Every datum this platform shows carries one of four states, and the difference
between them is the difference between evidence and decoration:

| State | Meaning | How the UI shows it |
|---|---|---|
| **REAL** | Fetched from the named upstream, checksummed where possible | `■ REAL` glyph + source named in the strip |
| **SYNTHETIC** | Generated, fenced, and labelled | `▲ SYNTHETIC` + warning banner + dashed rendering |
| **UNAVAILABLE** | The source exists but has no coverage here | `□ UNAVAILABLE` + the reason and the missing source named |
| **FAILED** | An upstream error occurred | `✕ FAILED` + the machine-readable reason code |

The single rule: **a number without provenance and a confidence interval is not
a result.** There is no code path that renders a measurement without also
rendering where it came from.

---

## 1. The four synthetic-data gates

All four default to **off**, and off means **fail closed**. An unset or
misspelled value is `False` — a typo must never open a gate.

| Gate | Permits | Default |
|---|---|---|
| `ALLOW_SYNTHETIC_FORCING` | CMEMS/ERA5 may fall back to a labelled constant field | `false` |
| `SENTINEL_DEMO_MODE` **and** `ALLOW_SYNTHETIC_AIS` | Synthetic vessel tracks, inside permitted boxes only | `false` |
| `ALLOW_SYNTHETIC_TRAINING` | `scripts/synthetic_train.py` may write fixtures | `false` |

`services/ingest/app/provenance.py` is the single place this rule is
implemented, so the sources cannot drift apart. `env_flag()` accepts
`true/TRUE/1/yes/on` and rejects everything else, including `""`, `0`, `false`,
`no`, `off`, `maybe` and `troo`.

### Why this is not paranoia

`docs/VERIFICATION.md` §9.2 records the failure this prevents, observed on this
machine: starting uvicorn by hand dropped every environment key, an ERA5 pull
silently degraded to synthetic wind, and **the run succeeded and returned
invented numbers**. The API returned HTTP 200 with a plausible origin ellipse
computed from wind that did not exist.

A crash is a good outcome. That was the bad one.

---

## 2. Source-by-source

### 2.1 SAR imagery — REAL

| | |
|---|---|
| Upstream | Copernicus Data Space Ecosystem, **Sentinel Hub Process API** |
| Endpoint | `POST https://sh.dataspace.copernicus.eu/api/v1/process` |
| Auth | CDSE OAuth client credentials (`CDSE_CLIENT_ID`/`_SECRET`), token TTL 1800 s |
| Product | Calibrated σ⁰ GeoTIFF, EPSG:4326, float32, VV + VH (+ dataMask) |
| Checksum | SHA-256 recorded per scene in `data/sar/<scene>.json` |

**Why the Process API and not a download.** The OData download endpoint returns
`401 DAT-ZIP-609 "Token audience not allowed"` — the client-credentials token
carries no `aud` claim. The Process API is the same engine the Copernicus
Browser runs on, and it returns a 44 MB calibrated GeoTIFF instead of a
1.7–2.0 GB `.SAFE` archive requiring SNAP (Java, 8 GB RAM) to preprocess. That
substitution is what makes this feasible on a 16 GB M1/M2 laptop.

**A correction to the original brief.** The demo product ID
`S1A_IW_GRDH_1SDV_20200807T024930_…` **does not exist** — the catalogue returns
0 matches, and there is no Sentinel-1 IW acquisition over Mauritius on
2020-08-07 at all. The real passes over the spill zone are 22 Jul, 29 Jul,
3 Aug, 10 Aug, 15, 16, 21 and 22 Aug 2020. Two requested windows returned no
data and were **omitted rather than filled** — `NoDataError` is raised and the
scene is skipped.

**Scenes on disk (6, real):**

| Scene | Date | Size | Notes |
|---|---|---|---|
| `wakashio_20200729_early` | 2020-07-29 | 44.0 MB | pre-spill reference |
| `wakashio_20200810_peak` | 2020-08-10 | 44.4 MB | **the headline detection: 3.347 km², −9.79 dB** |
| `wakashio_20200816_late` | 2020-08-16 | 44.4 MB | post-grounding |
| `wakashio_20200822_recovery` | 2020-08-22 | 44.4 MB | recovery |
| `mumbai_20260913_recent` | 2026-09-13 | 8.7 MB | modern pass, S1D |
| `browser_test_wakashio_…` | 2026-09-13 | 2.8 MB | archive-browser ingest test |

AOI for the Wakashio series: `57.60–58.20 E, 21.00–20.40 S` — open ocean SE of
Mauritius. Chosen because the spill drifted offshore, which keeps land out of
the σ⁰ statistics; land raises the p75 baseline and would suppress real
detections.

### 2.2 Ocean currents — REAL

| | |
|---|---|
| Upstream | Copernicus Marine Service, `cmems_mod_glo_phy_my_0.083deg_P1D-m` (GLORYS12) |
| Auth | `COPERNICUSMARINE_SERVICE_USERNAME` / `_PASSWORD` |
| Access | `copernicusmarine.open_dataset()`, **not** `subset().load()` |
| Depth | `[0.4941, 1.0]` m — the shallowest grid coordinate is 0.49402499… |
| Cadence | **daily** — a 24 h window may hold a single 00:00 snapshot |

Three traps, each of which silently breaks a drift run:

1. In `copernicusmarine` 2.x, `subset()` returns a `ResponseSubset` — a
   download *receipt* — not an xarray object. `.load()` raises `AttributeError`.
   The pre-existing `_fetch_cmems` used this form and had therefore **never
   worked**.
2. Depth bounds are validated strictly even with
   `coordinates_selection_method='nearest'`; an exact `0.494` is below the grid.
3. Because GLORYS12 is daily, `start_time == end_time` for a short window, and
   OpenDrift then reports **every** variable missing — including wind, from a
   perfectly healthy ERA5 reader. It reads exactly like an ERA5 outage. Pad the
   request ±1 day.

Also: drop the singleton depth axis (`isel(depth=0, drop=True)`). The CF reader
mis-maps a 3-D current field.

### 2.3 Winds — REAL (ERA5) or REAL (GFS), never guessed

| | ERA5 | GFS |
|---|---|---|
| Upstream | CDS `reanalysis-era5-single-levels` | NOMADS `filter_gfs_0p25.pl` |
| Retention | 1940 → present | **~10 days** |
| Use | all historical cases | live operations |
| Measured wall time | 123 s (2020-08-10) | **21 s** (recent window) |

**Wind is mandatory, not optional.** OpenOil refuses to run on currents alone
(`Readers must be added for ['x_wind','y_wind']`). A missing wind field is a
typed failure, not a silent substitution.

`GFS_OPENDAP_URL` is **dead** — NOAA retired the OPeNDAP service (SCN 25-81) and
it now returns an HTML page. The replacement is the NOMADS server-side
subsetting filter, which returns ~1 KB for a 5° box instead of ~30 MB.
`supports_window()` refuses a historical window **before** any download, with a
message naming the cutoff, rather than 404-ing.

**The off-by-one that impersonates an outage.** The ERA5 reader's coverage once
ended exactly at the detection time. A backward run *starts* at that instant,
and OpenDrift's time interpolation reaches for `index + 1` — one past the end.
The `IndexError` is swallowed and resurfaces as
`Missing variables: ['x_wind', 'y_wind', …]`. Fix: `FORCING_PAD_HOURS = 3`.
This only appeared once CMEMS was added, because the ERA5-only path happened to
keep one extra hour of slack.

### 2.4 Vessels — REAL where receivers exist, UNAVAILABLE where they do not

**AISStream is a terrestrial receiver network.** Measured on this machine:

```
global bbox              → 48 real messages in 13 s      ✅
Mumbai 68–76E / 15–22N   →  0 messages in 25 s
Indian Ocean AOI         →  0 messages in 40 s
```

**No free historical AIS covers the 2020 Indian Ocean.** MarineCadastre is
US-waters only; the Danish Maritime Authority archive is unreachable; satellite
AIS (Spire / ORBCOMM) is a paid subscription.

So the choice is a labelled synthetic layer or no layer. SENTINEL takes the
labelled one, fenced geographically:

```
NO_REAL_COVERAGE_BOXES       = (35, -30, 110, 35)   # Indian Ocean / Arabian Sea / Bay of Bengal
KNOWN_COASTAL_COVERAGE_BOXES = Mumbai, Chennai, Hormuz, Gulf of Aden,
                               Colombo, Karachi, Haldia, Singapore
```

An AOI gets synthetic data **only** if it overlaps the no-coverage box **and
does not** overlap a coastal carve-out. That ordering matters: real coverage
always wins where receivers exist, so synthetic tracks can never mask a real
feed. Verified behaviour:

| AOI | `provenance` | Vessels |
|---|---|---|
| Wakashio zone `57.6,-21.0,58.2,-20.4` | `no_real_coverage` | 0 |
| Mumbai `72.6,18.8,73.1,19.3` | `live_terrestrial` | real feed used |
| North Sea `3,52,5,54` | `live_terrestrial` | real feed used |

Three notification layers, no quiet path: a **pre-flight** call the moment the
AOI changes; **on fetch**, every response carries `provenance`, `is_synthetic`,
`notice` and `disclaimer`; **on the map**, synthetic tracks render dashed amber
and real tracks solid, so a screenshot is self-labelling.

**`data/ais_demo.csv` is deleted.** It was synthetic — 8 MMSIs × exactly 200
rows at a perfect 600.0 s cadence (σ = 0.0). Real AIS is never uniform, so a
single glance at the timestamps identified it. Shipping it would have
undermined every other provenance claim in the product.

---

## 3. The AOI contract — named axes, never a positional string

Every archive and AIS endpoint takes `min_lon, min_lat, max_lon, max_lat`. The
comma-separated `west,south,east,north` form survives only as a **deprecated
alias** that logs a warning. A partial set is `422`, never guessed at.

The reason is not style. A lat-first string is four numbers that are all legal
in either slot, so **no validator can detect the swap**. The AOI is silently
relocated, the AIS real/synthetic verdict flips with it, and the analyst sees a
confident, plausible, wrong answer. Named axes cannot be misordered.

Related: the gateway strips `None` query parameters before proxying. httpx
renders `None` as a bare `key=`, which the upstream then fails to parse as a
float.

---

## 4. What the persisted case record keeps

`services/api/app/case_store.py` writes one record per processed scene holding
everything needed to re-derive or dispute the result: scene metadata, source and
checksum, detection, polygon, drift, **forcing provenance**, **AIS provenance**,
suspect scores, model version, confidence and warnings.

The backend that served the write is recorded on the record itself. The file
backend is a real durable store, not a cache; the Postgres path mirrors
`database/init/07_case_persistence.sql` and has not been run against a live
server in this environment. The distinction is always reported, never inferred.

---

## 5. Verifying a claim yourself

```bash
cd /Users/praxsmac/projects/claude1/sentinel
PY=/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/python

# Is this scene real, and what is its checksum?
cat data/sar/wakashio_20200810_peak.json | head -40

# Does a detection result agree with the scene it claims to describe?
$PY -m json.tool data/sar/wakashio_20200810_peak.detection.json | head -60

# Is AIS real or synthetic for an AOI? (call BEFORE fetching)
curl -s "http://localhost:8000/api/v1/archive/ais/coverage?min_lon=57.6&min_lat=-21.0&max_lon=58.2&max_lat=-20.4" | $PY -m json.tool

# Which backend will persist my case, and is it the database?
curl -s http://localhost:8000/api/v1/scenes/store/health | $PY -m json.tool

# Are any synthetic gates open right now?
grep -E "ALLOW_SYNTHETIC|SENTINEL_DEMO_MODE" .env || echo "all gates unset = all closed"
```
