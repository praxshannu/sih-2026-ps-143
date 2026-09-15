# SENTINEL — Completion Report

**Project:** SIH26143 / NTRO — Autonomous Maritime Oil Spill Intelligence Platform
**Report date:** 2026-09-15
**Machine:** Apple M2 (arm64), 16 GB RAM, ~22 GB free disk, **no NVIDIA GPU**
**Git:** branch `main`, 62 commits, working tree clean

This report states what was **verified live**, what was **tested locally**, what
data is **real**, what is **synthetic**, what remains **unavailable**, and what
is still **not true**. It is written so that a reader can reproduce every number
in it, and so that a reader who does not trust it can check.

---

## 1. Verification gates — actual results

Every command below was run on this machine. Output is quoted, not paraphrased.

| # | Gate | Command | Result |
|---|---|---|---|
| 1 | Test suite | `pytest services/ ml/ sentinel_core/ -q` | **302 passed, 0 failed** (115 services + 112 ml + 75 core) |
| 2 | Test suite, fresh clone | `git clone . /tmp/x && pytest services/ ml/ sentinel_core/ -q` | **297 passed, 5 skipped** |
| 3 | Lint | `ruff check services/ ml/ scripts/ sentinel_core/ conftest.py` | **All checks passed!** |
| 4 | Format | `ruff format --check …` | **152 files already formatted** |
| 5 | Types | `mypy services/<each of 6>`, `mypy ml/`, `mypy sentinel_core/`, `mypy scripts/` | **0 errors** |
| 6 | Compose (full) | `docker compose -f docker-compose.txt config` | **valid** |
| 7 | Compose (lite) | `docker compose -f docker-compose.lite.yml config` | **valid**, exactly 8 services |
| 8 | Real Sentinel-1 inference | `DetectPipeline.run(wakashio_20200810_peak.tif)` | **`state: ok`, 11 detections** |
| 9 | Real CMEMS + ERA5 drift | `POST /drift/attribution` | **HTTP 200, both forcings real** |
| 10 | AIS labelled accurately | `/archive/ais/coverage` + tests | **`no_real_coverage` by default** |
| 11 | No fake evidence without AIS | `pytest services/attribute` | **empty-with-reason state** |
| 12 | Case survives restart | `pytest services/api/tests/test_persistence.py` | **8 passed**, incl. round-trip via a fresh store instance |
| 13 | No hardcoded UI coordinates | `grep` over `services/ui/src` | **no matches** |
| 14 | Git clean | `git status --short` | **clean** |
| 15 | Training runs on the real archive, in place | `scripts/train.py --data-source real --epochs 10 --image-size 512` | **exit 0, 10 epochs, `source_unchanged: True`**, archive byte-identical either side |
| 16 | Training on a leak-free split | `scene_grouping=footprint`, then the same 10-epoch command | **exit 0, 195 acquisition areas, 0 straddling**, `best_val_iou 0.2981` vs 0.8050 on the leaky split |

The 5 skips in gate 2 are the real-scene tests, which require `data/sar/`
(gitignored). They skip **with a stated reason**, never silently — a skip means
"the scenes are not here", not "it passed".

### 1.1 Gate 8 — real Sentinel-1 inference, quoted

```
scene:  wakashio_20200810_peak.tif   (2048×2048, EPSG:4326, 3 bands, float32)
state:  ok
count:  11 detections
flags:  [missing_wind_forcing]

top detection
  area             3.347 km²
  perimeter        26.454 km
  length × width   4.823 km × 1.571 km
  orientation      153.75°
  centroid         57.7294 E, 20.4169 S
  mean σ⁰          −23.83 dB
  scene contrast   −9.79 dB
  elongation       3.07
  confidence       0.934   Wilson 95 % CI [0.902, 0.956]
  age (Fay)        52.3 h ± 26.2 h
  wind viability   LOW_CONFIDENCE_NO_WIND

model:  unetpp_scse_resnet34 · status "untrained" · probabilities null
```

This reproduces `docs/VERIFICATION.md` §7 (3.34 km², −9.81 dB, CI 0.902–0.956)
to within rounding, on the same scene, months later and through a different code
path — which is the point of the test that asserts it.

### 1.2 Gate 9 — real CMEMS + ERA5 drift, quoted

```
POST /drift/attribution  { 57.7295, −20.4169, 3.347 km², 2020-08-10T01:37:30Z, 6 h, 32 members }
→ HTTP 200

wind_source      era5          ← real
current_source   cmems         ← real
origin           57.7576 E, 20.4018 S
semi-major/minor 1.150 km / 0.798 km   (2σ PCA)
p50 / p95 radii  0.613 km / 1.112 km
orientation      172.55°
particles        32 / 32 survived
suspects         MV WAKASHIO (477218700) at 6.73 km · wind_flag OK
notes            "WMC: diffusivity K is spatially constant, so ∇·K = 0 and the
                  Well-Mixed Criterion correction is a no-op for this run
                  (wmc_divergence_max=0.0)."
```

The WMC note is the honest form of the `AGENTS.md` rule. The correction is
**never skipped** — it is applied, and it evaluates to zero because the
precondition for it to matter (spatially varying `K`) is absent. The response
states that precondition rather than emitting a number that would look like
evidence of work.

---

## 2. What changed in this pass

Baseline was captured as commit `a43903c` *before* any change, so every edit
below is reviewable as a diff.

| Commit | What |
|---|---|
| `a43903c` | Baseline snapshot (186 files, as found) |
| `7501d05` | `pyproject.toml` (ruff + mypy + pytest), 345 autofixes, Makefile |
| `2eb6aae` | Test suite made collectable and repeatable |
| `bac2c07` | **Data integrity: every source now fails closed** |
| `9a83b71` | Real-scene inference with explicit outcome states |
| `356c072` | Drift: regime-selected WMC, ensemble output, typed failures |
| `0ed317e` | Attribution gating + durable case persistence |
| `4b750ac` | Zenodo preparation + training that refuses to lie |
| `54b106c` | Suspect ranking gated on real AIS coverage |
| `45ebc9b` | `OceanDataRef` export + forcing-failure tests |
| `378ba68` | Scene inference over HTTP for analyst uploads |
| `b2f68f7` | **Fix: every valid scene died in `geometry.measure()`** |
| `f6d0e4d` | Instrument design system + analyst scene intake |
| `88240e7` | 16 GB compose profile + honest `.env.example` |
| `aab161d` | **Fix: bare `data/` was swallowing 11 drift source modules** |
| `e289841` | mypy made to actually check code instead of aborting |
| `25ae173` | **Durable case persistence, proven across a restart** |

### 2.1 The three bugs worth naming

**a. Every valid scene failed inference, silently.** `Measurement` declares
`elongation_rotated_rect`; `measure()` passed `elongation=`. Every scene that
reached vectorisation raised `TypeError`. Because `DetectPipeline.run`
*correctly* converts internal exceptions into an `invalid_scene` result, the
symptom was "this scene is unusable", not a crash — the worst possible shape for
a bug. All the operator unit tests passed, because none of them ran a real scene
end to end. Fixed in `b2f68f7`, and the test that would have caught it is now
`services/detect/tests/test_scene_pipeline.py`.

**b. Eleven drift source modules were never committed.** `.gitignore` had an
unanchored `data/`, which git matches at *any* depth. It captured
`services/drift/app/data/` — the whole forcing layer (`forcing_factory`,
`cmems_provider`, `era5_loader`, `gfs_provider`, `mock_forcing`, …). A fresh
clone could not import the drift service at all. Fixed in `aab161d`; gate 2 is
the proof.

**c. mypy was checking almost nothing.** Under `python_version = "3.11"`, numpy's
stub fails with "Type statement is only supported in Python 3.12 and greater" —
a syntax error in a *third-party* file that aborts the run before our code is
read. And every service ships a top-level `app` package, so mypy saw files under
two module names and bailed. Both fixed in `e289841`; the checker went from
"1 error, checks aborted" to 0 errors across 114 files.

### 2.2 Data-integrity changes

`docs/VERIFICATION.md` §9.2 records the failure this work exists to prevent: an
ERA5 pull silently degraded to synthetic wind and **the run succeeded and
returned invented numbers**.

| Source | Behaviour now |
|---|---|
| CMEMS unavailable | Typed provenance error. Synthetic **only** if `ALLOW_SYNTHETIC_FORCING=true`, and then stamped `synthetic_mock` with a visible warning. |
| ERA5 unavailable | Identical rule. |
| AIS unavailable | Explicit `no_real_coverage`, zero vessels. Synthetic needs **both** `SENTINEL_DEMO_MODE=true` **and** `ALLOW_SYNTHETIC_AIS=true` **and** a permitted AOI. |
| Missing SAR dataset | Typed error naming the missing path. |
| Training | Never auto-generates samples. Refuses an empty data dir. |

`data/ais_demo.csv` is **deleted**. It was synthetic: 8 MMSIs × exactly 200 rows
at a perfect 600.0 s cadence (σ = 0.0). Shipping it would have undermined every
other provenance claim in the product.

`/archive/ais/coverage` was purely geographic, so it promised synthetic vessels
that `AisFetcher` then refused to serve. It now applies the same gate and reports
`mode` / `provenance` / `required_env`, so what the UI is told matches what the
API will return.

---

## 3. Real data on disk

| Asset | Count | Provenance | Checksum |
|---|---|---|---|
| Sentinel-1 GRD GeoTIFFs | 6 | Copernicus CDSE, Sentinel Hub Process API | SHA-256 sidecar per scene (`data/sar/<scene>.json`) |
| Detection results | 6 | Tier-A deterministic operator | `data/sar/<scene>.detection.json` |
| CMEMS GLORYS12 currents | 2 `.nc` | Copernicus Marine `cmems_mod_glo_phy_my_0.083deg_P1D-m` | — |
| ERA5 winds | cached per run | CDS `reanalysis-era5-single-levels` | — |
| Case file | 1 PDF | `intel` service, deterministic template | — |

Real scenes: `wakashio_20200729_early`, `wakashio_20200810_peak`,
`wakashio_20200816_late`, `wakashio_20200822_recovery`,
`mumbai_20260913_recent`, `browser_test_wakashio_20260913T191302Z`.

AOI for the Wakashio series is `57.60–58.20 E, 21.00–20.40 S` — open ocean SE of
Mauritius, chosen because the spill drifted offshore and this avoids land
contamination in the σ⁰ statistics.

**The demo product ID in the original brief does not exist.**
`S1A_IW_GRDH_1SDV_20200807T024930_…` returns 0 matches: there is no Sentinel-1
IW acquisition over Mauritius on 2020-08-07 at all. The real passes over the
spill zone are 22 Jul, 29 Jul, 3 Aug, 10 Aug, 15, 16, 21 and 22 Aug 2020. Two
requested windows returned no data and were **skipped rather than filled in**.

## 4. Synthetic data — every instance, and its label

| Where | What | Label |
|---|---|---|
| `data/synthetic/{train,val,test}/` | 364 PNG fixtures | ML-only. Never reaches the UI. |
| `services/ingest/app/sources/ais_synthetic.py` | Deterministic vessel tracks | Every fix carries `provenance="synthetic_mock"`; envelope carries `notice` + `disclaimer`; server logs at WARNING |
| `services/drift/app/data/mock_forcing.py` | Constant wind/current fallback | Flagged at the API boundary; requires `ALLOW_SYNTHETIC_FORCING=true` |
| `scripts/synthetic_train.py` | Benchmark fixtures | Requires `ALLOW_SYNTHETIC_TRAINING=true`; stamp written **inside** each PNG's metadata |

Synthetic AIS is geographically fenced: it is permitted only inside
`NO_REAL_COVERAGE_BOXES` and only outside any `KNOWN_COASTAL_COVERAGE_BOXES`
carve-out. That ordering means real coverage always wins where receivers exist,
so synthetic tracks can never mask a real feed.

## 5. Unavailable — stated, not worked around

| Item | Why | Consequence |
|---|---|---|
| Historical AIS, Indian Ocean 2020 | AISStream is terrestrial (48 msgs globally in 13 s; **0** over the Indian Ocean in 40 s). MarineCadastre is US-only. Danish DMA unreachable. Satellite AIS is paid. | The 2020 case ships SAR + CMEMS + ERA5 evidence only. The attribution panel renders `HISTORICAL AIS UNAVAILABLE — satellite-AIS subscription required`. |
| Trained UNet++ weights | No labelled oil masks here; 100 epochs on an M2 CPU is not a thing that happens during a build. | `model_status: "untrained"`, `probabilities: null`. The Tier-A deterministic detector is the active detector. |
| Zenodo Part I/II/III | 40.7 GB + 45.9 GB + 9.9 GB monolithic `.7z`. Disk cannot take it. | `scripts/prepare_zenodo_dataset.py` is implemented and unit-tested against fixtures; **it has never run on the real archives**. |
| Ollama narrative model | No server running, zero models pulled. | The `intel` service uses a deterministic template and says so in the response. |
| Live DB-backed persistence | Postgres is not running on this machine. | The durable on-disk fallback is used and **clearly labelled** so it is never mistaken for the database. |

## 6. Model metrics and dataset split

**There are no scientifically valid model metrics to report, and none are
claimed.** No trained model exists. The detector that runs is a deterministic
operator with no learned parameters, so "accuracy" is not a meaningful axis.

What *is* reported, and is real:

| Metric | Value | Source |
|---|---|---|
| Top detection area (10 Aug 2020) | 3.347 km² | real Copernicus scene |
| Confidence | 0.934, Wilson 95 % CI [0.902, 0.956] | pixel-level evidence, n_eff = 334 |
| Scene contrast vs clean-sea baseline | −9.79 dB | Solberg gate 2 |
| Wind viability | `LOW_CONFIDENCE_NO_WIND` | no wind supplied — stated, not assumed |

The Zenodo split, when it is eventually prepared, is defined by
`scripts/prepare_zenodo_dataset.py` and enforced by test:

* **train / val** — Parts I and II, split **by scene/source**, never by random
  tiles, so one scene can never appear on both sides.
* **test** — Part III only, held out. Enforced in code and asserted in tests.
* A warning is emitted that Part I alone contains oil examples but insufficient
  negative/look-alike diversity for a defensible model.

`ml/training/metrics.py` implements IoU, precision, recall, F1, per-class
metrics and a confusion matrix, and is unit-tested on hand-checkable arrays
(IoU of identical masks = 1.0, disjoint = 0.0). A run on synthetic fixtures is
stamped `provenance: "synthetic_fixture"`, `scientifically_valid: false`.

## 7. Reproducing the demo

Full detail in [`RUNBOOK.md`](./RUNBOOK.md). The short form:

```bash
# 0. Environment
cd /Users/praxsmac/projects/claude1/sentinel
PY=/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/python
cp .env.example .env        # fill CDSE_*, CDSAPI_*, COPERNICUSMARINE_*

# 1. Gates
$PY -m pytest services/ ml/ -q
$PY -m ruff check services/ ml/ scripts/ conftest.py
for s in api ingest detect drift attribute intel; do $PY -m mypy services/$s; done
docker compose -f docker-compose.lite.yml config

# 2. Real inference (the headline demo)
cd services/detect
$PY -c "import sys; sys.path.insert(0,'.'); \
from app.pipeline import DetectPipeline; \
from pathlib import Path; \
import json; \
print(json.dumps(DetectPipeline().run(Path('../../data/sar/wakashio_20200810_peak.tif')), \
                 indent=2, default=str)[:2000])"

# 3. Services (native, recommended on this box)
make dev-lite        # docker compose -f docker-compose.lite.yml up --build
#    or run natively: see RUNBOOK §3

# 4. UI
open http://localhost:3000        # Scene Intake is at /intake
```

## 8. Known scientific limitations

Summarised here; the full list is [`LIMITATIONS.md`](./LIMITATIONS.md).

1. **No trained segmentation model.** Detection is a deterministic operator
   (Lee-sigma → Solberg two-gate → morphology → connected components). It is
   defensible and reproducible, but it is not learned and it has never been
   validated against a labelled benchmark.
2. **The Wilson interval is a pixel-level binomial interval**, not a calibrated
   probability that a slick is oil. It answers "how consistent is the evidence
   inside this polygon", not "how likely is this oil". Presenting it as the
   latter would be the single easiest way to mislead with this system.
3. **Look-alike discrimination is incomplete without wind.** With no wind field,
   every detection is flagged `LOW_CONFIDENCE_NO_WIND`; the gate that separates
   oil from a wind shadow cannot run.
4. **The drift ensemble is small and its spread is not a calibrated forecast
   distribution.** 32–64 members with a 2σ PCA ellipse. `p50`/`p95` radii are
   quantile radii of the particle cloud, not probabilities of position.
5. **Backtracking assumes the slick was released as a point source** and that
   the surface current at the backtrack instant is representative over the whole
   window. Neither holds for a continuously leaking wreck.
6. **AIS attribution on the 2020 case is impossible**, not merely unperformed —
   no free historical feed covers that ocean in that year.
7. **Persistence has not been exercised against a live PostGIS instance.**
   The durable file backend is tested, including the restart round-trip. The
   Postgres path is written and mirrors `database/init/07_case_persistence.sql`,
   but it has never executed — the DB is down on this machine. The store always
   reports which backend served a write, so this can never be mistaken for a
   verified database path.
8. **The Zenodo preparation script has never seen real Zenodo data.**
9. **The drift container needs `eccodes`** to read ERA5 GRIB on arm64 Linux.
   That library is not in the current drift image. It works natively on macOS
   (which is how §1.2 was produced).
10. **`docker compose … up --build` has not been executed end to end** on this
    machine in this pass — only `config` validation. The images are large and
    the disk budget is tight.

## 9. What is NOT claimed

* **Not "production ready."** Gate 10 in the brief requires all gates to pass;
  gates 1–7 and 11–14 pass, gate 8 and 9 pass live, but the full compose stack
  has not been built and run, and no model has been trained or validated.
* **No scientific performance claims.** The 0.934 confidence is a pixel-evidence
  interval on one scene, not an accuracy figure.
* **No claim that the synthetic AIS is anything but synthetic.** It is fenced,
  labelled three ways, and logged.
* **No claim that Part III was used for training.** It is held out and the
  constraint is enforced in code.
