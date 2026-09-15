# SENTINEL — Limitations

This file exists because a system that reports its own limitations is more
trustworthy than one that does not, and because every entry below is something a
reviewer would otherwise discover the hard way.

Entries are grouped by whether they are **scientific** (the answer may be wrong),
**operational** (the answer is right but the system cannot deliver it here), or
**unverified** (nobody has checked yet).

---

## A. Scientific limitations

### A1. No trained segmentation model — the detector is deterministic

Detection is a hand-tuned operator (Lee-sigma → Solberg two-gate threshold →
morphology → connected components), not a learned model. It has no weights, so
"accuracy" is not a meaningful axis for it.

Consequence: it is **reproducible and explainable**, and it has **never been
validated against a labelled benchmark**. It cannot generalise the way a trained
model would, and its thresholds were calibrated on the Wakashio scenes and the
scene-statistics envelope in `pipeline.py`.

See `MODEL_CARD.md` §1.

### A2. The confidence interval is not P(oil)

The Wilson 95 % CI answers *"how consistent is the dark-pixel evidence inside
this polygon?"* It does **not** answer *"how likely is this oil?"*

It has never been calibrated against ground truth, because no labelled ground
truth exists in this environment. `n_effective` is a pixel count, not a sample of
independent spill observations — adjacent pixels are correlated, so the interval
is narrower than a truly independent sample of the same size would justify.

**This is the most dangerous number in the system**, precisely because it looks
like a probability. Any downstream report must restate what it measures.

### A3. Look-alike discrimination is incomplete without wind

Slicks are separable from look-alikes only in roughly 2–10 m/s wind. Outside that
band — and with no wind field at all — rain cells, low-wind patches, biogenic
films and grease ice can pass the thresholds.

Without wind every detection carries `wind_viability: LOW_CONFIDENCE_NO_WIND` and
the response carries `missing_wind_forcing`. That is a **caveat, not a
correction**: the detections are still emitted.

### A4. The drift ensemble is small, and its spread is not a forecast distribution

32–64 members. `semi_major_km` / `semi_minor_km` are 2σ axes of a PCA fit that
**assumes a Gaussian particle cloud**. `p50_radius_km` / `p95_radius_km` are
quantile radii from the centre and make no Gaussian assumption — which is why
both are reported. Neither is a probability of position.

### A5. Backtracking assumes a point-source release and a stationary current field

The backward integration treats the slick as released instantaneously at one
location, and the surface current at the backtrack instant as representative
over the whole window. Neither holds for a wreck leaking continuously over days
— which is exactly the Wakashio case. The inferred origin is therefore a
*plausible* origin, not the origin.

### A6. The WMC ∇·K correction is usually a no-op here, and that is reported

With spatially constant `K`, `∇·K = 0` and the Well-Mixed Criterion correction
evaluates to zero. `AGENTS.md` forbids skipping it, and it is not skipped — it is
applied and the precondition under which it is a no-op is **stated in the
response**. A number would be worse than the explanation.

### A7. Attribution weights are chosen, not fitted

`ATTRIBUTION_FUZZY_WEIGHT=0.6` / `ATTRIBUTION_XGB_WEIGHT=0.4` were picked by
hand. No labelled set of confirmed spill attributions exists to fit them
against, so `K_ij` is a **structured opinion with a stated interval**, not a
calibrated probability.

### A8. Fay age estimates are regime-dependent and wide

`age_hours_fay` on the 10 Aug peak is **52.3 h ± 26.2 h** — a ±50 % band. It
assumes Fay-type spreading from a point release and is invalid for a slick that
has already broken into wind-driven streaks.

### A9. The 2020 case cannot be attributed by AIS at all

Not "has not been" — **cannot be**. AISStream is terrestrial (0 messages over the
Indian Ocean in 40 s), MarineCadastre is US-only, the Danish archive is
unreachable, and satellite AIS is paid. There is no free historical feed for
that ocean in that year.

The system renders `HISTORICAL AIS UNAVAILABLE — satellite-AIS subscription
required` and lists the source that would fill the gap.

---

### A10. The synthetic VV tail is 4.9 dB short, and the check cannot settle it

The synthetic set matches the real archive on 27 of 28 distribution checks
(`data/synthetic/match_report.json`). The one that misses is **VV p99.9**: the
real archive's brightest 0.1% of pixels sits at −16.79 dB, the synthetic set's
at −21.67 dB.

This is measured, not guessed, and the mechanism is understood. For a two-class
population where bright pixels are a fraction `f` of a scene, the scene's p99.9
is the bright pool's quantile `q = 1 − 0.001/f` — about the pool's p78 for VV.
The generator draws its bright coverage from per-scene records measured on a
32-scene sample, and its estimate (`f = 0.00368`) is 20% below the real mean
(`f = 0.00459`). Applied through `q`, a 20% coverage shortfall costs ~4.5 dB on a
population that rises steeply there, which is the whole of the 4.9 dB gap.

The reason this is listed as a limitation rather than a bug: **a 32-scene mean of
VV bright coverage has a 5–95% bootstrap band of 0.0027–0.0067** (±46%). The
synthetic value sits inside it. A check whose estimator carries ±46% sampling
noise cannot support a ±2 dB tolerance on a statistic with a ~60 dB/dB slope, so
the honest reading is "the sample is too small to decide", not "the generator is
biased". The per-scene bright statistics do match closely (VV p99 mean: real
−10.54, synthetic −10.21).

Two things would settle it: profile all 1200 scenes instead of 32
(`scripts/generate_synthetic.py --sample 1200`), or widen the tolerance for
pooled tail statistics and say why. Until one of those is done, treat the VV
p99.9 comparison as unresolved.

---

## B. Operational limitations

### B1. The full compose stack has not been built or run

`docker compose -f docker-compose.txt config` **validates**, and
`docker-compose.lite.yml` is verified to contain exactly 8 services. Neither
stack has been brought up end to end in this pass — the images are large and the
disk budget is tight (22 GB free).

Everything reported as "verified live" was run natively.

### B2. The drift container needs `eccodes`, which its image lacks

`cfgrib` requires the eccodes system library to read ERA5 GRIB. On arm64 Linux
that means adding `libeccodes0` to the drift Dockerfile. It is **not** there.

The verified drift numbers were produced **natively on macOS**, where Homebrew's
eccodes 2.48.0 works (`ECCODES_DIR=/opt/homebrew/Cellar/eccodes/2.48.0`).

### B3. Zenodo preparation has never seen real Zenodo data

`scripts/prepare_zenodo_dataset.py` is implemented, with the pairing, split-by-
scene, Part III hold-out and manifest logic all unit-tested against fixtures.
It has never run on the real archives: Part I is 40.7 GB, Part II 45.9 GB, Part
III 9.9 GB, as monolithic `.7z` files with no safe HTTP-only partial extraction.

**Do not report Zenodo metrics.** There are none.

### B4. Ollama is not running

No server, zero models. The `intel` service falls back to a deterministic
template and labels the response accordingly — it does not invent a narrative and
present it as model output.

### B5. Postgres is down, so the database path is unexercised

`services/api/app/case_store.py` prefers Postgres and falls back to durable JSON.
The **file backend is tested**, including the restart round-trip. The Postgres
path is written and mirrors `database/init/07_case_persistence.sql`, but has
never executed.

The store always reports which backend served a write, so this can never be
mistaken for a verified database path.

### B6. The gateway's legacy case CRUD is still in-memory

`services/api/app/routers/cases.py` keeps cases in a module-level `dict`. The
durable store was added **additively** at `/api/v1/scenes/*` rather than
rewriting that router in the same change, so a reviewer can tell which behaviour
changed.

Consequence: cases created through the old `/api/v1/cases` endpoints still do not
survive a restart. Only `/api/v1/scenes/persist` is durable.

### B7. GFS is live-operations only

NOMADS retains ~10 days, so every historical case study must use ERA5. The
resolver **refuses** an out-of-range window with a message naming the cutoff,
rather than silently degrading.

### B8. `AIS_BASE_URL` is not a REST endpoint

`https://services.aisstream.io` is WebSocket-only. Anything treating it as REST
will fail. The key is also empty in `.env`.

### B9. Disk headroom is real but finite

~22 GB free. The Zenodo archives cannot be staged here. A Docker build of the
full stack would consume several GB more.

### B10. `Dockerfile.train` has never been built

Docker is not installed on the machine it was written on, so the image is
unbuilt. Every COPY source was checked by hand and the dependency set is the one
the trainer was verified against, but "the Dockerfile exists" is not "the image
works". The file says so at the top, where the next person will read it.

Related and worth knowing before relying on it: the encoder weights come from
the Hugging Face hub, so a machine with no network cannot train with the default
`encoder_weights=imagenet`. The image bakes them at build time for that reason.
The trainer detects the failure and names the workaround
(`--encoder-weights none`) rather than dying with a stack trace.

### B11. Scene ids collapse to one value, so the leakage check is vacuous

`default_scene_id` falls back to the nearest non-generic parent directory, which
for a flat archive of tiles is the archive directory itself. Every one of the
1200 scenes in `/Volumes/Ventoy/Oil` therefore reports as scene `"Oil"`, and the
run report says `"n_scenes": 1` for a 19-pair split.

The consequence is narrow but real: `scene_leakage` finds nothing to leak,
because as far as it can tell there is only one scene. The split itself is still
made per file (`split_strategy: per_file_hash`), and the trainer prints a loud
warning that tiles were split per file rather than per acquisition and that
cross-split metrics are therefore optimistic — so the risk is disclosed, just
not in the field where a reader would look for it.

This is not fixed here on purpose. Changing `default_scene_id` changes which
files land in train and val, which would invalidate the end-to-end run that was
just verified. The archive genuinely carries no scene identifier — every GeoTIFF
has only generic TIFF tags — so the honest fix is an explicit
`--scene-regex` (or a scene-grouping index), not a better guess from the path.

---

## C. Unverified — nobody has checked

| Item | What is missing |
|---|---|
| `Dockerfile.train` | never built — no Docker on the machine it was written on |
| A training run long enough to learn anything | the verified runs are bounded smoke runs (`--max-steps-per-epoch`), so their IoU is not a result |
| Postgres/PostGIS persistence path | never executed against a live server |
| `docker compose up --build`, either profile | only `config` validated |
| Full 7-service native stack under load | services started individually, not together |
| Zenodo preparation on real archives | fixture-tested only |
| Any trained model | no checkpoint exists |
| Anomaly detector | not trained; trainer refuses empty input |
| UI against a live backend | type-checked and built; not driven end-to-end against running services |
| `intel` narrative with a real LLM | no model available |
| Alerts (SMTP/webhook) | no endpoint configured |
| Grafana / Prometheus | not started in the lite profile |

---

## D. Corrections to the original brief

These were found by bisecting failures, not by preference. Full evidence in
`docs/VERIFICATION.md`.

| Brief says | Reality |
|---|---|
| Demo product ID `S1A_IW_GRDH_1SDV_20200807T024930_…` | **Does not exist.** 0 catalogue matches; no S1 IW pass over Mauritius on 2020-08-07 at all. |
| OData filter `ContentDate/Start gt datetime'…'` | **HTTP 400.** CDSE rejects the `datetime'…'` prefix; use a bare ISO literal. |
| `$select=…,OrbitNumber` | **HTTP 400.** Not a selectable field. |
| Download `.SAFE` via OData | **401** `DAT-ZIP-609 "Token audience not allowed"`. Use the Sentinel Hub Process API. |
| SNAP / pyroSAR preprocessing | Java- and x86-centric; not viable on arm64. Process API replaces it. |
| CUDA / `Dockerfile.gpu` | No NVIDIA GPU. CPU/MPS only. |
| `cdsapi` + `~/.cdsapirc` | Dead. Use the CDS OGC process API, or legacy `cdsapi` against `/api`. |
| `sentinelsat` / SciHub | Permanently closed. Use CDSE OData (metadata) + Process API (pixels). |
| MarineCadastre for Indian Ocean AIS | US waters only. Never covers that AOI. |
| NOAA NOMADS GFS OPeNDAP | **Retired** (SCN 25-81); returns HTML. Use the NOMADS filter. |
| `o.set_config('drift:horizontal_diffusivity')` | Removed in OpenDrift ≥1.14. |
| `o.set_config('drift:wind_drift_factor')` | Removed; pass `wind_drift_factor=` to `seed_elements`. |
| `o.seed_elements(oiltype=…)` | Renamed to `oil_type=`. |
| Full 11-container compose on 16 GB | Would thrash. DB+Redis in Docker, services native. |
| 2 GB per-scene download | 44 MB Process API GeoTIFF. 180 MB for the whole real scene set. |

---

## E. What would change these answers

1. **A satellite-AIS subscription** → real vessel attribution for the 2020 case.
2. **A cloud GPU + the Zenodo archives** → a trained, benchmarked detector.
3. **An accepted ERA5 licence and a running Postgres** → the database persistence
   path and full-stack compose could both be verified.
4. **A labelled benchmark** → the confidence interval could be calibrated, and
   only then called a probability.
