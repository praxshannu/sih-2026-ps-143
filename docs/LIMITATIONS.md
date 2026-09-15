# SENTINEL — Limitations

This file exists because a system that reports its own limitations is more
trustworthy than one that does not, and because every entry below is something a
reviewer would otherwise discover the hard way.

Entries are grouped by whether they are **scientific** (the answer may be wrong),
**operational** (the answer is right but the system cannot deliver it here), or
**unverified** (nobody has checked yet).

---

## A. Scientific limitations

### A1. The deployed detector is not a trained model

Detection is a hand-tuned operator (Lee-sigma → Solberg two-gate threshold →
morphology → connected components), not a learned model. It has no weights, so
"accuracy" is not a meaningful axis for it.

Consequence: it is **reproducible and explainable**, and it has **never been
validated against a labelled benchmark**. It cannot generalise the way a trained
model would, and its thresholds were calibrated on the Wakashio scenes and the
scene-statistics envelope in `pipeline.py`.

A Tier-B learned model now exists in the sense that one has been *trained* —
three times, for 10 epochs each on the real archive (MODEL_CARD §2.4). The first
run scored `best_val_iou 0.8050` on a per-file split; once the split was made
leak-free the same command scored **0.2981**, so about 63 % of the first number
was leakage (B11). The third run finally carried a held-out test set and scored
**0.1447** on it, against 0.5180 on val in the same run — so the leak-free val
figures were optimistic by a further ~3.6× (B11).

None of them is deployed and none must be treated as validated. **Quote 0.1447.**
It rests on 6 acquisitions and is therefore wide, and there is still no labelled
benchmark to check any of it against. "Trained" is not "usable".

See `MODEL_CARD.md` §1 and §2.4.

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

### A10. The two pooled-tail checks miss, and the check cannot settle itself

The synthetic set matches the real archive on **26 of 28** distribution checks
(`data/synthetic/match_report.json`). The two that miss are both pooled tails:

| Check | Real | Synthetic | Δ | Tolerance |
|---|---|---|---|---|
| `VV.p99.9` | −16.79 dB | −20.59 dB | 3.80 dB | 2.0 dB |
| `VH.p99.9` | −8.77 dB | −6.56 dB | 2.21 dB | 2.0 dB |

These numbers are **reproducible**: two consecutive
`generate_synthetic.py --verify-only --sample 32` runs return identical values,
and the real side is identical across every run recorded here (the sampler sorts
by name and walks a fixed stride, so the same scenes are measured every time).
An earlier note in this repository recorded 27/28; that figure came from an
earlier state of the generator and does not reproduce against the current code.

**Why they miss, mechanically.** For a two-class population where bright pixels
are a fraction `f` of a scene, the scene's p99.9 is the bright pool's quantile
`q = 1 − 0.001/f` — about the pool's p78 for VV and p41 for VH. Both checks
therefore sit on the steepest part of the extreme population, where the
generator's coverage estimate and the shape of its knot curve both matter most.

**Why this is a limitation and not a bug: the reference value is not stable.**
Re-running the same generator with `--sample 96` instead of `--sample 32` — same
code, same seed, three times the real scenes — moved the *real* `VV.p99.9` from
−16.79 to **−13.32 dB**, a 3.5 dB swing produced purely by looking at more of the
archive. `VH.p99.9` moved from −8.77 to −7.75 dB the same way. A statistic that
moves 3.5 dB with sample size cannot be a ±2 dB acceptance gate, and a generator
cannot track a target that moves with the sample.

A bootstrap agrees: a 32-scene mean of VV bright coverage has a 5–95% band of
0.0027–0.0067 (±46%). The per-scene statistics, which do not pool, match closely
(VV p99 mean: real −10.54, synthetic −10.21) — which is what a generator fitted
to the measured population should do.

One fix was attempted and **reverted**: walking the measured records without
replacement (`record_index`) so the drawn multiset matches the measured one
instead of resampling it with replacement. It is principled — a draw with
replacement is a bootstrap and adds variance the archive does not have — but at
the same 32-scene sample it produced 25/28 rather than 26/28, adding a `VH.p0.1`
failure and widening `VV.p99.9` to 6.7 dB. The effect is not separable from the
noise of the check it was meant to fix, and a change that worsens the measured
outcome on an unverifiable theory is not worth shipping.

What would settle it: profile all 1200 scenes rather than a sample, or replace
these two checks with ones that state their sampling band and fail only when the
value falls outside it. Until then, treat the two `p99.9` comparisons as
unresolved rather than as evidence about the generator.

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

### B11. The split leaked, and it was measured rather than argued

**Fixed in code; the verified run's number is still inflated.** The paragraph
below is the original entry, kept because the reasoning is what the fix had to
answer.

The archive has no scene identifier — every GeoTIFF carries only generic TIFF
tags — so there was nothing to split *by*. Two things went wrong at once:

1. `default_scene_id` fell back to the nearest non-generic parent directory,
   which for a flat archive of tiles is the archive directory itself. Every one
   of the 1200 scenes reported as `"Oil"`, and the run report said
   `"n_scenes": 1` for both splits.
2. `SAROilSpillDataset` did not use the manifest's `scene_id` at all; it
   re-derived ids from the file path, so the loader undid whatever grouping the
   indexer had computed.

Together those made the trainer's `scene_leakage` guard **structurally unable to
fail**. It compares scene ids across splits; with one scene per archive there was
nothing to leak, and with per-file ids every tile was its own scene, so the check
passed vacuously in both arrangements. A guard that cannot fail is not a guard,
and it is why the per-file split survived being called verified: the warning was
in the log, it was accurate, and it was printed on every single run.

**What the leakage actually was.** Tiles were grouped by the ground they share
(union-find over bounding boxes; `scene_grouping="footprint"`), which is the fact
a centroid or a grid cell was standing in for. Measured:

| | count |
|---|---|
| tiles in the archive | 1200 |
| acquisition areas (footprint groups) | 195 |
| areas straddling a *per-file* split of all 1200 tiles | 78 |
| largest single area | 114 tiles |
| tiles in the 10-epoch run | 120 |
| areas the run touched | 42 |
| **areas straddling the run's split** | **11** |
| **tiles inside a straddling area** | **58 of 120 (48.3%)** |

So nearly half of the tiles the run trained and validated on came from an
acquisition that appears on both sides of the boundary. The largest,
`00045`, put 18 of its tiles across the split. This is why the run's
`best_val_iou 0.8050` (MODEL_CARD §2.4) is an upper bound: the number was
partly measuring memorisation of tiles the model had already seen the
neighbours of.

**What is fixed.** `scene_grouping="footprint"` assigns ids from shared ground;
`SAROilSpillDataset` reads them from the manifest; the mode that produced an
index is recorded in `source.json` and an index built by a different mode is
rebuilt rather than reused; and the leakage guard now has something to check, so
it can fail. The `--max-pairs` cap is leak-safe under this grouping — it trims
within a split, in name order, which shortens acquisitions but never moves one
across the boundary.

**What is not fixed — and now measured.** A footprint split removes near-duplicate
*ground*, not shared *conditions*: tiles within one acquisition share its
calibration, incidence angle and wind regime. The first run with a genuine
held-out test set (`20260915T180945Z-real-10ep-512-holdout-7e9d57`, 114 train /
24 val / 12 test pairs) shows what that costs:

| metric | val @ epoch 7 | held-out test | gap |
|---|---|---|---|
| IoU | 0.5180 | **0.1447** | 0.3732 |
| F1 | 0.6824 | 0.2529 | 0.4296 |
| precision | 0.9507 | 0.1516 | 0.7991 |
| recall | 0.5323 | 0.7618 | −0.2295 |

The failure is neither subtle nor noise. On held-out acquisitions the model
over-predicts: it finds 76 % of the true oil while emitting **333,079
false-positive pixels against 59,513 true positives** — roughly six wrong pixels
for every right one. At the epoch the run selected, its val precision was 0.95.

So **0.5180 on this split and 0.2981 on the previous one were both val numbers**,
each selected on the split it is reported from. Every headline this project has
produced was between 2× and 5.5× the one number nothing was tuned against:

| split | number | what it actually is |
|---|---|---|
| per-file, val | 0.8050 | leaked — an upper bound |
| footprint, val (run B) | 0.2981 | selected on val, 8 acquisitions |
| footprint, val (this split) | 0.5180 | selected on val, 8 acquisitions |
| **footprint, held-out test** | **0.1447** | **the number to quote**, 6 acquisitions |

Caveat on the caveat: 6 acquisitions is small, so 0.1447 is a wide estimate. But
the direction and the mechanism — precision collapse on unseen acquisitions — are
unmistakable, and a wide estimate of the right quantity beats a tight estimate of
the wrong one.

**Why the earlier numbers could not have revealed this.** Until
`test_fraction` was set, the trainer *built* a test split and never scored it, and
the footprint claim ended with "there is no held-out test set" unconditionally.
The code was written for the state of the world that had already bitten and never
revisited; the caveat was present, accurate, and therefore read as boilerplate.

This was not fixed by changing `default_scene_id`. That would still be a guess
from a path; the archive genuinely carries no scene identifier, so the honest fix
is the explicit grouping index that now exists.

---

### B12. The archive's `VV`/`VH` band order is assumed, not measured

The pipeline labels the two archive bands `("VV", "VH")` in that order
(`SAR_BAND_NAMES`). Nothing verifies it. The archive carries no band metadata —
every GeoTIFF has only generic TIFF tags — so the order is inherited from the
brief rather than read from the data.

The measured levels point the other way. Fitting per-band statistics from 32
training scenes gives band 0 `mean −32.34 dB` and band 1 `mean −20.16 dB`: band 1
is **12.2 dB brighter**. The reference product in `data/sar/`, whose bands *are*
declared (`sigma0_VV_linear`, `sigma0_VH_linear`), has the co-pol 15–19 dB
brighter than the cross-pol across three scenes:

| scene | VV median | VH median | VV − VH |
|---|---|---|---|
| `browser_test_wakashio_20260913T191302Z.tif` | −15.23 dB | −33.82 dB | +18.58 dB |
| `mumbai_20260913_recent.tif` | −21.41 dB | −36.58 dB | +15.17 dB |
| `wakashio_20200729_early.tif` | −16.76 dB | −31.84 dB | +15.08 dB |

So the archive is most likely stored `(VH, VV)` and the two names are swapped.

What this does and does not affect:

- **Training: nothing.** Both channels are fed to the network either way, and
  the synthetic generator profiles and reproduces by band index, so the 26/28
  match in A10 is unaffected.
- **Physical interpretation: everything.** Any statement in this repository
  attached to the name "VV" — including the equivalent-looks figures in
  `docs/MODEL_CARD.md` and the per-band normalization above — is attached to a
  band whose polarisation is unverified. The numbers are sound; the labels on
  them are not.

Not renamed here, deliberately. Renaming would change the band labels written
into every index, checkpoint and match report, invalidating verified artifacts,
in exchange for a claim that is well-supported but not proven. The fix is to
read the order from data that declares it, or to obtain the acquisition metadata
for this archive — not to relabel on inference.

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
