# SENTINEL — Model Card

**Status: NO TRAINED MODEL IS DEPLOYED.** This card documents what actually
runs, what it can and cannot claim, and what a trained model would have to
demonstrate before it replaced the current detector.

This is deliberately written as a model card for a model that does not exist,
because the alternative — quietly presenting a deterministic operator as if it
were a learned detector, or shipping an untrained network's outputs — is the
specific failure this project's rules exist to prevent.

---

## 1. What runs today: Tier-A deterministic detector

| | |
|---|---|
| **Identifier** | `deterministic-tier-a-lee-solberg-v1` |
| **Type** | Deterministic image operator. **No learned parameters. No weights. No training data.** |
| **Code** | `services/detect/app/pipeline.py`, `processors/{tiling,deterministic,geometry,evidence,validation}.py` |
| **Input** | Sentinel-1 GRD GeoTIFF, ≥2 bands (VV, VH), georeferenced, EPSG:4326 or any CRS with a valid transform |
| **Output** | GeoJSON polygons (EPSG:4326) + per-polygon metrics + Wilson 95 % CI + per-pixel evidence map |
| **Dependencies** | numpy, scipy, scikit-image, rasterio, shapely. **No torch.** |

### 1.1 Algorithm

1. **Validate** — dimensions, CRS, band count, nodata, valid-pixel fraction.
   Rejected scenes return `invalid_scene` with a reason code; they never reach
   inference.
2. **Tile** — 1024 px tiles with a 192 px halo, so every operator sees the same
   neighbourhood it would in an untiled run. A slick crossing a tile boundary is
   merged because connected components run over the **stitched** mask.
3. **Lee-sigma speckle filter** (7×7) on linear σ⁰.
4. **σ⁰ → dB**, land masked via `dataMask`.
5. **Solberg two-gate adaptive threshold** (Solberg et al., 1999):
   * **Gate 1 (local):** pixel ≥ `k·σ` below its local 151×151 mean.
   * **Gate 2 (scene):** pixel ≥ 2.5 dB below the 75th percentile of ocean dB
     (the clean-sea baseline; the upper quartile is assumed uncontaminated).
6. **Morphological** opening (1) + closing (2).
7. **Connected components**, then reject blobs with elongation > 5 (wind streaks).
8. **Per-blob metrics** — area, perimeter, compactness, elongation, mean dB,
   local contrast, scene contrast.
9. **Fay (1971) inverse spreading** → age in hours, with a regime statement.
10. **Confidence** combines contrast, scene validity and blob area. The Wilson
    95 % binomial CI is reported for every polygon.

### 1.2 Verified behaviour on real data

Real Copernicus GeoTIFFs, this machine, `docs/VERIFICATION.md` §7 and
`docs/COMPLETION_REPORT.md` §1.1:

| Scene | Top area | Mean σ⁰ | Scene contrast | Confidence | Wilson 95 % CI |
|---|---|---|---|---|---|
| `wakashio_20200729_early` | 0.51 km² | −19.6 dB | −4.23 dB | 0.811 | [0.682, 0.896] |
| **`wakashio_20200810_peak`** | **3.347 km²** | **−23.83 dB** | **−9.79 dB** | **0.934** | **[0.902, 0.956]** |
| `wakashio_20200816_late` | 0.29 km² | −19.44 dB | −9.19 dB | 0.698 | [0.513, 0.835] |
| `wakashio_20200822_recovery` | 0.59 km² | −23.95 dB | −8.43 dB | 0.823 | [0.706, 0.900] |

The 10 Aug peak detection — 3.347 km² at −9.79 dB below scene baseline — is the
textbook SAR oil-spill signature, and published Copernicus imagery for the
Wakashio spill corroborates the size and contrast. That is **corroboration, not
validation**: it is one event, checked against a published image, not against a
labelled benchmark.

### 1.3 What the confidence number IS, and is NOT

**IS:** a Wilson 95 % binomial interval over the pixel-level evidence inside the
polygon (for the peak detection, `n_effective = 334`). It answers *"how
consistent is the dark-pixel evidence within this polygon?"*

**IS NOT:** a calibrated probability that the polygon is oil. It has never been
calibrated against ground truth, because no labelled ground truth exists here.

Presenting it as P(oil) would be the single easiest way to mislead with this
system, and the response schema names the scale explicitly
(`"probability the top detection is a mineral-oil slick"` is *not* what the
field means — the `scale` field states the actual basis) so that a caller
cannot do it by accident.

### 1.4 Known failure modes

| Mode | Behaviour |
|---|---|
| Wind outside 2–10 m/s | `wind_viability: LOW_CONFIDENCE_NO_WIND`; the look-alike gate cannot run |
| No wind supplied at all | `missing_wind_forcing` flag; every detection flagged |
| Rain cells, low-wind patches, biogenic films, grease ice | Partially filtered by the elongation gate; not fully separable without wind |
| Scene statistics outside the calibration envelope | `out_of_distribution` state; detections returned **flagged, not suppressed** |
| Land contamination | Land masked, but a scene mostly land raises the p75 baseline and suppresses real detections |

---

## 2. Tier-B: UNet++ with SCSE attention — NOT DEPLOYED

| | |
|---|---|
| **Identifier** | `unetpp_scse_resnet34` |
| **Code** | `services/detect/app/models/unetpp_scse.py`, `processors/unet_path.py` |
| **Status** | `model_status: "untrained"`, `probabilities: null` |
| **Checkpoint** | `checkpoints/best_model.pth` — **does not exist on this machine** |

### 2.1 Why it is not deployed

There are **no labelled oil-spill masks** on this machine, and 100 epochs of
UNet++/efficientnet-b4 on an M2 CPU is many hours plus ~10 GB of dataset.

Shipping a network that has never converged would produce numbers that look
authoritative and mean nothing — exactly what the "no synthetic data" rule
exists to prevent. So the response says `untrained` and emits **no
probabilities at all**: not zeros, not a soft-max of an ImageNet encoder, not a
random-init forward pass.

This is enforced by test:
`test_untrained_unet_emits_no_probabilities` asserts `probabilities is None`.

### 2.2 Measured: a random-init UNet++ has no bounded output range

An earlier test asserted `−50 < logits < 50` from a randomly-initialised
network. Measured on this host, the logits span roughly **−4700 … +5700**. The
old bound was never a contract; it held for the initialisation the network
happened to ship with. The test now asserts finiteness (no NaN, no Inf) — the
property that actually matters — alongside the guard that keeps such output from
ever reaching an analyst.

### 2.3 What a trained model would have to demonstrate

1. Trained on **real** labelled masks — the Zenodo set (DOI
   `10.5281/zenodo.8320179`), split **by scene/source**, never by random tiles.
2. **Part III held out.** Never trained on. Enforced in code and asserted in tests.
3. Reported: IoU, precision, recall, F1, per-class metrics, confusion matrix —
   `ml/training/metrics.py`.
4. Normalisation statistics and an out-of-distribution check against the
   training envelope, so a scene unlike the training data is flagged rather than
   silently scored.
5. Metrics stamped with `provenance` and `scientifically_valid`. A run on
   synthetic fixtures is stamped `synthetic_fixture` / `false` — a number that
   looks authoritative and means nothing is worse than no number.

**None of this has happened.** `scripts/prepare_zenodo_dataset.py` is
implemented and unit-tested against fixtures; it has never seen the real
archives (40.7 GB + 45.9 GB + 9.9 GB monolithic `.7z` — the disk cannot take it).

---

## 3. The anomaly detector (LSTM autoencoder)

| | |
|---|---|
| **Code** | `ml/training/train_anomaly.py`, `services/attribute/app/engine/anomaly_detector.py` |
| **Status** | Not trained. `train_anomaly.py` **refuses to run** on an empty or missing data directory. |
| **Purpose** | Score AIS behavioural anomalies (gaps, speed deviations) for attribution |

It never auto-generates synthetic samples. If it had, "anomalous" would mean
"anomalous relative to data we invented", which is meaningless.

---

## 4. Attribution scoring — not a learned model

`K_ij = w₁·S_prox + w₂·S_temp + w₃·S_traj + w₄·S_anom + w₅·S_type`

Fuzzy membership functions (Gaussian proximity, triangular temporal, cosine
trajectory, reconstruction-error anomaly). Weights default to
`ATTRIBUTION_FUZZY_WEIGHT=0.6` / `ATTRIBUTION_XGB_WEIGHT=0.4`.

**These weights are uncalibrated.** They were chosen, not fitted. No labelled
set of confirmed oil-spill attributions exists to fit them against, and the
system says so rather than implying the score is a probability.

Every score carries a Wilson 95 % CI (AGENTS.md). With no real AIS coverage, the
API returns an **empty ranking with a reason** — it never fabricates a
suspect list.

---

## 5. Summary of claims

| Claim | Status |
|---|---|
| A deterministic detector runs on real Sentinel-1 data | **Verified**, 4 scenes, reproduced twice |
| It produces Wilson 95 % CIs, never point estimates | **Verified** by test |
| Its confidence is a calibrated P(oil) | **False.** Pixel-evidence interval only. |
| A trained segmentation model exists | **False.** None deployed, none trained. |
| Performance on a labelled benchmark is known | **False.** No benchmark has been run. |
| Anomaly detection is operational | **False.** Not trained; trainer refuses empty input. |
| Attribution weights are fitted | **False.** Chosen by hand, uncalibrated. |
