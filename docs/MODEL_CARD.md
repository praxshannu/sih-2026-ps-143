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

### 2.4 The training pipeline exists, and has now been run twice on the real archive

This does **not** change the status above: no trained model is deployed. What
changed is that the pipeline which would produce one is real, runs to
completion, and has been run for a full 10 epochs over the real archive twice —
once on a leaky split, once on a leak-free one. The second run is the one worth
reading.

#### Two runs, one budget

Both runs used the same budget: 120 image/mask pairs, 512², batch 4, 10 epochs,
MPS, the archive read in place. The only difference is how tiles were grouped
before the split.

| | run A | run B |
|---|---|---|
| run id | `20260915T154010Z-real-10ep-512-828cee` | `20260915T164046Z-real-10ep-512-footprint-b79e83` |
| `scene_grouping` | `per_file` | `footprint` |
| split strategy | `per_file_hash` | `footprint_connected` |
| train | 96 pairs / 1 "scene" | 106 pairs / **33 acquisitions** |
| val | 24 pairs / 1 "scene" | 14 pairs / **8 acquisitions** |
| val oil fraction | 15.2 % | 3.6 % |
| duration | 1703 s | 1581 s |
| **best val IoU** | **0.8050** | **0.2981** |
| `source_unchanged` | True | True |

Run A's `n_scenes: 1` is not a typo. The archive has no scene identifier and the
loader was deriving ids from the file path, which for a flat archive collapses
to the archive directory name — so the trainer's leakage guard saw one scene and
had nothing to check. See LIMITATIONS B11.

#### Run A — per-file split

```bash
python scripts/train.py --data-source real --epochs 10 --image-size 512 \
    --batch-size 4 --max-pairs 120 --run-name real-10ep-512
```

| epoch | train IoU | val IoU | val precision | val recall | val F1 |
|---|---|---|---|---|---|
| 1 | 0.1861 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.4030 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 3 | 0.3185 | 0.3969 | 0.9913 | 0.3982 | 0.5682 |
| 4 | 0.4940 | 0.7840 | 0.9252 | 0.8371 | 0.8789 |
| 5 | 0.5815 | 0.7807 | 0.9340 | 0.8262 | 0.8768 |
| 6 | 0.5887 | 0.7098 | 0.9202 | 0.7564 | 0.8303 |
| 7 | 0.4606 | 0.7815 | 0.9625 | 0.8060 | 0.8773 |
| 8 | 0.5072 | 0.7611 | 0.8636 | 0.8650 | 0.8643 |
| 9 | 0.5740 | 0.7594 | 0.8126 | 0.9207 | 0.8633 |
| 10 | 0.5922 | **0.8050** | 0.9153 | 0.8698 | 0.8920 |

`best_val_iou 0.8050` at epoch 10.

#### Run B — footprint split

```bash
python scripts/train.py --data-source real --epochs 10 --image-size 512 \
    --batch-size 4 --max-pairs 120 --run-name real-10ep-512-footprint
```

Tiles were grouped by the ground they share, so no acquisition appears on both
sides of the boundary. 1200 tiles resolve to 195 acquisition areas; the 120 kept
here span 33 train acquisitions and 8 val ones, and **zero** straddle the split
(verified independently against the manifest, and enforced by the trainer).

| epoch | train IoU | val IoU | val precision | val recall | val F1 |
|---|---|---|---|---|---|
| 1 | 0.1779 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.3621 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.4333 | 0.2458 | 0.3230 | 0.5070 | 0.3946 |
| 4 | 0.3646 | 0.2764 | 0.4957 | 0.3845 | 0.4331 |
| 5 | 0.5688 | 0.2520 | 0.4314 | 0.3774 | 0.4026 |
| 6 | 0.5611 | 0.2663 | 0.3175 | 0.6227 | 0.4206 |
| 7 | 0.5994 | 0.2071 | 0.2527 | 0.5342 | 0.3431 |
| 8 | 0.6469 | **0.2981** | 0.4289 | 0.4943 | 0.4593 |
| 9 | 0.5857 | 0.2865 | 0.3352 | 0.6639 | 0.4454 |
| 10 | 0.7044 | 0.2552 | 0.3213 | 0.5537 | 0.4066 |

`best_val_iou 0.2981` at epoch 8. Train IoU sits *below* val IoU for most of the
run because train is measured under augmentation — random flips, 90° rotations
and crops — while val uses the eval transform. Train is the harder measurement,
so the two are not comparable to each other.

#### What the difference between them measures

**0.8050 − 0.2981 = 0.5069.** About **63 % of the number run A reported was the
leak.** That is the honest headline of this section, and it is the reason run A's
figure was labelled an upper bound rather than a result.

Three details make it more than an arithmetic comparison:

- **Train went up while val went down.** Run A's final train IoU was 0.5922; run
  B's is 0.7044. The model fits its training tiles *better* on the leak-free
  split and generalises *worse*. That is exactly what removing a leak looks
  like: with near-duplicate tiles on both sides, the validation set was a
  slightly harder copy of the training set, and the model could score well on it
  without learning anything transferable.
- **The two val sets are not the same difficulty.** Run A's val tiles were 15.2 %
  oil; run B's are 3.6 %. A quarter of the positives, so far less room to be
  right by accident.
- **Run B's val IoU stops improving.** It reaches 0.2458 at epoch 3 and then
  oscillates between 0.21 and 0.30 for seven more epochs with no trend, while
  train IoU climbs to 0.70. Run A climbed steadily to 0.8050. The plateau is the
  more informative shape: on unseen ground this model has stopped learning.

#### 0.2981 is not a detection performance either

It is a better number than 0.8050 because it is not inflated, not because it is
good. Two limitations survive the fix:

- **The val split is thin.** 14 tiles across 8 acquisitions. `val_fraction=0.2`
  counts *scenes*, and acquisitions range from 1 to 114 tiles, so a scene-level
  fraction cannot hit a tile fraction — this run got 11.7 % of tiles for a 20 %
  scene fraction. One unusually easy or hard acquisition would move the score a
  lot.
- **There is no held-out test set.** `test_fraction` is 0, so val is the only
  cross-split number and it was used for model selection. A footprint split
  removes shared *ground*, not shared *conditions*: tiles within one acquisition
  still share its calibration, incidence angle and wind regime. This measures
  transfer to unseen ground, not to unseen weather.

The run's own `claim` string says all of this, and says it conditionally — the
`per_file_hash` caveat is no longer printed for a footprint split, because a
caveat that appears on every run is boilerplate and boilerplate is how run A's
leak survived being called verified.

#### The two epochs of zero are not a defect

Epochs 1–2 report val IoU `0.0000` in both runs. In run B both also report
precision `0.0000`, which is the honest reading: at the 0.5 threshold the model
predicted **nothing at all** — `tp 0, fp 0, fn 131129` — against a val set that
is 3.6 % oil. The train IoU over those same epochs was 0.18–0.36 because train
metrics accumulate on batches the optimiser has just stepped on, while val is
seen without a gradient. The model was sitting near the class prior and had not
yet escaped it. It escaped at epoch 3 in both runs.

`degenerate_no_positive_evidence` is `0.0` for every epoch of both runs, which
is correct: it means `union == 0`, i.e. nothing predicted *and* nothing labelled.
Here there were 131,129 labelled positive pixels, so the metric is well-defined
and the zero is a real measurement.

#### What the runs establish

- **The archive is read where it lies.** Two independent checks. The trainer's
  `SourceFingerprint` records `mtime_ns`, entry count and `st_dev` for both
  `/Volumes/Ventoy/Oil` (2400 entries, `st_dev 16777238`) and the mask
  directory, and re-checks them at the end → `source_unchanged: True` in both
  runs. And a SHA-256 over every filename, size and `mtime_ns` in the archive,
  taken before and after, is **identical** across the whole session — the index
  rebuild, the header scan and the 10 epochs:

  ```
  before: {"dir_mtime_ns": 1682009727000000000, "entries": 2400,
           "fingerprint": "69d35c71…d7ac4", "free_bytes": 14094827520}
  after:  {"dir_mtime_ns": 1682009727000000000, "entries": 2400,
           "fingerprint": "69d35c71…d7ac4", "free_bytes": 14094827520}
  ```

  Nothing was copied, staged or written. Free space is unchanged to the byte.
- **Both data sources work from the same code path**, switchable at runtime
  with `--data-source`, neither copying anything to the local machine.
- **Checkpoints, `run_report.json` and `metrics.jsonl` are written atomically**
  and carry the provenance stamp, the normalization statistics, the exact
  source directories and the fingerprints.
- **The split can be checked rather than trusted.** The trainer refuses to start
  if any scene id appears in more than one split (`reason="scene_leakage"`), and
  that guard is now capable of failing — under `per_file` it could not.

#### Defects these runs found, and fixed

Five, all of which had made training impossible, the report wrong, or the
number meaningless:

1. **The geometry transforms were written for a 2-D mask** while the loader
   supplies `(1, H, W)`. `mask[rows, cols]` on a 3-D array means
   `mask[rows, cols, :]`, so crops came back with an empty axis; and
   `mask[:, ::-1]` flipped the *height* while the image flipped the *width*,
   which would have trained on masks desynchronised from their images without
   raising anything.
2. **A split with no rows was fatal.** `test_fraction` defaults to `0.0`, so the
   real path has no test split — and building a dataset from an empty manifest
   sent the loader looking for `images/` + `masks/` that a prepared root does
   not have.
3. **The extreme-value model lost the scene tail** — a coverage recorded with an
   all-`NODATA_DB` curve, and a knot set that stopped below the statistic the
   match check actually measures.
4. **`--resume` reset the report's best-so-far.** `best_val_iou`/`best_epoch`
   were seeded into local variables but never into the report, so a resumed run
   that did not beat its checkpoint reported `best val IoU 0.0000 at epoch -1`
   while holding a checkpoint that scored better than that.
5. **The split leaked, and the guard meant to catch it could not.** Two bugs at
   once: `default_scene_id` collapsed every tile to the archive directory name,
   and `SAROilSpillDataset` ignored the manifest's `scene_id` column and
   re-derived ids from the path. Either alone made `scene_leakage` vacuous. The
   fix is footprint grouping plus reading the id the index actually wrote; the
   measurement of what it was worth is run A versus run B above. LIMITATIONS B11.

#### Per-band normalization, and an unverified band order

Run B fitted per-band statistics from 32 training scenes:
`VV mean −32.35 std 4.26`, `VH mean −19.99 std 3.53` (dB). Run A's were
`−32.34 / 4.63` and `−20.16 / 4.04`; they differ because the training subset
changed, which is expected and is why the statistics are recorded in the report
rather than assumed.

Those labels are an assumption, not a measurement — see LIMITATIONS B12. The
archive has no band metadata, and band 1 is 12.2 dB *brighter* than band 0,
whereas the reference product in `data/sar/`, whose bands are declared
`sigma0_VV_linear` / `sigma0_VH_linear`, has the co-pol 15–19 dB brighter than
the cross-pol. On that evidence the archive's bands are most likely stored
`(VH, VV)`, so the names on these two numbers are probably swapped. Training is
unaffected — both channels are used either way — but any physical reading of
"VV" in this document is unverified.

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
| A trained segmentation model exists | **Half true.** Two 10-epoch runs exist (§2.4). Neither is deployed; neither was evaluated on held-out data; neither is fit for use. |
| The training pipeline runs end-to-end on both data sources | **Verified**, two 10-epoch real runs plus a synthetic run, `source_unchanged: True` in both |
| The real archive is read in place and never written | **Verified** — trainer fingerprint plus an independent SHA-256 either side of a run |
| Val IoU 0.8050 is a detection performance | **False.** 0.5069 of it — 63 % — was leakage from a per-file split. The leak-free run on the same budget scores **0.2981**. |
| The leak-free run produced a useful model | **False.** 0.2981 rests on 14 val tiles across 8 acquisitions with no held-out test set, and val IoU stops improving after epoch 3. It is an honest number, not a good one. |
| No acquisition straddles the train/val split | **Verified** for the current index — 195 acquisition areas, 0 straddling — independently against the manifest and enforced by the trainer. |
| `--resume` reports the checkpoint's own best, not its last epoch | **Verified** by test and in situ |
| The `VV`/`VH` band order of the archive is known | **False.** No band metadata; the measured levels suggest `(VH, VV)` is stored. LIMITATIONS B12. |
| The synthetic set reproduces the real distribution | **26 of 28 checks.** Both misses are pooled `p99.9` tails — see LIMITATIONS A10. |
| Performance on a labelled benchmark is known | **False.** No benchmark has been run. |
| Anomaly detection is operational | **False.** Not trained; trainer refuses empty input. |
| Attribution weights are fitted | **False.** Chosen by hand, uncalibrated. |
