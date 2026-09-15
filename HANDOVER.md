# HANDOVER

Everything you need to pick this up cold. Read this first; it points at the rest.

Repository is clean at commit `4076c90`, **64 commits**, no uncommitted work.

---

## What this is

SENTINEL — an oil-spill detection and attribution system for SAR imagery
(SIH26143 / NTRO). Seven FastAPI services, a React/deck.gl globe UI, a UNet++
detector and an OpenDrift-based drift/attribution engine. It runs on one Apple
M2 laptop, CPU/MPS only, no CUDA.

Two things make it unusual, and both are deliberate:

1. **It trains on the real archive in place**, off an external disk, without
   copying a byte. Proven by SHA-256 fingerprint either side of a full run.
2. **It tells you what it does not know.** Every metric carries a provenance
   tag, every claim has a confidence interval, and the docs list the limitations
   rather than burying them.

---

## Run it

```bash
cd /Users/praxsmac/projects/claude1/sentinel
python -m venv .venv && . .venv/bin/activate && pip install -r ml/requirements.txt
cp .env.example .env          # defaults work for the synthetic path

# The fast, self-contained path — no external disk, no credentials:
make test                     # 302 tests
python scripts/train.py --data-source synthetic --epochs 2 --max-pairs 24
```

That is the whole loop. `--data-source real` reads `/Volumes/Ventoy/Oil`
(1200 GeoTIFFs, 2-band float32 dB, 2048²) in place; it needs that disk mounted
and the index built (`--rebuild-index`, ~7 min cold).

Full service stack, API reference and the MV Wakashio demo walkthrough are in
`README.md`. Operational detail is in `docs/RUNBOOK.md`.

---

## What is actually verified

| Claim | Evidence |
|---|---|
| 302 tests pass (115 services + 112 ml + 75 core) | `make test`; fresh clone = 297 passed + 5 skipped (the 5 need `data/sar`, which is untracked) |
| Real path trains end-to-end, in place | run `20260915T164046Z-…-footprint-b79e83`, 10 epochs, 1581.3 s, `source_unchanged: True`, disk fingerprint byte-identical |
| Synthetic path trains end-to-end | run `20260915T171617Z-synth-regression-check-fe26e9` — metrics **bit-identical** to the pre-change baseline |
| The leak-free split works | index rebuilt to **195 acquisitions, 0 straddling**, `train=1059 / val=141` |
| A held-out test set now exists **and is scored** | run `20260915T180945Z-…-holdout-7e9d57`: 114 / 24 / 12 pairs over 41 / 8 / 6 acquisitions, **test IoU 0.1447** against val 0.5180 |
| Deterministic detection (no torch) | 10 Aug 2020 real hit: 3.34 km², −9.81 dB, confidence 0.934 |
| Synthetic set matches the real one | `data/synthetic/match_report.json`: **26/28** checks within tolerance, reproducible |
| Lint, format, types | `make lint` / `make format` / `make typecheck` clean across 8 mypy targets |

---

## The one number that matters, and the trap around it

```
per-file split, val            0.8050   ← leaked, an upper bound, NOT a result
footprint split, val (run B)   0.2981   ← selected on the split it reports
footprint split, val (run C)   0.5180   ← same problem, different acquisition mix
footprint split, held-out test 0.1447   ← QUOTE THIS
```

`0.8050 − 0.2981 = 0.5069`, so **roughly 63 % of the first headline was
leakage**: the per-file split let tiles from one satellite acquisition land on
both sides, so the model was validated on ground whose neighbours it had seen.

But the deeper problem was that *every* number above the last line was a **val**
figure, selected on the same split it was reported from. Run C added a genuine
held-out test set and the val estimate collapsed from 0.5180 to **0.1447** — a
gap of 0.3732 from one model at one checkpoint.

And it fails in a diagnosable way. At the checkpoint val liked best, the model was
conservative on val (precision 0.95) and aggressive on test (precision 0.15,
recall 0.76): on held-out acquisitions it emitted **333,079 false-positive pixels
against 59,513 true positives** — six wrong pixels per right one. It does not
transfer to unseen acquisitions; it over-predicts them.

**Quote 0.1447.** Caveat: it rests on only 6 acquisitions, so it is wide. It is
still the only number here that nothing was tuned against, and a wide estimate of
the right quantity beats a tight estimate of the wrong one.

---

## What is NOT verified — read this before claiming anything

- **The held-out estimate is wide.** Run C's test split is **6 acquisitions / 12
  tiles**, so 0.1447 carries a large interval. The val-to-test gap of 0.3732 is
  convincing evidence of a real transfer failure; the precise value is not. Val
  is 8 acquisitions, and epoch-to-epoch val spread (std ≈ 0.107) is larger than
  most effects you would want to detect. **More acquisitions is the single
  highest-value improvement available**, ahead of any modelling work.
- **Nothing is deployed.** `Dockerfile.train` and the compose files exist but
  **have never been built** — no Docker daemon on this machine. The Dockerfile
  says so in its own header.
- **The SAR band order is assumed, not measured.** Band 1 is 12.2 dB brighter
  than band 0, so the stored order is probably `(VH, VV)` while the code labels
  them `(VV, VH)`. Training is unaffected (both channels are used), but any
  *physical* reading of "VV" is unverified. Relabelling would invalidate every
  index, checkpoint and match report. See LIMITATIONS B12.
- **The synthetic generator is a statistical stand-in, not a simulation.** It
  reproduces the bright/dark extremes in coverage and brightness but places them
  as compact patches, not as objects on real tracks. Metrics from it describe the
  generator, not the ocean.
- **`data/sar/` holds 6 COG GeoTIFFs** (Wakashio AOI) and is untracked. The demo
  product ID in the original brief does not exist — there is no Sentinel-1 IW
  pass over Mauritius on 2020-08-07.
- **Ollama has no server and zero models**, so `services/intel` LLM narratives
  are inert. AISStream is terrestrial and returns nothing over open ocean; there
  is no free AIS feed there.

---

## Where things live

```
sentinel_core/     config, logging, errors, datasource + scene grouping   ← start here
ml/synth/          profile real → generate → re-profile → match report
ml/training/       transforms, dataset, splits, metrics, train_detector
services/          api(8000) ingest(8001) detect(8002) drift(8003) attribute intel ui(3000)
scripts/train.py   the documented training entry point
data/index/real/   manifest.jsonl — THE dataset definition (absolute paths)
docs/              RUNBOOK, MODEL_CARD, LIMITATIONS, COMPLETION_REPORT, VERIFICATION
```

Two conventions worth internalising before you edit anything:

- **The manifest defines the dataset**, not the directory listing. `scene_id` must
  keep coming from the manifest — re-deriving it from the path is what silently
  collapsed all 1200 tiles into one "scene" and made the leakage check vacuous.
- **AOIs travel as named axes** (`min_lon/min_lat/max_lon/max_lat`), never a
  `west,south,east,north` string. A lat-first string is four numbers that are all
  legal in either slot, so no validator can catch the swap.

`docs/COMPLETION_REPORT.md` lists the 16 verification gates and, explicitly, what
is *not* claimed. `docs/LIMITATIONS.md` is the honest register — read it before
writing anything user-facing.

---

## Suggested next steps, in value order

1. **Get more acquisitions into val and test.** 8 and 6 is why the held-out
   estimate is wide. Raise `--max-pairs` (the archive holds 195 acquisitions) or
   run uncapped overnight; cost is linear at ~180 s/epoch for 114 train pairs on
   this M2.
2. **Attack the over-prediction.** Run C fails on precision (0.15 on unseen
   acquisitions), not recall (0.76). That points at threshold selection, and at
   augmentation that varies *acquisition conditions* rather than only geometry —
   the current transforms flip, rotate and add speckle, none of which changes the
   incidence angle or wind regime the model appears to be keying on.
3. **Build `Dockerfile.train` once** on a machine with a daemon, then delete the
   "NOT YET BUILD-VERIFIED" paragraph.
4. **Verify the band order** against a known acquisition, which would let the
   polarisation labels be trusted and B12 be closed.
