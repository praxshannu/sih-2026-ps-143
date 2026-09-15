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
| Deterministic detection (no torch) | 10 Aug 2020 real hit: 3.34 km², −9.81 dB, confidence 0.934 |
| Synthetic set matches the real one | `data/synthetic/match_report.json`: **26/28** checks within tolerance, reproducible |
| Lint, format, types | `make lint` / `make format` / `make typecheck` clean across 8 mypy targets |

---

## The one number that matters, and the trap around it

```
per-file split    best val IoU 0.8050   ← an upper bound, NOT a result
footprint split   best val IoU 0.2981   ← the number to quote
```

`0.8050 − 0.2981 = 0.5069`, so **roughly 63 % of the headline was leakage.**
The first split let tiles from one satellite acquisition land in both train and
val, so the model was validated on ground it had already seen.

Three details make it more than arithmetic: train IoU went *up* (0.5922 → 0.7044)
while val went *down*; the two val sets differ in difficulty (15.2 % vs 3.6 % oil);
and the leak-free val plateaus at 0.21–0.30 for seven epochs while train climbs
to 0.70. That plateau is the informative shape.

**If you quote one number, quote 0.2981.** It is also optimistic — see below.

---

## What is NOT verified — read this before claiming anything

- **There is no held-out test set.** The real path runs with `test_fraction=0.0`,
  so `0.2981` is `best_val_iou` over 10 epochs — selected on the split it is
  reported from. Measured selection bias on that run: **+0.0429** (last epoch
  scored 0.2552, best-of-10 scored 0.2981). The val set is only **8 acquisitions
  / 14 tiles**, and the run-to-run spread across epochs (std 0.107) is larger
  than most effects you would want to detect. Closing this is the single highest-
  value piece of work left.
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

1. **Build the held-out test set.** Set `test_fraction` on the real path, re-run
   the footprint split, and report the test IoU as the headline with val demoted
   to model selection. Until then `0.2981` is an upper bound too.
2. **Build `Dockerfile.train` once** on a machine with a daemon, then delete the
   "NOT YET BUILD-VERIFIED" paragraph.
3. **Verify the band order** against a known acquisition, which would let the
   polarisation labels be trusted and B12 be closed.
