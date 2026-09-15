# SENTINEL — Runbook

Operational guide for running, verifying and debugging SENTINEL on an Apple
Silicon laptop. Written for the machine it was built on (M2, 16 GB, no CUDA),
because that is the machine whose constraints shaped every decision here.

---

## 0. TL;DR

```bash
cd /Users/praxsmac/projects/claude1/sentinel
PY=/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/python

cp .env.example .env          # then fill CDSE_*, CDSAPI_*, COPERNICUSMARINE_*
$PY -m pytest services/ ml/ sentinel_core/ -q
make dev-lite                 # 8 containers, ~8.4 GB
open http://localhost:3000    # Scene Intake is at /intake

# train the detector on either data source, switchable at runtime (§5)
$PY scripts/train.py --data-source synthetic --epochs 10
$PY scripts/train.py --data-source real --epochs 3 --max-scenes 24 --max-steps-per-epoch 1
```

---

## 1. Interpreter and environment

Everything runs against one managed virtualenv:

```bash
PY=/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/python
```

Installed and verified: `rasterio`, `xarray`, `netCDF4`, `cfgrib`,
`copernicusmarine` 2.4.1, `opendrift` 1.14.11, `torch` 2.14 (MPS available),
`timm`, `fastapi`, `httpx`, `pydantic` v2, `asyncpg`, `pytest`, `ruff`, `mypy`.

**Python version.** Deployment targets 3.11 (`services/detect` and
`services/drift` build on `python:3.11-slim`; `services/api` on 3.12). The
development interpreter is 3.13, and `pyproject.toml` tells mypy to type-check
under 3.13 — see the comment there for why. The 3.11 target is validated by
building the images, not by mypy's parse target.

**`ps` is blocked** by the sandbox on this machine. To see what is alive:

```bash
pgrep -fl uvicorn
curl -s localhost:8000/health | $PY -m json.tool
```

---

## 2. Verification gates

Run these before believing anything:

```bash
cd /Users/praxsmac/projects/claude1/sentinel
PY=/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/python

# 1. Tests — expect "277 passed"
$PY -m pytest services/ ml/ sentinel_core/ -q

# 2. Lint and format — expect "All checks passed!" and "141 files already formatted"
$PY -m ruff check services/ ml/ scripts/ sentinel_core/ conftest.py
$PY -m ruff format --check services/ ml/ scripts/ sentinel_core/ conftest.py

# 3. Types — expect "Success: no issues found" for each
for s in api ingest detect drift attribute intel; do echo "--- $s"; $PY -m mypy services/$s; done
$PY -m mypy ml/
$PY -m mypy sentinel_core/

# 4. Compose — both must print a resolved config
docker compose -f docker-compose.txt config >/dev/null && echo "base OK"
docker compose -f docker-compose.lite.yml config >/dev/null && echo "lite OK"

# 5. Fresh-clone reproducibility — expect "110 passed, 5 skipped"
rm -rf /tmp/sentinel_clone && git clone -q . /tmp/sentinel_clone
cd /tmp/sentinel_clone && $PY -m pytest services/ ml/ -q
```

The 5 skips are the real-scene tests, which need `data/sar/` (gitignored). They
skip **with a stated reason**. A skip means "the scenes are not here", never
"it passed".

### 2.1 `tmp_path` and the sandbox

`conftest.py` points pytest's temp root at a **unique per-session** directory
inside the workspace. This is not cosmetic: the sandbox broker rejects `mkdir`
inside the system temp root and raises `PermissionError("EEXIST")` rather than
`FileExistsError`. `Path.mkdir(exist_ok=True)` only suppresses
`FileExistsError`, so once `pytest-of-<user>` existed from a previous run, every
`tmp_path` test errored during setup — permanently, and looking exactly like a
broken test suite.

Override the location with `SENTINEL_PYTEST_TMP` if you want a tmpfs.

**The sandbox can also kill a run mid-collection.** This environment wraps
deletions in a bulk-delete guard that counts *cumulative* deletions per turn and
raises `SystemExit(1)` when the count crosses a threshold. A pytest session
under `.pytest_tmp/` deletes enough files to cross it, and because the guard
fires inside a fixture rather than at exit, the symptom is a scattering of
failures and errors that have nothing to do with the code — and the summary line
is lost, so it reads as "the suite is broken".

```bash
# If you see unexplained F/E with no summary, move the tmp root out of the tree:
SENTINEL_PYTEST_TMP=/tmp/sentinel_pytest $PY -m pytest services/ ml/ sentinel_core/ -q
```

277 passed is the real number. A run that reports failures *and* prints a
`[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED]` line has been interfered with;
re-run it before believing anything.

---

## 3. Running the stack

### 3.1 Native (recommended on this machine)

Seven uvicorn workers cost ~450 MB total. The *containers* are what exhaust
16 GB once Postgres, Redis, Celery, Prometheus and Grafana are added.

```bash
cd /Users/praxsmac/projects/claude1/sentinel
set -a; . ./.env; set +a      # WITHOUT THIS, ERA5 silently loses its key

cd services/ingest  && $PY -m uvicorn app.main:app               --port 8001 &
cd services/detect  && $PY -m uvicorn app.main_det:app           --port 8002 &
cd services/drift   && $PY -m uvicorn app.main_attribution:app   --port 8003 &
cd services/attribute && $PY -m uvicorn app.main:app             --port 8004 &
cd services/api     && $PY -m uvicorn app.main:app               --port 8000 &
cd services/ui      && npm run dev                                # :3000
```

Two things to know:

* **Load `.env` first.** Starting uvicorn by hand without it drops every key and
  ERA5 degrades to synthetic wind — the run *succeeds and returns invented
  numbers*. `services/drift/app/env_bootstrap.py` mitigates this (it walks up for
  the repo-root `.env` and logs loudly when `CDSAPI_KEY` is absent), but do not
  rely on it.
* **`.env` must hold `127.0.0.1` service URLs**, not `sentinel-*`. Natively,
  `sentinel-*` does not resolve and every gateway proxy returns an opaque
  500/502. `docker-compose.lite.yml` overrides them to `sentinel-*` for compose.

Entry points matter:
* `detect` → `app.main_det:app` (torch-free Tier-A). `app.main:app` needs torch
  and a trained checkpoint.
* `drift` → `app.main_attribution:app` (verified OpenDrift/OpenOil path).
  `app.main:app` is the older SDE variant.

### 3.2 Docker, lightweight profile

```bash
make dev-lite      # docker compose -f docker-compose.lite.yml up --build
```

Eight services: `db`, `redis`, `api`, `ingest`, `detect`, `drift`, `attribute`,
`ui`. No Prometheus, Grafana, Celery worker or intel service. Memory caps total
~8.4 GB.

`docker-compose.lite.yml` is **standalone, not an override**. Compose overrides
can add or change services but cannot remove them, so layering it on
`docker-compose.txt` would still start all four of the things it exists to drop.

Every service has a real HTTP healthcheck, so a broken start names the service
instead of leaving a silent gap:

```bash
docker compose -f docker-compose.lite.yml ps        # look at the health column
docker compose -f docker-compose.lite.yml logs -f sentinel-detect
```

### 3.3 Full stack

```bash
make dev           # docker compose -f docker-compose.txt -f docker-compose.override.yml up --build
```

**Not verified end to end on this machine.** `config` validates; the images have
not been built and run in this pass — they are large and the disk budget is
tight. See `LIMITATIONS.md`.

---

## 4. The demo, step by step

### 4.1 Real Sentinel-1 inference (the headline)

```bash
cd /Users/praxsmac/projects/claude1/sentinel/services/detect
$PY -c "
import sys, json; sys.path.insert(0,'.')
from pathlib import Path
from app.pipeline import DetectPipeline, PipelineOptions
r = DetectPipeline().run(Path('../../data/sar/wakashio_20200810_peak.tif'), PipelineOptions(with_evidence=True))
print('state  ', r['state'])
print('count  ', r['count'], 'flags', r['flags'])
d = r['detections'][0]
print('area   ', round(d['area_km2'], 3), 'km2')
print('centroid', d['centroid'])
print('conf   ', d['confidence'])
"
```

Expected:

```
state   ok
count   11 flags ['missing_wind_forcing']
area    3.347 km2
centroid [57.72945073545187, -20.416901124019486]
conf    {'value': 0.934, 'low': 0.902, 'high': 0.956, 'interval': 'wilson_95', ...}
```

Pass `wind_speed_ms=5.0` to `PipelineOptions` to move the detections out of
`LOW_CONFIDENCE_NO_WIND` — 5 m/s sits inside the 2–10 m/s band where slicks are
separable from look-alikes.

### 4.2 Real drift

```bash
cd /Users/praxsmac/projects/claude1/sentinel/services/drift
set -a; . ../../.env; set +a
export ECCODES_DIR=/opt/homebrew/Cellar/eccodes/2.48.0     # cfgrib needs this
$PY -c "
import sys, json; sys.path.insert(0,'.')
from fastapi.testclient import TestClient
from app.main_attribution import app
c = TestClient(app)
print(json.dumps(c.get('/drift/health').json())[:300])
r = c.post('/drift/attribution', json={
  'detection_lon': 57.7295, 'detection_lat': -20.4169,
  'detection_area_km2': 3.347, 'detection_time': '2020-08-10T01:37:30Z',
  'duration_h': 6, 'n_members': 32, 'use_era5': True, 'use_cmems': True, 'forcing': 'era5'})
print('HTTP', r.status_code)
d = r.json()
print('wind', d['wind_source'], '| current', d['current_source'])
print('origin', d['origin']['center_lon'], d['origin']['center_lat'], 'n=', d['origin']['n_particles'])
print('notes', d['notes'])
"
```

Expected: `HTTP 200`, `wind era5 | current cmems`, ~32/32 particles, and a note
explaining that `∇·K = 0` because `K` is spatially constant — the WMC correction
is applied and evaluates to a no-op, which is stated rather than faked.

You may see a `KeyError: '/mp-…'` traceback from
`multiprocessing/resource_tracker.py` **after** the result prints. That is a
known CPython 3.13 shutdown issue in the resource tracker, unrelated to this
code; the result is already out.

### 4.3 The analyst UI

```
http://localhost:3000/intake
```

1. Drop a `.tif` (or pick one from the on-disk list) — validation runs first.
2. Read the **state banner**: `DETECTION`, `NO DETECTION`, `LOW CONFIDENCE`,
   `OUT OF DISTRIBUTION`, `FORCING MISSING` or `INVALID SCENE`. These are six
   different answers and the banner gives each its own treatment.
3. Inspect the detection table: area, length/width, orientation, contrast, and a
   confidence bar that draws the Wilson interval as the track and the point
   estimate as the fill — the gap between them *is* the uncertainty.
4. Read the explanation panel. It is labelled `not grad-cam` and quotes the
   reason: there is no trained checkpoint, so no gradient exists to weight.
5. Backtrack 24 h, then forecast 48 h. Vessel evidence appears only after the
   hindcast, because scoring depends on the inferred origin.
6. Reopen any scene with a cached result from the list — no re-inference.

---

## 5. Training the detector

### 5.1 The in-place contract

The real archive is **read where it lies** and is never copied to the local
machine. That is enforced, not promised:

- The real index (`data/index/real/manifest.jsonl`) stores **absolute** paths
  into `/Volumes/Ventoy/Oil`. Nothing is staged.
- `SourceFingerprint` records each source directory's `mtime_ns`, entry count,
  `st_dev` and inode at the start of the run and re-checks it at the end.
  `source_unchanged` in `run_report.json` is that check.
- `assert_outputs_are_outside_the_source` refuses a checkpoint, run or log path
  inside a source directory before the first batch.

To prove it yourself, fingerprint the archive either side of a run:

```bash
PY=/Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/python
$PY - <<'EOF'
import hashlib
from pathlib import Path
src = Path("/Volumes/Ventoy/Oil")
h = hashlib.sha256()
for p in sorted(src.iterdir()):
    st = p.stat()
    h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
print(len(list(src.iterdir())), h.hexdigest(), src.stat().st_mtime_ns)
EOF
```

Run it before and after training; all three numbers must be identical. On the
1200-scene archive that is `2400 69d35c71…ac4 1682009727000000000`.

### 5.2 Running

```bash
cd /Users/praxsmac/projects/claude1/sentinel
set -a; . ./.env; set +a
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR   # 268 ms -> 1.8 ms per open

# Real archive, read in place. Bounded so it finishes.
$PY scripts/train.py --data-source real --epochs 3 --max-scenes 24 \
    --max-steps-per-epoch 1 --image-size 512

# The matched synthetic set.
$PY scripts/train.py --data-source synthetic --epochs 10

# Switch sources at runtime without editing anything:
$PY scripts/train.py --data-source synthetic --run-name ab-synthetic
$PY scripts/train.py --data-source real      --run-name ab-real
```

`--data-source` overrides `SENTINEL_DATA_SOURCE`; both override nothing else.
`--max-steps-per-epoch` is what makes a smoke run cheap — the epoch still
validates, so the whole pipeline is exercised.

**Exit codes**: `0` ok, `2` bad arguments, `78` configuration error,
`1` runtime failure, `130` interrupted (the last checkpoint is complete and
resumable with `--resume`).

### 5.3 What a good run looks like

```
data source: 1200 pair(s), 2 band(s), images=/Volumes/Ventoy/Oil in_place=True
device: mps (auto: Apple MPS available, no CUDA)
splits: train=19 val=5 test=0
model: UNet++/resnet34 in_channels=2 params=24,719,654 trainable=24,719,654
source unchanged: True
■ real | best val IoU 0.0975 at epoch 1 | Trained on real SAR scenes read in place…
```

Then read the report, not the log line:

```bash
$PY -m json.tool runs/<run_id>/run_report.json | head -40
```

`claim` states in words what the numbers are allowed to mean, and
`scientifically_valid` is false whenever they are not a scientific result.

### 5.4 The training image

```bash
docker build -f Dockerfile.train -t sentinel-train .
docker run --rm -v /Volumes/Ventoy/Oil:/data/real:ro -v "$PWD/runs:/app/runs" \
    -e SENTINEL_DATASOURCES__REAL_IMAGES_DIR=/data/real \
    sentinel-train --data-source real --epochs 3 --max-scenes 24
```

**This image has never been built** — Docker is not installed on the machine it
was written on. The COPY sources and the dependency set were checked by hand;
the build is not verified. The Dockerfile says so at the top.

One thing to know before relying on it: the encoder weights come from the
Hugging Face hub, so a machine with no network cannot train with
`encoder_weights=imagenet`. The image bakes them at build time for that reason.
On a machine without them, pass `--encoder-weights none` — the trainer says so
itself when the build fails.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `configured: false` on `/archive/*` | `.env` not loaded | `set -a; . ./.env; set +a` before starting uvicorn |
| Gateway returns opaque 500/502 | `.env` has `sentinel-*` URLs while running natively | use `127.0.0.1` in `.env`; compose overrides handle the container case |
| `Missing variables: ['x_wind','y_wind',…]` from a healthy ERA5 reader | `FORCING_PAD_HOURS` off-by-one, or a daily CMEMS window with `start == end` | already fixed; if it recurs, check the padding and the ±1 day CMEMS pad |
| `CoordinatesOutOfDatasetBounds` | depth below the grid minimum | request `[0.4941, 1.0]`, not `[0, …]` |
| `AttributeError` on `.load()` from copernicusmarine | 2.x `subset()` returns a receipt | use `open_dataset()` |
| Drift refuses with `Readers must be added for ['x_wind','y_wind']` | OpenOil requires wind | supply ERA5 or GFS; it will not run on currents alone |
| `GFS not usable: … retains ~10 days` | historical window | expected — use ERA5 |
| `PermissionError: EEXIST … pytest-of-…` | system temp root unusable | already handled by `conftest.py`; check `SENTINEL_PYTEST_TMP` |
| `ModuleNotFoundError: No module named 'timm'` | torch present, timm absent | `pip install timm`; only the UNet++ path needs it |
| detect tests all skip | `data/sar/` absent (gitignored) | expected on a fresh clone; re-run where the scenes exist |
| `eccodes` error inside the drift container | arm64 Linux lacks `libeccodes0` | add it to the drift Dockerfile, or run drift natively (see §4.2) |

### 6.1 Reading a provenance state

* `■ REAL` — checksummed from the named upstream. Trust it to the extent the
  source deserves.
* `▲ SYNTHETIC` — generated and fenced. **Never evidence.** Do not put it in a
  report as fact.
* `□ UNAVAILABLE` — the source exists but has no coverage here, and the reason
  is named. This is an honest negative, not a failure.
* `✕ FAILED` — an upstream error. The reason code is machine-readable.

---

## 7. Maintenance

```bash
make test          # pytest services/ ml/ sentinel_core/
make lint          # ruff check + format --check
make typecheck     # mypy per service (never `mypy services/` — see pyproject)
make test-all      # all three
make clean         # docker down -v --rmi local; rm -rf data/
```

**Never commit `.env`.** It is gitignored and holds live Copernicus credentials.
`.gitignore` anchors data at the repo root (`/data/`) — a bare `data/` matches at
any depth and once silently swallowed 11 drift *source* modules. Do not un-anchor
it.
