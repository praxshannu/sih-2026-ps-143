# SENTINEL — environment-variable guide

**Read this top to bottom. It is written to be followed at 1 a.m. without guessing.**

The instance is built and the images are compiled. The stack has **not** been
started, because `.env` on the instance ships deliberately empty and
`deploy/preflight.sh` refuses to pass while the three required secrets are
unset. That is the gate working, not a failure.

Everything you need to paste is below. Nothing in this document is a secret
value — where a value is a credential, it tells you which line of your local
`.env` to copy it from.

---

## 1. Connect

```bash
ssh -i ~/.ssh/sentinel-key.pem ubuntu@54.173.214.223
```

The key was generated during provisioning and is at `~/.ssh/sentinel-key.pem`
on your Mac, mode `0600`. The Elastic IP is `54.173.214.223` and it survives
the daily stop/start, so this command does not change.

> If SSH is refused, your IP has changed. The security group allows port 22 from
> `49.204.226.158/32` only. Get your current IP with
> `curl -s https://checkip.amazonaws.com`, then in the AWS console add it to the
> `sentinel-sg` inbound rule for port 22.

---

## 2. The file

```
/opt/sentinel/.env
```

It is already there, with all 61 keys present and every value empty.

## 3. Open it

```bash
cd /opt/sentinel
nano .env
```

`nano` is the friendliest choice here. Save with `Ctrl-O`, `Enter`; exit with
`Ctrl-X`. (`vim .env` if you prefer.)

---

## 4. Fill it in

### 4a. Do these three first — the stack will not start without them

All three are placeholders in your local `.env` today. Generate fresh values on
the instance, not on your Mac, and paste each into the key named:

```bash
openssl rand -hex 32     # -> JWT_SECRET
openssl rand -hex 32     # -> DB_PASSWORD
openssl rand -hex 32     # -> SECRET_KEY
```

Then **make `DATABASE_URL` agree with `DB_PASSWORD`** — if these two disagree,
Postgres initialises with one password and the API connects with the other, and
every query fails:

```
DATABASE_URL=postgresql+asyncpg://sentinel:<the DB_PASSWORD you just made>@db:5432/sentinel
```

The host is `db`, never `127.0.0.1`. Inside a container `127.0.0.1` is the
container itself.

> **About `SECRET_KEY`:** nothing in the repository reads it. Only `JWT_SECRET`
> is consumed (`services/api/app/middleware/auth.py`). `SECRET_KEY` is kept
> solely because `deploy/preflight.sh` refuses to pass while it is empty.
> Setting it is harmless — just do not expect it to protect anything.
>
> **About `SENTINEL_SECRET_KEY`:** it is not in this template and you do not
> need it. Nothing reads it; `sentinel_core/tests/test_config.py` sets it
> specifically to assert it does *not* leak into settings.

### 4b. The full table

**Secret** = a credential. Never commit it, never paste it into a chat window.

| # | Key | Secret | Paste | Where the value comes from |
|---|---|---|---|---|
| 1 | `DB_USER` | | `sentinel` | literal — must match the user in `DATABASE_URL` |
| 2 | `DB_PASSWORD` | **yes** | generated above | `openssl rand -hex 32` |
| 3 | `DATABASE_URL` | **yes** | see 4a | built from `DB_PASSWORD` |
| 4 | `REDIS_URL` | | `redis://redis:6379/0` | literal — compose service name |
| 5 | `REDIS_BROKER_URL` | | `redis://redis:6379/1` | literal |
| 6 | `REDIS_RESULT_BACKEND` | | `redis://redis:6379/2` | literal |
| 7 | `ENV` | | `production` | literal |
| 8 | `PORT` | | `8000` | literal |
| 9 | `SENTINEL_API_VERSION` | | `v1` | literal |
| 10 | `JWT_SECRET` | **yes** | generated above | `openssl rand -hex 32` |
| 11 | `JWT_ALGORITHM` | | `HS256` | literal |
| 12 | `JWT_EXPIRE_SECONDS` | | `86400` | literal |
| 13 | `SECRET_KEY` | **yes** | generated above | satisfies preflight; unread by code |
| 14 | `CORS_ORIGINS` | | `http://54.173.214.223` | the public origin. Change if you add a domain |
| 15 | `RESULT_TTL_SECONDS` | | `86400` | literal |
| 16 | `INGEST_SERVICE_URL` | | `http://sentinel-ingest:8001` | literal — compose overrides it anyway |
| 17 | `DETECT_SERVICE_URL` | | `http://sentinel-detect:8002` | literal |
| 18 | `DRIFT_SERVICE_URL` | | `http://sentinel-drift:8003` | literal |
| 19 | `ATTRIBUTE_SERVICE_URL` | | `http://sentinel-attribute:8004` | literal |
| 20 | `INTEL_SERVICE_URL` | | `http://sentinel-intel:8005` | literal — service not deployed |
| 21 | `ARCHIVE_PROXY_TIMEOUT` | | `120` | seconds |
| 22 | `DETECT_PROXY_TIMEOUT` | | `300` | seconds |
| 23 | `DETECT_UPLOAD_TIMEOUT` | | `900` | seconds |
| 24 | `DRIFT_PROXY_TIMEOUT` | | `600` | seconds — a 64-member ensemble is slow |
| 25 | `SENTINEL_DATA_DIR` | | `/app/data` | the compose mount point |
| 26 | `SENTINEL1_STORAGE` | | `/app/data/sar` | holds the 6 Wakashio scenes |
| 27 | `OCEAN_STORAGE` | | `/app/data/ocean` | scratch |
| 28 | `WIND_STORAGE` | | `/app/data/wind` | scratch |
| 29 | `AIS_PARQUET_PATH` | | *(leave empty)* | empty on purpose — resolves to nothing rather than a stale file |
| 30 | `CDSE_CLIENT_ID` | **yes** | copy from local `.env` **line 83** | `~/projects/claude1/sentinel/.env` |
| 31 | `CDSE_CLIENT_SECRET` | **yes** | copy from local `.env` **line 84** | same file |
| 32 | `CDSE_TOKEN_URL` | | `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token` | literal |
| 33 | `CDSE_CATALOG_URL` | | `https://catalogue.dataspace.copernicus.eu/odata/v1` | literal |
| 34 | `CDSE_PROCESS_URL` | | `https://sh.dataspace.copernicus.eu/api/v1/process` | literal |
| 35 | `CDSAPI_URL` | | `https://cds.climate.copernicus.eu/api` | literal |
| 36 | `CDSAPI_KEY` | **yes** | copy from local `.env` **line 112** | same file |
| 37 | `COPERNICUSMARINE_SERVICE_USERNAME` | **yes** | copy from local `.env` **line 101** | same file |
| 38 | `COPERNICUSMARINE_SERVICE_PASSWORD` | **yes** | copy from local `.env` **line 102** | same file |
| 39 | `CMEMS_DATASET_PATH` | | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | literal |
| 40 | `COPERNICUS_MARINE_PRODUCT_ID` | | `GLOBAL_MULTIYEAR_PHY_001_030` | literal |
| 41 | `COPERNICUS_MARINE_VARIABLES` | | `uo,vo` | literal |
| 42 | `GFS_THREDDS_URL` | | `https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl` | literal |
| 43 | `GFS_OPENDAP_URL` | | *(leave empty)* | dead — NOAA retired OPeNDAP |
| 44 | `AISHUB_API_KEY` | **yes** | your AISHub key | **not in your local `.env`** — see 4c |
| 45 | `ALLOW_SYNTHETIC_FORCING` | | `false` | **must be false** |
| 46 | `ALLOW_SYNTHETIC_AIS` | | `false` | **must be false** |
| 47 | `SENTINEL_DEMO_MODE` | | `false` | **must be false** |
| 48 | `ALLOW_SYNTHETIC_TRAINING` | | `false` | **must be false** |
| 49 | `SENTINEL_DATASOURCES__ALLOW_SYNTHETIC_FALLBACK` | | `false` | **must be false** |
| 50 | `INGEST_INTERVAL_HOURS` | | `1` | literal |
| 51 | `MODEL_CHECKPOINT` | | `/app/models/detector_best.pth` | compose sets this explicitly |
| 52 | `MODEL_NAME` | | `unetpp_scse_resnet34` | literal |
| 53 | `ENCODER_NAME` | | `resnet34` | compose sets this explicitly |
| 54 | `IN_CHANNELS` | | `2` | **must be 2** — the checkpoint is 2-band |
| 55 | `NUM_CLASSES` | | `1` | compose sets this explicitly |
| 56 | `DEVICE` | | `cpu` | compose sets this explicitly |
| 57 | `ATTRIBUTION_SEARCH_RADIUS_NM` | | `50` | literal |
| 58 | `ATTRIBUTION_TIME_WINDOW_HOURS` | | `6` | literal |
| 59 | `ATTRIBUTION_FUZZY_WEIGHT` | | `0.6` | literal |
| 60 | `ATTRIBUTION_XGB_WEIGHT` | | `0.4` | literal |
| 61 | `ATTRIBUTION_XGB_PATH` | | *(leave empty)* | empty on purpose |

**On keys 51, 53, 54, 55, 56** — `docker-compose.aws.yml` pins these in the
`detect` service's `environment:` block, which overrides `env_file`. You may
leave them blank and the container still gets the right values. They are listed
because a native (non-Docker) run reads them from here.

### 4c. Two things the brief flagged that turned out differently

**`AIS_API_KEY` is defined twice — but you can ignore it.** In your local
`.env` it appears on **line 29 (empty)** and **line 91 (populated)**. The brief
is right that a duplicate is a coin flip. But it is a coin flip over a key that
**no code reads** — not in any service, script or test. The key the ingest
service actually reads is `AISHUB_API_KEY`, and that one is **absent from your
local `.env` entirely**. So:

- do **not** copy either `AIS_API_KEY` line across;
- get an AISHub key from aishub.net, or leave `AISHUB_API_KEY` empty and accept
  that the AIS overlay has no live feed.

**The older `CMEMS_USER` / `CMEMS_PASS` pair (lines 26–27) is retired.** The
client reads `COPERNICUSMARINE_SERVICE_USERNAME` / `_PASSWORD` (lines 101–102).
Your local file populates **both** naming schemes; only the new pair is live.

### 4d. Resolving a secret's line number yourself, safely

This yields the line number and **nothing else**. Use it, not a bare `grep`:

```bash
grep -n '^CDSAPI_KEY=' ~/projects/claude1/sentinel/.env | cut -d: -f1
```

This is the mistake to avoid — it prints the secret into your terminal scrollback
and, if you are in a chat session, into a transcript that cannot be recalled:

```bash
grep -nE '^(DB_PASSWORD|CDSAPI_KEY)=' ~/projects/claude1/sentinel/.env   # DO NOT RUN
```

---

## 5. Start the stack

```bash
cd /opt/sentinel

# Preflight should now be green. If it is not, stop and read the output.
bash deploy/preflight.sh

docker compose -f docker-compose.aws.yml up -d
```

Give `sentinel-detect` **up to four minutes** on first start: importing torch on
2 vCPU plus reading a 283 MB checkpoint is slow, and its `start_period` is 240 s
for that reason. Do not restart it during that window.

---

## 6. Confirm it worked

```bash
cd /opt/sentinel
DC="docker compose -f docker-compose.aws.yml"

# 1. Every row must say (healthy)
$DC ps --format 'table {{.Service}}\t{{.Status}}'

# 2. The trained model actually loaded, on CPU.  <- the assertion that matters
$DC logs sentinel-detect | grep -i -E 'Loading model|Checkpoint not found|device'
#    want: "Loading model from /app/models/detector_best.pth" + "Using device: cpu"

# 3. Only the UI is published, and only on loopback.  expect: 0
docker ps --format '{{.Names}}\t{{.Ports}}' | grep -c '0.0.0.0'

# 4. The synthetic gates are shut.  every one must read false
$DC exec sentinel-api env | grep -E 'ALLOW_SYNTHETIC|DEMO_MODE'

# 5. The API answers through the UI's nginx proxy — this is the path the
#    browser actually takes, so it proves more than hitting :8000 directly.
curl -s localhost:3000/health
curl -s localhost:3000/api/v1/cases | head -c 300
```

Then from **your Mac**:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://54.173.214.223/
```

**If step 2 prints `Checkpoint not found ... loading pretrained encoder only`,
stop.** The service will answer `/health` with 200 while emitting noise from an
untrained decoder. Fix the mount; do not demo it.

---

## 7. If a service is unhealthy

Work down this list before restarting anything:

```bash
DC="docker compose -f docker-compose.aws.yml"

$DC ps                                   # which one, and what state
$DC logs --tail=80 <service>             # the actual error
$DC logs sentinel-detect | tail -40
df -h / && docker system df              # a full disk takes Postgres down first
free -h                                  # OOM?
```

The three failures worth recognising:

| Symptom | Cause | Fix |
|---|---|---|
| `sentinel-api` healthy, every query 500s | `DB_PASSWORD` and the password inside `DATABASE_URL` disagree | make them match, then `$DC up -d --force-recreate sentinel-api` |
| `sentinel-detect` crash-loops on `size mismatch for encoder.conv1.weight` | `IN_CHANNELS` is not 2 | it is pinned in the compose file — do not remove that line |
| `sentinel-drift` healthy, forecasts error | expected — `cfgrib` and the OpenDrift stack are not installed | see the note below |

**`drift` is expected to fail on its forecast routes.** Its `/health` returns
200 because every `opendrift` / `copernicusmarine` / `cdsapi` / `cfgrib` import
is lazy and guarded, so the container starts and the route fails with a typed
error. This is documented behaviour (`docs/DEPLOYMENT_FREE_TIER.md` §13.4), not
a deployment mistake. Wind is also required — without `CDSAPI_KEY` the service
fails closed rather than substituting synthetic wind, which is correct.

A container that will not come up at all:

```bash
$DC up -d --force-recreate <service>     # then re-check, do not loop on this
```

---

## 8. Changing a value later

`env_file` is read at container **create** time, not at run time. Editing `.env`
and running `$DC restart` does nothing. Use:

```bash
$DC up -d --force-recreate <service>     # or: $DC up -d  (recreates changed ones)
```

---

## 9. Keys that are deliberately absent

Do not add these — nothing consumes them, and a key that nothing reads is worse
than no key, because it makes a missing integration look configured:

`SENTINEL_SECRET_KEY` · `CMEMS_USER` · `CMEMS_PASS` · `AIS_API_KEY` ·
`AIS_BASE_URL` · `MARINE_API_KEY` · `DETECT_TILE_SIZE` · `DETECT_OVERLAP` ·
`DETECT_BATCH_SIZE` · `DRIFT_PARTICLE_COUNT` · `DRIFT_FORECAST_HOURS` ·
`DRIFT_ORIGIN_UNCERTAINTY_MINUTES` · `DARK_VESSEL_SAR_ENABLED` ·
`INGEST_POLL_INTERVAL_MINUTES` · `MODEL_WEIGHTS_PATH` · `IMAGE_SIZE` ·
`TRAIN_EPOCHS` · `TRAIN_BATCH_SIZE` · `LR` · `SYNTHETIC_DATA_DIR` ·
`CDSE_WMS_INSTANCE_ID` · `GRAFANA_PASSWORD` · `PROMETHEUS_PORT` · `UI_PORT` ·
`API_HOST` · `API_PORT` · `DB_PORT` · `REDIS_PORT` · and the entire
`SENTINEL_TRAINING__*` / `SENTINEL_PATHS__*` / `SENTINEL_LOGGING__*` block.

The last group are real settings on `sentinel_core.config`, but **no deployed
service imports `sentinel_core`** — only `ml/` and `scripts/` do, and neither
runs on this box.
