# SENTINEL — AWS Deployment Plan

Project-level requirements and three AWS deployment tiers (minimum / mid-range /
maximum) with monthly cost estimates.

> **Read first.** The compose files and per-service Dockerfiles have not been
> build-verified on this machine (Docker daemon is absent — `HANDOVER.md` §3.2,
> `RUNBOOK.md` §3.3). All sizing below is derived from the verified resource
> caps in `docker-compose.lite.yml` and the verified native profile, both of
> which actually ran. Treat the `docker-compose.txt` (full) stack as the upper
> bound on memory, not a guarantee that the full image set builds clean.

---

## 1. Project requirements

### 1.1 Service inventory

The full stack is **12 long-running processes + 3 infra sidecars** (full
profile) or **8 services** (lite profile). The API service (`sentinel-api`,
port 8000) is the only thing the UI talks to; everything else is reached via
service-name DNS inside the compose network.

| Service               | Port  | Lang     | Image base (Dockerfile) | Memory cap (lite) | Purpose                                                  |
|-----------------------|-------|----------|-------------------------|-------------------|----------------------------------------------------------|
| `sentinel-api`        | 8000  | Python 3.12 | `python:3.12-slim`   | 768 MB            | REST + WebSocket gateway; the only thing the UI calls    |
| `sentinel-ingest`     | 8001  | Python 3.11 | `python:3.11-slim`   | 768 MB            | SAR/AIS/ocean data ingestion (CDSE, CDS, CMEMS, AIS)     |
| `sentinel-detect`     | 8002  | Python 3.11 | `python:3.11-slim`   | 3 GB              | UNet++ SCSE segmentation (torch + timm, tiled inference) |
| `sentinel-drift`      | 8003  | Python 3.11 | `python:3.11-slim`   | 2 GB              | Lagrangian drift (OpenDrift/OpenOil, ERA5 + CMEMS)       |
| `sentinel-attribute`  | 8004  | Python 3.11 | `python:3.11-slim`   | 512 MB            | Vessel attribution (K_ij fuzzy + XGBoost)                |
| `sentinel-intel`      | 8005  | Python 3.11 | `python:3.11-slim`   | 512 MB            | LLM case narrative (Ollama or OpenAI)                    |
| `sentinel-worker`     | -     | Python 3.12 | same as api          | 768 MB            | Celery async task worker (queue: celery/pipeline/detect/drift/attribute/intel) |
| `sentinel-ui`         | 3000  | Node 20 (build) → nginx-alpine | multi-stage | 128 MB            | React + deck.gl + MapLibre globe, served by nginx |
| `sentinel-db`         | 5432  | -        | `postgis/postgis:15-3.4` | 1 GB           | PostgreSQL 15 + PostGIS 3.4                              |
| `sentinel-redis`      | 6379  | -        | `redis:7-alpine`        | 256 MB         | Broker + result backend (Celery), cache                  |
| `prometheus`          | 9090  | -        | `prom/prometheus:v2.52.0` | -           | `/metrics` scrape every 15 s                             |
| `grafana`             | 3000→3001 | -     | `grafana/grafana:10.4.2`  | -           | Dashboards (admin/admin by default)                      |

**Two entry points the Dockerfile pins but the compose overrides can swap:**
- `services/detect/Dockerfile` defaults to `app.main:app` (UNet++ stack,
  needs torch). The lite profile overrides to `app.main_det:app` (torch-free
  Tier-A operator).
- `services/drift/Dockerfile` defaults to `app.main:app` (SDE). The lite
  profile overrides to `app.main_attribution:app` (verified OpenDrift path).

Deploy whichever matches what you have trained. A production tier-1 deployment
that serves only the deterministic detector does **not** need the torch image.

### 1.2 Runtime dependencies

#### Python packages (per service — all pinned in `services/<name>/requirements.txt`)

| Service       | Key deps (versions matter) |
|---------------|----------------------------|
| api           | fastapi, uvicorn[standard], pydantic 2.7, httpx, asyncpg, sqlalchemy 2.0.30, geoalchemy2 0.15.1, celery[redis] 5.3, redis 5.0, python-jose, python-multipart, loguru |
| ingest        | same + apscheduler 3.10, rasterio 1.3.10, scipy, xarray 2024.6, dask 2024.6 |
| detect        | **torch 2.3.0 + CUDA/timm 0.9.7 + segmentation-models-pytorch 0.3.4 + opencv-python-headless + scikit-image 0.22** |
| drift         | xarray, dask[complete], **netCDF4 1.6**, xgboost 2.0, scikit-learn 1.3 (+ needs `libeccodes0` system lib for `cfgrib`) |
| attribute     | duckdb 0.10, shapely 2.0, pyogrio, geopandas 1.0 |
| intel         | **openai 1.30**, weasyprint 62.3, pydyf, Pillow 10.4, qrcode, jinja2 |

#### Training image (`Dockerfile.train`, Python 3.11-slim)

```
torch>=2.2, torchvision>=0.17, timm>=1.0,
numpy>=1.26, scipy>=1.11, rasterio>=1.3,
pydantic>=2.7, pydantic-settings>=2.3,
loguru>=0.7, python-dotenv>=1.0
```
plus the system library `libexpat1` (rasterio's manylinux wheel).

#### Frontend (`services/ui/package.json`, Node 20-alpine build → nginx)

React 18.3, **deck.gl 9.1 + MapLibre GL 4.7** (3D globe is the hero — runs on
WebGL, not server CPU), axios 1.7, framer-motion 11, recharts 2.14, zustand 5,
react-router-dom 6.28. TypeScript strict, vite 6 build.

#### System libraries the Dockerfiles install

- `libexpat1` (detect, train) — rasterio GDAL runtime
- `libeccodes0` (drift) — needed by `cfgrib` to read ERA5 GRIB. **Not** in the
  current drift Dockerfile; install it on the image or run drift natively
  (`RUNBOOK.md` §6).

### 1.3 Environment variables

Three classes:

#### Class A — required to start the API at all

| Var                       | Default (`.env.example`)                  | Purpose |
|---------------------------|-------------------------------------------|---------|
| `DATABASE_URL`            | `postgresql+asyncpg://sentinel:sentinel_secret@127.0.0.1:5432/sentinel` | Asyncpg DSN |
| `REDIS_URL`               | `redis://127.0.0.1:6379/0`                | Cache |
| `REDIS_BROKER_URL`        | `redis://127.0.0.1:6379/1`                | Celery broker |
| `REDIS_RESULT_BACKEND`    | `redis://127.0.0.1:6379/2`                | Celery results |
| `SECRET_KEY`, `JWT_SECRET`, `JWT_ALGORITHM`, `JWT_EXPIRE_SECONDS` | — | Auth (generate per-env with `openssl rand -hex 32`) |
| `CORS_ORIGINS`            | `http://localhost:3000,http://localhost:5173` | UI origin list |

#### Class B — required for the demo path (without them ingest drifts to silent failure)

| Var                                | Required for | Source |
|------------------------------------|--------------|--------|
| `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` | SAR archive browser + GeoTIFF ingest | Copernicus Data Space Ecosystem (OAuth2) |
| `CDSE_TOKEN_URL`, `CDSE_CATALOG_URL`, `CDSE_PROCESS_URL` | Sentinel-1 | sane defaults in `.env.example` |
| `CDSAPI_URL`, `CDSAPI_KEY`         | ERA5 winds | CDS — accept the licence on the account |
| `COPERNICUSMARINE_SERVICE_USERNAME`, `_PASSWORD`, `CMEMS_DATASET_PATH` | GLORYS12 currents | Copernicus Marine |
| `COPERNICUS_MARINE_PRODUCT_ID`, `_VARIABLES` | CMEMS | sane defaults |
| `GFS_THREDDS_URL`                  | live-operations fallback | NOAA NOMADS server-side subsetting |
| `AIS_API_KEY` (AISStream), `AISHUB_API_KEY` | live AIS | terrestrial-only |
| `LLM_PROVIDER` + `OLLAMA_BASE_URL`/`OPENAI_API_KEY` | case narratives | local Ollama or hosted OpenAI |

> **Without CDS keys, drift fails closed with a typed error** (`docs/
> DATA_PROVENANCE.md`). It does *not* silently substitute synthetic wind — that
> `verification §9.2` incident is what the gate was built for.

#### Class C — synthetic / demo gates (default OFF; typo = FALSE)

`ALLOW_SYNTHETIC_FORCING`, `ALLOW_SYNTHETIC_AIS`, `SENTINEL_DEMO_MODE`,
`ALLOW_SYNTHETIC_TRAINING`, `SENTINEL_DATASOURCES__ALLOW_SYNTHETIC_FALLBACK`.
**Never enable these in prod.** Off means fail loud, on means labelled
substitution; an unset or misspelled value must never open a gate.

#### Class D — operational knobs (sensible defaults)

Ports (`API_PORT=8000`, `UI_PORT=3000`, `PROMETHEUS_PORT=9090`, etc.),
inter-service URLs (`INGEST_SERVICE_URL`, `DETECT_SERVICE_URL`, …),
proxy timeouts (`DRIFT_PROXY_TIMEOUT=600`, `DETECT_PROXY_TIMEOUT=300`,
`DETECT_UPLOAD_TIMEOUT=900`, `ARCHIVE_PROXY_TIMEOUT=120`),
training defaults (`SENTINEL_TRAINING__EPOCHS`, `__IMAGE_SIZE=512`,
`__SCENE_GROUPING=footprint`), drift/attribute parameters
(`DRIFT_PARTICLE_COUNT=1000`, `ATTRIBUTION_SEARCH_RADIUS_NM=50`, …),
SMTP creds for alert webhooks, alert retry/timeout, log level.

### 1.4 Database requirements

- **Engine**: PostgreSQL 15 with **PostGIS 3.4** (the `postgis/postgis:15-3.4`
  image is pinned). Schemas are loaded in filename order from `database/init/`
  on first start (00_extensions → 01_ais → 02_spill → 03_drift → 04_suspect
  → 05_case → 06_spatial_indexes → 07_case_persistence).
- **Extensions required**: `postgis`, `postgis_topology`, `pg_trgm`,
  `uuid-ossp`, `btree_gist`.
- **Schema size today**: 7 SQL files, UUID PKs on every table, `GEOMETRY`
  columns are SRID 4326 (WGS84), GIST indexes on every geometry column, BRIN
  index on `ais_positions.timestamp`, covering index
  `idx_ais_mmsi_spatial_time`.
- **Migration**: migrations dir is empty. Schema is applied via the init
  volume; new schemas append a new numbered file to `database/init/` and
  ship a migration manually.
- **Capacity**: spills + drift particle origins + AIS positions are the
  growth drivers. Today:
  - 12 spill detections (seeded Wakashio case)
  - ~30k AIS positions in `ais_demo`
  - case files ~272 KB
  - Spatial index ~304 KB
- **Connection**: `asyncpg` (asyncpg 0.29) for all services. Parameterised
  SQL only; raw user input never reaches `text(...)`.
- **Backup**: PostGIS tables can be dumped via `pg_dump -Fc`; PITR via
  continuous WAL archiving is recommended once you have steady-state writes.

### 1.5 External services

| Service | Used for | Auth | Failure mode |
|---------|----------|------|--------------|
| **Copernicus Data Space Ecosystem (CDSE)** | Sentinel-1 SAR archive search + Process API GeoTIFF ingest | OAuth2 client credentials (`CDSE_CLIENT_ID/_SECRET`) | No archive browser; ingest cannot fetch new scenes |
| **Climate Data Store (CDS)** | ERA5 reanalysis winds | `CDSAPI_KEY` | **Drift refuses to run** — OpenOil needs wind |
| **Copernicus Marine Service (CMEMS)** | GLORYS12 currents (`cmems_mod_glo_phy_my_0.083deg_P1D-m`, vars `uo,vo`) | Username + password | Drift runs without currents (degraded) |
| **NOAA NOMADS GFS** | live-operations wind fallback | none | live ops only — retains ~10 days, so historical cases must use ERA5 |
| **AISStream.io** | live AIS (WebSocket) | `AIS_API_KEY` | **Terrestrial only** — verified 48 messages globally in 13 s, 0 over open ocean |
| **AISHub** | live AIS (HTTP) | `AISHUB_API_KEY` | Same caveat as AISStream |
| **Ollama** | LLM case narrative | none | Intel service falls back to a deterministic template; logs that it did |
| **OpenAI API** | LLM case narrative (alternative) | `OPENAI_API_KEY` | Same fallback |
| **SMTP** | alert email webhooks | host/port/user/pass | Alert retries; failures logged

> The `GFS_OPENDAP_URL` key in `.env.example` is **deliberately empty** —
> NOAA retired OPeNDAP (SCN 25-81). The replacement is `GFS_THREDDS_URL`.
> Carrying the old URL would have been a paper trail, not a working integration.

### 1.6 Storage requirements

| Asset | Size today | Growth driver |
|-------|------------|---------------|
| Real SAR archive (1200 GeoTIFFs, 2-band float32 dB, 2048², LZW) | **~38 GB** on external disk | New acquisitions only — read in place, not copied |
| Real masks (`data/oil_spill_masks/`) | **~4.7 GB** | Paired 1:1 with images |
| Synthetic dataset (`data/synthetic/`) | **~2.9 GB** | Regenerable from `scripts/generate_synthetic.py` |
| Checkpoint (`detector_best.pth` UNet++/resnet34 in_ch=2, 24.7M params) | **283 MB** per run × ~12 runs = ~5 GB | Per completed training run |
| `data/sar/` (live inference GeoTIFFs) | 180 MB for 6 scenes | Each new upload |
| `data/index/real/manifest.jsonl` | 304 KB | Built once, rebuilt only when archive changes |
| Case files (`data/case_files/`) | 272 KB | Per investigation case |
| Logs / Prometheus TSDB | minimal (~MB) | 15-day retention recommended |
| `runs/<run_id>/run_report.json + metrics.jsonl` | KB per run | Per training run |

**S3 layout recommended** (mirrors local):
```
s3://sentinel-prod-<region>/
  sar/                 (live inference uploads; lifecycle: IA → Glacier 90d)
  index/               (manifest.jsonl + checksum)
  archive/             (real SAR archive when ported off external disk; Glacier)
  checkpoints/         (only `detector_best.pth` of the prod model)
  runs/                (run reports + metrics.jsonl; lifecycle: Glacier 1y)
  case-files/          (PDF/HTML output; lifecycle: IA 30d)
  logs/                (Prometheus + service logs)
```

### 1.7 Observability

- All Python services expose `/metrics` via `prometheus-fastapi-instrumentator`.
- `infra/prometheus/prometheus.yml` scrapes 7 services on 15 s intervals
  (api, ingest, detect, drift, attribute, intel, worker).
- Grafana provisioning dir is `infra/grafana/dashboards` (mount-only).
- Healthchecks are real HTTP `urllib.request.urlopen(...)` calls on every
  service in the lite profile — broken starts name the service rather than
  leaving a silent gap.

---

## 2. AWS deployment tiers

All three tiers share the same **service-inventory** above. What changes is
**how you host** them, what redundancy you buy, and how much idle headroom
you pay for.

Cost estimates below are **us-east-1 on-demand list prices** (Sep 2026),
rounded, **excluding** data-transfer-out beyond 100 GB/mo, tax, and support
tier fees. Apply 1-year reserved or Savings Plans where indicated to roughly
halve the compute line.

### 2.1 Tier 1 — Minimum (single analyst / demo / pilot)

**Use case**: 1–3 concurrent analysts, demo data only, no live ML training,
no GPU inference. The whole stack on one host.

#### Architecture

- **1× EC2** `t3.xlarge` (4 vCPU, 16 GB RAM, gp3 EBS) running the 8-service
  lite profile under Docker Compose (or migrate to a small ECS Fargate
  cluster for cleanliness — both are listed).
- **RDS** `db.t3.small` PostgreSQL 15 (single-AZ to start; promote to Multi-AZ
  when the first customer is on it). 50 GB gp3 storage, 7-day backup retention.
- **ElastiCache** `cache.t3.micro` Redis 7, single node.
- **S3** bucket for `sar/`, `case-files/`, `runs/` (~10 GB).
- **ALB** in front of the UI (port 443 → nginx on :3000) and the API
  (port 8000). Single ALB; one ACM cert; one Route 53 record.
- **NAT Gateway** in 1 AZ for egress to CDSE/CDS/CMEMS.
- **Secrets Manager** holds Copernicus + LLM credentials (rotated quarterly).
- **CloudWatch** logs + basic metrics; no Prometheus/Grafana (use CloudWatch
  dashboards).

#### Resource breakdown

| Resource | Spec |
|----------|------|
| Compute (EC2 t3.xlarge) | 4 vCPU, 16 GB RAM, 30 GB gp3 root |
| Database | 2 vCPU, 2 GB RAM, 50 GB gp3, single-AZ |
| Cache | 1 vCPU, 0.5 GB RAM, single node |
| Storage | S3 10 GB + EBS 30 GB |
| Network | ALB + NAT GW (1 AZ) |
| ML/GPU | **none** (Tier-A torch-free detector) |
| ML training | **none** (use the managed tier-2 GPU path or run training on a laptop) |

#### Estimated monthly cost

| Line item | On-demand | 1-yr reserved / savings |
|-----------|-----------|-------------------------|
| EC2 t3.xlarge 730 h (Linux, us-east-1) | $120 | $60 |
| RDS db.t3.small PostgreSQL, 50 GB gp3, single-AZ | $60 | $45 |
| ElastiCache cache.t3.micro, 1 node | $15 | $12 |
| S3 Standard 10 GB + requests | $1 | $1 |
| ALB + 1 LCU avg | $20 | $20 |
| NAT Gateway 730 h + 10 GB processed | $35 | $35 |
| EBS gp3 30 GB + 3k IOPS | $5 | $5 |
| Data transfer out (5 GB) | $1 | $1 |
| CloudWatch logs (5 GB) + metrics | $15 | $15 |
| Secrets Manager (5 secrets × $0.40) | $2 | $2 |
| Route 53 hosted zone + 1 record | $1 | $1 |
| **Total** | **~$275/mo** | **~$195/mo** |

> Add **~$30/mo** for a `db.t3.medium` Multi-AZ + ALB cross-AZ if you want the
> first day of customer use to be safe.

**When to stop**: this tier is fine for the Wakashio demo and 1–3 analysts.
Once you have ≥5 concurrent users, real-time ingest from CDSE, or any GPU
inference, promote to Tier 2.

### 2.2 Tier 2 — Mid-range (production with periodic retraining)

**Use case**: up to ~50 concurrent analysts, real CDSE/CMEMS ingest, one
retraining run per week, GPU inference for the trained UNet++ (Tier-B),
managed observability.

#### Architecture

- **ECS on Fargate** behind an ALB, three subnets across 2 AZs:
  - `sentinel-api`, `sentinel-ui` (nginx), `sentinel-ingest`, `sentinel-attribute`,
    `sentinel-intel`, `sentinel-worker` (Celery) — 0.5 vCPU / 1 GB each
    (`FARGATE` profile, no Spot, on-demand).
  - `sentinel-detect` — **GPU Fargate task** with `g5.xlarge` (1× A10G, 4 vCPU,
    16 GB) × 2 tasks behind a target group; autoscaling on
    `ApproximateNumberOfMessagesVisible` (Celery queue depth).
  - `sentinel-drift` — 1 vCPU / 4 GB × 2 tasks (CPU is enough; OpenDrift is
    Python + numpy).
- **RDS** `db.r6g.large` PostgreSQL 15 + PostGIS 3.4, Multi-AZ, 100 GB gp3
  with autoscaling to 500 GB. 14-day backup retention.
- **ElastiCache** `cache.r6g.large` Redis 7, 1 primary + 1 replica.
- **S3** bucket, ~500 GB initial (SAR archive lifted off the external disk).
  Lifecycle: `sar/` → IA at 30 d → Glacier at 90 d; `runs/` → Glacier at 1 y.
- **EFS** for shared `/app/data` (small — case files + inference outputs).
- **SageMaker** training jobs on `g4dn.xlarge` (4 vCPU, 16 GB, **1× T4 GPU**)
  via Spot, triggered by EventBridge schedule or manually. Container =
  `Dockerfile.train` (untested as noted — first build is a known risk; budget
  for one failure). Writes checkpoints back to S3.
- **SageMaker endpoint** (or self-hosted on ECS Fargate GPU) for the trained
  UNet++ in `sentinel-detect`. Or stay with the Tier-A torch-free detector and
  skip this entirely.
- **Networking**: ALB (2 AZs), NAT GW × 2 AZs, VPC endpoints for S3 + ECR +
  Secrets Manager + CloudWatch Logs (no NAT for those).
- **Observability**: **Amazon Managed Service for Prometheus** (AMP) +
  **Amazon Managed Grafana** (AMG). Prometheus scrapes the Fargate tasks via
  the service-discovery exporter.
- **Secrets**: Secrets Manager for CDSE/CDS/CMEMS/AIS/OpenAI/Ollama endpoints.
- **Backups**: RDS automated + manual snapshot before each deploy.

#### Resource breakdown

| Resource | Spec |
|----------|------|
| API / UI / ingest / attribute / intel / worker Fargate | 6 × (0.5 vCPU, 1 GB) = 3 vCPU, 6 GB |
| Worker + detect (GPU) Fargate | 2 × g5.xlarge (4 vCPU, 16 GB, 1× A10G) |
| Drift Fargate | 2 × (1 vCPU, 4 GB) = 2 vCPU, 8 GB |
| Database | 2 vCPU, 16 GB RAM, 100 GB gp3, Multi-AZ |
| Cache | 2 vCPU, 13 GB RAM, 1 primary + 1 replica |
| Storage | S3 500 GB (lifecycle), EFS 50 GB Standard |
| Network | ALB (2 AZ) + NAT GW × 2 AZ + VPC endpoints |
| ML training | g4dn.xlarge Spot, ~30 h/month = ~$10/mo (Spot avg) |

#### Estimated monthly cost

| Line item | On-demand |
|-----------|-----------|
| Fargate compute (6 × 0.5 vCPU × 1 GB × 730 h) | $110 |
| Fargate compute drift (2 × 1 vCPU × 4 GB × 730 h) | $100 |
| Fargate GPU detect (2 × g5.xlarge on-demand, 730 h each) | $2,400 |
| ALB + LCU (2 AZ) | $30 |
| NAT Gateway × 2 AZ + processing | $70 |
| VPC endpoints (S3, ECR, Logs, Secrets Mgr) | $25 |
| RDS db.r6g.large Multi-AZ PostgreSQL, 100 GB gp3 | $310 |
| ElastiCache cache.r6g.large, 1 primary + 1 replica | $260 |
| S3 500 GB Standard + IA + Glacier + requests | $15 |
| EFS 50 GB Standard | $15 |
| SageMaker training (g4dn.xlarge Spot, 30 h/mo) | $10 |
| Amazon Managed Prometheus (5 GB ingest, 50k samples/s) | $40 |
| Amazon Managed Grafana (1 editor, 1 viewer) | $30 |
| CloudWatch logs (30 GB) + metrics | $50 |
| Secrets Manager (10 secrets × $0.40) + KMS | $5 |
| Data transfer out (50 GB) | $5 |
| **Total** | **~$3,475/mo** |

**Cut to ~$1,900/mo** by:
- Detecting via the torch-free Tier-A detector (drop the 2× g5.xlarge =
  -$2,200/mo). The Tier-B UNet++ checkpoint is the *only* thing the GPU is
  used for in `sentinel-detect`. Until a checkpoint is in prod, do not pay
  for GPU.
- 1-year Compute Savings Plan on Fargate (-30%).
- 1-year reserved RDS and ElastiCache (-35%).

**Realistic committed cost**: **$1,500–$2,500/mo**, depending on whether GPU
inference is on.

**When to stop**: this tier copes with one burst-retraining per week, ~50
analysts, and ingest from all four real sources. It does not give you
multi-region, sub-minute failover, or p4d-class training. Promote to Tier 3
when the customer count crosses ~250, when the regulator starts asking for
multi-region DR, or when you want to retrain daily.

### 2.3 Tier 3 — Maximum (production at scale / multi-region)

**Use case**: ≥250 concurrent analysts, daily retraining, GPU inference in
prod, multi-AZ HA, multi-region DR, full audit trail, WAF + Shield, CloudFront
in front of the UI.

#### Architecture

- **EKS** (Kubernetes) on a managed control plane, 3 AZs in one region.
  - **System node group** (m6i.large) for cluster add-ons.
  - **API / UI / worker / intel** node group: 3× `m6i.2xlarge` (8 vCPU, 32 GB)
    spread across 3 AZs, mixed-instances policy, HPA on CPU + Celery depth.
  - **Detect (GPU)** node group: 3× `g5.2xlarge` (8 vCPU, 32 GB, 1× A10G) in
    3 AZs, k8s `nodeSelector: workload=detect`, taints to keep non-GPU pods
    out. Karpenter or Cluster Autoscaler for spikes.
  - **Drift / attribute** node group: 3× `c6i.2xlarge` (8 vCPU, 16 GB).
  - **Ingest** node group: 3× `m6i.xlarge` (4 vCPU, 16 GB) — APScheduler
    polling + httpx fetches; modest CPU.
- **RDS Aurora PostgreSQL Serverless v2** (PostGIS-compatible; verify the
  PostGIS version on the chosen Aurora minor), 2–16 ACU autoscaling,
  Multi-AZ, 1 writer + 2 readers, 500 GB initial storage.
- **ElastiCache for Redis** cluster mode disabled, 3× `cache.r6g.large`
  shards, Multi-AZ.
- **S3** multi-region: primary bucket in `us-east-1`, replicated to
  `us-west-2` and `eu-west-1` via Cross-Region Replication for `archive/`,
  `checkpoints/`, `case-files/`. Lifecycle: Standard → IA 30 d → Glacier
  90 d → Deep Archive 1 y.
- **CloudFront** in front of the UI bucket (static `dist/` upload via
  `services/ui/Dockerfile` build → S3 + CloudFront origin).
- **SageMaker** training on **p4d.24xlarge** (8× A100, 1152 GB RAM) via
  Spot, 2–4 runs/day. 24× A100 GPU-hours per retrain. Container =
  `Dockerfile.train`.
- **SageMaker endpoint** with multi-AZ GPU autoscaling for the trained
  detector, OR an EKS `sentinel-detect` deployment with HPA on Celery queue
  depth.
- **Networking**: ALB (3 AZ), NAT GW × 3 AZ, VPC endpoints, **AWS WAF** on
  the public ALB, **AWS Shield Standard** (free) at minimum, Shield
  Advanced for the regulator-facing UI.
- **Multi-region DR**: pilot-light in a second region. Aurora Global
  Database (1 secondary), EKS second cluster (warm), Route 53 failover on
  health check.
- **Observability**: Amazon Managed Prometheus + Amazon Managed Grafana,
  CloudWatch Container Insights, AWS X-Ray, VPC Flow Logs to S3.
- **Audit + secrets**: Secrets Manager + KMS, CloudTrail data events on the
  S3 archive bucket, AWS Config rules for "all secrets encrypted",
  "RDS Multi-AZ on", "S3 public access blocked".

#### Resource breakdown

| Resource | Spec |
|----------|------|
| EKS control plane | 1 cluster, 3 AZs |
| API / UI / worker / intel nodes | 3 × m6i.2xlarge (8 vCPU, 32 GB) |
| Detect (GPU) nodes | 3 × g5.2xlarge (8 vCPU, 32 GB, 1× A10G) |
| Drift / attribute nodes | 3 × c6i.2xlarge (8 vCPU, 16 GB) |
| Ingest nodes | 3 × m6i.xlarge (4 vCPU, 16 GB) |
| System / add-on nodes | 3 × m6i.large |
| Database | Aurora PostgreSQL Serverless v2, 2–16 ACU, Multi-AZ, 1 writer + 2 readers |
| Cache | ElastiCache Redis, 3× cache.r6g.large |
| Object storage | S3 multi-region, ~2 TB total |
| CDN | CloudFront, ~500 GB egress |
| Network | ALB (3 AZ), NAT GW × 3 AZ, WAF, Shield |
| ML training | p4d.24xlarge Spot, intermittent |

#### Estimated monthly cost (single-region baseline, no DR standby cost)

| Line item | On-demand |
|-----------|-----------|
| EKS control plane | $150 |
| EC2 API/UI/worker/intel (3 × m6i.2xlarge × 730 h) | $1,050 |
| EC2 detect GPU (3 × g5.2xlarge × 730 h) | $2,650 |
| EC2 drift/attribute (3 × c6i.2xlarge × 730 h) | $700 |
| EC2 ingest (3 × m6i.xlarge × 730 h) | $420 |
| EC2 system (3 × m6i.large × 730 h) | $270 |
| ALB + LCU (3 AZ) | $60 |
| NAT Gateway × 3 AZ + processing | $105 |
| VPC endpoints + PrivateLink | $50 |
| Aurora PG Serverless v2 (avg 8 ACU × 730 h) + 500 GB | $700 |
| ElastiCache 3× cache.r6g.large, Multi-AZ | $390 |
| S3 2 TB Standard + IA + Glacier + Deep Archive + requests | $60 |
| CloudFront 500 GB egress + requests | $80 |
| WAF (1 web ACL, 5 rules) | $15 |
| Shield Advanced (optional) | $3,000 |
| SageMaker training (p4d.24xlarge Spot, ~120 GPU-h/mo) | $1,200 |
| Amazon Managed Prometheus (20 GB ingest, 200k samples/s) | $150 |
| Amazon Managed Grafana (5 editors, 20 viewers) | $200 |
| CloudWatch Container Insights + logs (100 GB) + X-Ray | $250 |
| Secrets Manager (15 secrets) + KMS | $20 |
| Data transfer out (500 GB) | $45 |
| Route 53 + ACM | $5 |
| CloudTrail data events on S3 | $40 |
| **Subtotal (single region, no Shield Advanced)** | **~$8,600/mo** |
| **+ Shield Advanced** | **~$11,600/mo** |
| **+ DR standby (Aurora secondary + warm EKS + S3 CRR)** | **+$2,500–$5,000/mo** |

**Realistic committed cost** (1-year Compute Savings Plan on EC2 + Fargate
equivalents, 1-year reserved DB and cache, multi-year S3 commitment):
**$5,000–$7,500/mo** single region, **$7,500–$12,000/mo** with Shield
Advanced and pilot-light DR.

---

## 3. Tier comparison

| Dimension | Tier 1 | Tier 2 | Tier 3 |
|-----------|--------|--------|--------|
| Concurrent analysts | 1–3 | ~50 | ≥250 |
| ML inference | Tier-A (torch-free) | Tier-A or Tier-B GPU | Tier-B GPU, autoscaled |
| Retraining cadence | manual / laptop | weekly | daily |
| DB availability | single-AZ | Multi-AZ | Aurora Multi-AZ + Global DB |
| HA target | none | Multi-AZ in one region | Multi-AZ + multi-region |
| Observability | CloudWatch | AMP + AMG | AMP + AMG + X-Ray + Container Insights |
| WAF / Shield | none | optional WAF | WAF + Shield Standard/Advanced |
| **Monthly cost (on-demand)** | **~$275** | **~$3,475** (GPU) / ~$1,275 (CPU-only detect) | **~$8,600–$11,600** single region |
| **Monthly cost (1-yr commit)** | **~$195** | **~$1,500–$2,500** | **~$5,000–$12,000** |
| **Best for** | pilot, demo, internal eval | first paying customer, regulator demo | multi-tenant production |

---

## 4. Deployment prerequisites & open risks

These are the things that *will* bite during the first AWS build, called out
so the work is in scope from day one rather than found the hard way.

1. **`Dockerfile.train` is unverified.** `HANDOVER.md` §3.2 and `RUNBOOK.md`
   §5.6 say so. Build it on an AWS CodeBuild or local Docker-equipped CI
   runner *before* committing the SageMaker training pipeline. The first build
   failure is the most likely place a stale `requirements.txt` line will
   show up.

2. **CDSE / CDS / CMEMS credentials must be secrets, not env vars.** Wire
   Secrets Manager from day one; don't let `.env` files reach the image.
   `services/ingest/app/provenance.py` already treats unset values as
   fail-closed — that contract must hold in prod.

3. **The lite compose profile is the safest starting point.** It drops
   Prometheus, Grafana, Celery worker, and the intel service — the four
   components whose Dockerfiles have the least production evidence. Promote
   to the full profile only when the lite path is steady-state.

4. **PostGIS on Aurora.** Aurora PostgreSQL supports PostGIS via the
   `postgresql-aws-15-postgis` extension pack, but the version pinned in
   `database/init/00_extensions.sql` (`postgis`) must match what Aurora
   exposes. Verify on the chosen Aurora minor before you commit to it; a
   silent "extension not found" on first migration is the worst kind of
   surprise.

5. **GIST indexes on every geometry column** (AGENTS.md rule). Aurora's
   storage layer handles this, but the index build time on first load is
   non-trivial — budget 20–30 min for the full `database/init/*.sql` run on
   empty Aurora.

6. **eccodes for the drift container.** The drift image needs `libeccodes0`
   (`cfgrib` runtime) — not currently installed in `services/drift/Dockerfile`.
   Patch the Dockerfile or run drift natively (`RUNBOOK.md` §4.2).

7. **Ollama is local by default.** `services/intel` expects `OLLAMA_BASE_URL=
   http://localhost:11434`. In AWS, point this at a **persistent** endpoint
   (ECS sidecar, SageMaker LLM endpoint, or Bedrock if you want managed).
   Without one, intel falls back to a deterministic template — by design.

8. **The synthetic-data gates default OFF, and that is correct.** Verify in
   every env file that `ALLOW_SYNTHETIC_FORCING=false`,
   `SENTINEL_DATASOURCES__ALLOW_SYNTHETIC_FALLBACK=false`, etc. A typo here
   silently flips "fail loud" to "silently invent numbers" — the exact bug
   `docs/VERIFICATION.md §9.2` records.

9. **Band order on the SAR archive is assumed, not measured.** Treat any
   "VV" reading in user-facing output as provisional until `LIMITations.md`
   B12 is closed. The trained UNet++ is band-agnostic (both channels used),
   so this affects only the *displayed* label, not inference correctness.

10. **No held-out test set yet.** Until `test_fraction > 0` and a leak-free
    split is run, `best_val_iou 0.2981` is the optimistic bound, not a
    deployable SLA. Quote carefully.