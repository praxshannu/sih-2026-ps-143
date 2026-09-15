#!/usr/bin/env bash
# ============================================================================
# SENTINEL — pre-deployment checks
# ============================================================================
#
# Run this on the EC2 instance, from the repo root, BEFORE the first build:
#
#     bash deploy/preflight.sh
#
# Exit code 0 = safe to build. Non-zero = fix the failures first.
#
# Every check here corresponds to something that has actually gone wrong in
# this repo, or that would fail silently if it did. The silent ones are the
# reason this script exists: a container that starts, answers /health with 200
# and emits garbage is worse than one that refuses to boot.
#
#     docs/DEPLOYMENT_FREE_TIER.md §5   the blockers
#     docs/DEPLOYMENT_FREE_TIER.md §10  the verification checklist

set -uo pipefail

FAILS=0
WARNS=0

if [ -t 1 ]; then
  R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; D=$'\033[2m'; Z=$'\033[0m'
else
  R=""; G=""; Y=""; D=""; Z=""
fi

ok()   { printf '  %s[ ok ]%s %s\n'   "$G" "$Z" "$1"; }
bad()  { printf '  %s[FAIL]%s %s\n'   "$R" "$Z" "$1"; FAILS=$((FAILS+1)); }
warn() { printf '  %s[warn]%s %s\n'   "$Y" "$Z" "$1"; WARNS=$((WARNS+1)); }
head_() { printf '\n%s%s%s\n' "$D" "$1" "$Z"; }

COMPOSE=docker-compose.aws.yml

printf '\nSENTINEL preflight\n'
printf '%s\n' "------------------"

# ---------------------------------------------------------------- environment
head_ "environment"

if [ "$(uname -s)" != "Linux" ]; then
  warn "host is $(uname -s), not Linux — these checks target the EC2 instance"
fi

if ! command -v docker >/dev/null 2>&1; then
  bad "docker not found. Install it: sudo apt-get install -y docker.io docker-compose-v2"
else
  ok "docker present: $(docker --version 2>/dev/null | cut -d, -f1)"
fi

if docker compose version >/dev/null 2>&1; then
  ok "compose v2 present: $(docker compose version --short 2>/dev/null)"
else
  bad "docker compose v2 not available (needs the 'docker compose' subcommand, not docker-compose v1)"
fi

if [ ! -f "$COMPOSE" ]; then
  bad "$COMPOSE not found — run this from the repo root"
else
  ok "found $COMPOSE"
  if docker compose -f "$COMPOSE" config -q >/dev/null 2>&1; then
    ok "$COMPOSE parses and interpolates cleanly"
  else
    warn "$COMPOSE did not parse (usually a missing .env — see below)"
  fi
fi

# ----------------------------------------------------------------------- .env
head_ ".env"

if [ ! -f .env ]; then
  bad ".env missing. Fix: cp .env.example .env"
else
  ok ".env present"

  # Values that must not still be the placeholder.
  for key in SECRET_KEY JWT_SECRET DB_PASSWORD; do
    val="$(grep -E "^${key}=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"
    if [ -z "$val" ]; then
      bad "$key is unset"
    elif printf '%s' "$val" | grep -qiE 'change-me|change_me|^sentinel_secret$|^too$|^$'; then
      bad "$key is still the placeholder — generate one: openssl rand -hex 32"
    else
      ok "$key is set and not the placeholder"
    fi
  done

  # Synthetic-data gates. These are fail-closed by design: unset or misspelled
  # reads as False, which is correct. The danger is the opposite direction —
  # a truthy value silently swaps "fail loud" for "invent numbers".
  # services/ingest/app/provenance.py
  head_ "synthetic-data gates (all must be false)"
  for gate in ALLOW_SYNTHETIC_FORCING ALLOW_SYNTHETIC_AIS SENTINEL_DEMO_MODE \
              ALLOW_SYNTHETIC_TRAINING SENTINEL_DATASOURCES__ALLOW_SYNTHETIC_FALLBACK; do
    val="$(grep -E "^${gate}=" .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' \r' | tr '[:upper:]' '[:lower:]')"
    case "$val" in
      ""|false|0|no|off) ok "$gate = ${val:-unset}" ;;
      *) bad "$gate = $val  ← TRUTHY. This makes the system invent data instead of failing." ;;
    esac
  done
fi

# --------------------------------------------------------- the trained model
head_ "trained checkpoint (the silent-failure check)"

if [ ! -d models ]; then
  bad "models/ does not exist. Docker would create it empty, the mount would"
  bad "      succeed, and detect would serve an UNTRAINED encoder at HTTP 200."
  bad "      Fix: mkdir -p models && cp checkpoints/<run>/detector_best.pth models/"
elif [ ! -f models/detector_best.pth ]; then
  bad "models/detector_best.pth missing — same silent fallback as above."
  bad "      Fix: cp checkpoints/20260915T180945Z-real-10ep-512-holdout-7e9d57/detector_best.pth models/"
else
  sz=$(stat -c %s models/detector_best.pth 2>/dev/null || stat -f %z models/detector_best.pth 2>/dev/null || echo 0)
  mb=$((sz / 1048576))
  if [ "$mb" -ge 200 ] && [ "$mb" -le 400 ]; then
    ok "models/detector_best.pth present, ${mb} MB (expect ~283 MB)"
  else
    bad "models/detector_best.pth is ${mb} MB — expected ~283 MB. Truncated transfer?"
  fi
  # --- integrity -----------------------------------------------------------
  # A .pth is a zip archive. Checking that catches a truncated rsync or an
  # HTML error page saved under a .pth name, both of which torch rejects later
  # with a confusing message. Pure stdlib, so it works on the instance.
  if command -v python3 >/dev/null 2>&1; then
    integ=$(python3 - "$PWD/models/detector_best.pth" <<'PY' 2>/dev/null
import sys, zipfile
try:
    z = zipfile.ZipFile(sys.argv[1])
except Exception:
    print("BADZIP"); raise SystemExit
n = z.namelist()
if not any(x.endswith('/data.pkl') for x in n):
    print("NOPICKLE"); raise SystemExit
print("OK %d %.1f" % (len(n), sum(i.file_size for i in z.infolist()) / 1048576))
PY
)
    case "$integ" in
      "OK "*) set -- $integ
              ok "valid torch archive (${2} entries, ${3} MB uncompressed)" ;;
      BADZIP)   bad "not a valid zip — truncated transfer, or an HTML error page saved as .pth" ;;
      NOPICKLE) bad "no data.pkl inside — not a torch checkpoint" ;;
      *)        warn "could not validate the archive (python3 zipfile unavailable?)" ;;
    esac
  fi

  # --- in_channels ---------------------------------------------------------
  # This is the one that matters, and it needs real torch. Scraping the pickle
  # without torch was tried and is NOT reliable enough to gate a deploy on, so
  # the check runs inside the built detect image instead. Before the first
  # build there is no image, and the authoritative check is then the container
  # log after start (§6.8).
  if docker image inspect sentinel-aws-sentinel-detect >/dev/null 2>&1 \
     || docker compose -f "$COMPOSE" images sentinel-detect 2>/dev/null | grep -q sentinel; then
    ic=$(docker compose -f "$COMPOSE" run --rm --no-deps --entrypoint python \
           sentinel-detect -c "
import torch
ck = torch.load('/app/models/detector_best.pth', map_location='cpu', weights_only=True)
print('IN_CHANNELS=' + str(ck.get('in_channels','?')))" 2>/dev/null | grep -o 'IN_CHANNELS=.*' | cut -d= -f2)
    case "${ic:-}" in
      2)  ok "checkpoint in_channels = 2, matches IN_CHANNELS in $COMPOSE" ;;
      "") warn "could not read in_channels from the image (built yet?)" ;;
      *)  bad "checkpoint in_channels = $ic but $COMPOSE pins IN_CHANNELS=2 — load will fail" ;;
    esac
  else
    warn "detect image not built yet — in_channels is verified after start."
    warn "      Authoritative: docker compose -f $COMPOSE logs sentinel-detect | grep 'Loading model'"
  fi
fi

# ------------------------------------------------------------- runtime assets
head_ "runtime assets"

if [ -d data/sar ] && [ "$(ls -1 data/sar/*.tif 2>/dev/null | wc -l)" -gt 0 ]; then
  ok "data/sar/ has $(ls -1 data/sar/*.tif 2>/dev/null | wc -l) GeoTIFF scene(s)"
else
  bad "data/sar/ has no .tif files — the demo has nothing to detect on."
  bad "      Fix: rsync data/sar/ from the Mac (see §6.5)"
fi

if [ -f data/index/real/manifest.jsonl ]; then
  ok "data/index/real/manifest.jsonl present ($(wc -l < data/index/real/manifest.jsonl) rows)"
else
  warn "data/index/real/manifest.jsonl absent — fine for the demo path, which does"
  warn "      not read it. Only needed to re-derive the real training split."
fi

sqlcount=$(ls -1 database/init/*.sql 2>/dev/null | wc -l)
if [ "$sqlcount" -eq 8 ]; then
  ok "database/init/ has all 8 migration files"
elif [ "$sqlcount" -gt 0 ]; then
  warn "database/init/ has $sqlcount SQL files, expected 8"
else
  bad "database/init/ is empty — the schema will not be created"
fi

# ------------------------------------------------------------------ build ctx
head_ "build context"

if [ -f services/ui/.dockerignore ]; then
  ok "services/ui/.dockerignore present (blocks the 891 MB node_modules copy)"
else
  bad "services/ui/.dockerignore MISSING — the UI build will copy the host's"
  bad "      node_modules over the image's and fail, or produce a broken bundle."
fi

if [ -d services/ui/node_modules ]; then
  warn "services/ui/node_modules exists on this host (891 MB). Harmless now that"
  warn "      .dockerignore excludes it, but it should not have been transferred."
fi

# --------------------------------------------------------------- host capacity
head_ "host capacity"

avail=$(df -Pk . 2>/dev/null | awk 'NR==2 {print int($4/1048576)}')
if [ -n "${avail:-}" ]; then
  if [ "$avail" -ge 20 ]; then
    ok "${avail} GB free on this volume (need >= 20 for ~10 GB of images + build cache)"
  elif [ "$avail" -ge 10 ]; then
    warn "${avail} GB free — tight. Prune first: docker builder prune -af"
  else
    bad "${avail} GB free — the build will fill the disk. Prune, or grow the volume."
  fi
fi

memgb=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}')
if [ -n "${memgb:-}" ]; then
  if [ "$memgb" -ge 7 ]; then
    ok "${memgb} GB RAM (m7i-flex.large has 8)"
  else
    bad "${memgb} GB RAM — this profile needs ~6.5 GB of caps plus the kernel."
    bad "      m7i-flex.large is the smallest instance that fits."
  fi
fi

swaptotal=$(free -m 2>/dev/null | awk '/^Swap:/ {print $2}')
if [ -n "${swaptotal:-}" ]; then
  if [ "$swaptotal" -gt 0 ]; then
    ok "swap active (${swaptotal} MB)"
  else
    warn "no swap. Building the torch image on 2 vCPU without swap risks an OOM"
    warn "      kill mid-build that reads as a network error. See §6.4."
  fi
fi

# ---------------------------------------------------------------------- result
printf '\n%s\n' "------------------"
if [ "$FAILS" -eq 0 ] && [ "$WARNS" -eq 0 ]; then
  printf '%sAll checks passed.%s Build with:\n\n' "$G" "$Z"
  printf '    COMPOSE_PARALLEL_LIMIT=1 docker compose -f %s build\n\n' "$COMPOSE"
  exit 0
elif [ "$FAILS" -eq 0 ]; then
  printf '%s0 failures, %d warning(s).%s Review the warnings, then build:\n\n' "$Y" "$WARNS" "$Z"
  printf '    COMPOSE_PARALLEL_LIMIT=1 docker compose -f %s build\n\n' "$COMPOSE"
  exit 0
else
  printf '%s%d failure(s), %d warning(s).%s Fix the failures before building.\n\n' "$R" "$FAILS" "$WARNS" "$Z"
  exit 1
fi
