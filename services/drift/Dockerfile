FROM python:3.11-slim AS base
WORKDIR /app

# libeccodes0 is the system half of GRIB decoding: cfgrib drives it to read
# ERA5/GFS. Without it every wind fetch fails with an eccodes error.
# See docs/DEPLOYMENT_FREE_TIER.md §13.4.
#
# Note this is only the *system* half. The Python-side `cfgrib` package is not
# in requirements.txt either, so GRIB reads remain unavailable until both are
# added. That is deliberate and matches the architecture: every opendrift /
# copernicusmarine / cdsapi / cfgrib import in app/ is lazy and guarded, so the
# service still starts and answers /health 200 — the forecast routes fail with a
# typed error instead. Adding the full OpenDrift dependency set is a much larger
# change than this deployment authorises.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libeccodes0 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r sentinel && useradd -r -g sentinel sentinel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
RUN chown -R sentinel:sentinel /app
USER sentinel
EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
