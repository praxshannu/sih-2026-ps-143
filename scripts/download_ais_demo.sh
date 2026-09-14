#!/usr/bin/env bash
# Download a small real AIS sample for the SENTINEL DuckDB scorer.
# Source: NOAA MarineCadastre AIS (2020 archive, public domain).
# The DuckDB scorer also reads CSV, so no parquet conversion is required.
# Never fabricates data: on failure it exits non-zero with instructions.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
mkdir -p "$DATA_DIR"

# One daily zone file (~tens of MB zipped). July 2020 matches the demo window.
URL="${AIS_SAMPLE_URL:-https://coast.noaa.gov/htdata/CSV/AIS/2020/AIS_2020_07_07.zip}"
ZIP="$DATA_DIR/ais_marinecadastre_sample.zip"

echo "[ais] downloading $URL"
curl -fL --retry 3 -o "$ZIP" "$URL"
echo "[ais] extracting to $DATA_DIR"
unzip -o "$ZIP" -d "$DATA_DIR"

echo "[ais] done. Point the scorer at the extracted CSV:"
echo "      AIS_PARQUET_PATH=$DATA_DIR/<extracted>.csv  (CSV works too)"
echo "  or convert to parquet with: python -c \"import pandas as pd; pd.read_csv('<f>.csv').to_parquet('ais_demo.parquet')\""
