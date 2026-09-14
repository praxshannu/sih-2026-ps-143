#!/usr/bin/env bash
# Download UNet++ model weights for SENTINEL oil spill detection.
# Creates weights directory and either downloads from a URL or generates
# random weights for demo/testing purposes.
#
# Usage:
#   bash scripts/download_model_weights.sh
#   bash scripts/download_model_weights.sh --download

set -euo pipefail

WEIGHTS_DIR="services/detect/app/models/weights"
WEIGHTS_FILE="${WEIGHTS_DIR}/unetpp_scse_best.pth"
DOWNLOAD_URL="${MODEL_WEIGHTS_URL:-}"
GENERATE_DEMO="${1:---generate}"

mkdir -p "${WEIGHTS_DIR}"

if [ -f "${WEIGHTS_FILE}" ]; then
    echo "[weights] Model weights already exist at ${WEIGHTS_FILE}"
    echo "[weights] Size: $(du -h "${WEIGHTS_FILE}" | cut -f1)"
    exit 0
fi

if [ "${GENERATE_DEMO}" = "--download" ] && [ -n "${DOWNLOAD_URL}" ]; then
    echo "[weights] Downloading from ${DOWNLOAD_URL}..."
    if command -v curl &>/dev/null; then
        curl -fSL --retry 3 --retry-delay 5 -o "${WEIGHTS_FILE}" "${DOWNLOAD_URL}"
    elif command -v wget &>/dev/null; then
        wget --tries=3 -O "${WEIGHTS_FILE}" "${DOWNLOAD_URL}"
    else
        echo "[weights] ERROR: Neither curl nor wget found"
        exit 1
    fi
    echo "[weights] Downloaded: $(du -h "${WEIGHTS_FILE}" | cut -f1)"
else
    echo "[weights] Generating demo weights (random initialization)..."
    python3 -c "
import torch
import sys
sys.path.insert(0, 'services/detect')
try:
    from app.models.unetpp_scse import UNetPlusPlusSCSE
    model = UNetPlusPlusSCSE(encoder_name='resnet34', encoder_weights=None)
    state = {'model_state_dict': model.state_dict(), 'epoch': 0, 'val_iou': 0.0}
    torch.save(state, '${WEIGHTS_FILE}')
    print('[weights] Saved random UNet++ weights (demo only)')
except Exception as e:
    print(f'[weights] Could not create model weights: {e}', file=sys.stderr)
    # Fallback: save a minimal state dict
    state = {'model_state_dict': {}, 'epoch': 0, 'val_iou': 0.0}
    torch.save(state, '${WEIGHTS_FILE}')
    print('[weights] Saved empty placeholder weights')
" 2>/dev/null || {
        echo "[weights] Python/torch not available, creating placeholder..."
        python3 -c "
import pickle, os
state = {'model_state_dict': {}, 'epoch': 0, 'val_iou': 0.0}
with open('${WEIGHTS_FILE}', 'wb') as f:
    pickle.dump(state, f)
print('[weights] Created placeholder weights file')
"
    }
fi

chmod 644 "${WEIGHTS_FILE}"
echo "[weights] Ready: ${WEIGHTS_FILE} ($(du -h "${WEIGHTS_FILE}" | cut -f1))"
