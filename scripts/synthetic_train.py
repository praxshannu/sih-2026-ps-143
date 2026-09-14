"""Generate synthetic SAR oil spill dataset for training the detect service model.

Based on OceanTrace generate_synthetic_data.py adapted for sentinel services layout.
Uses PNG output instead of GeoTIFF to avoid rasterio dependency on host.

FAIL-CLOSED CONTRACT
--------------------
Nothing here is real. A model trained on these images must never be presented as
a model trained on real SAR, so this script:

* refuses to run at all unless ``ALLOW_SYNTHETIC_TRAINING=true`` is set — a
  dataset that appears by accident is worse than no dataset;
* stamps **every** generated sample: a ``provenance=synthetic_mock`` PNG text
  chunk inside each file, so a leaked image still identifies itself;
* writes ``provenance.json`` next to the splits, listing every sample with its
  provenance and seed.

The gate is read through ``app.provenance.env_flag`` — the same helper the
ingest sources use — so the meaning of "true" cannot drift between them.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.ingest.app.provenance import (  # noqa: E402
    PROVENANCE_SYNTHETIC,
    SYNTHETIC_WARNING,
    env_flag,
)

ENV_ALLOW_SYNTHETIC_TRAINING = "ALLOW_SYNTHETIC_TRAINING"

# ---------------------------------------------------------------------------
# SAR Scene Generation (adapted from OceanTrace agent1 training)
# ---------------------------------------------------------------------------


def generate_sar_scene(
    width: int = 512,
    height: int = 512,
    num_slicks: int = 2,
    seed: int = 42,
    inject_hard_negative: bool = False,
) -> tuple:
    """
    Simulates a realistic Sentinel-1 SAR scene of open ocean with dark oil slick candidates.
    SAR features simulated:
    - High ocean backscatter background (~0.6 - 0.8 intensity)
    - Speckle noise (multiplicative Rayleigh/Gamma noise)
    - Dark oil slicks with suppressed backscatter (~0.1 - 0.2 intensity)
    - Irregular organic slick shapes with thin tails
    - Hard negative look-alikes if requested (wind-shadow zones or biogenic slicks)
    """
    np.random.seed(seed)

    # 1. Base ocean background with slight spatial intensity gradient
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    ocean_background = 0.65 + 0.1 * np.sin(3 * xx) * np.cos(2 * yy)

    # 2. Add SAR Speckle Noise (multiplicative noise)
    speckle = np.random.gamma(shape=4.0, scale=0.25, size=(height, width))
    sar_image = ocean_background * speckle

    # 3. Create ground truth mask and draw oil slicks
    mask_pil = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_pil)

    for _ in range(num_slicks):
        cx = np.random.randint(100, width - 100)
        cy = np.random.randint(100, height - 100)
        rx = np.random.randint(30, 70)
        ry = np.random.randint(15, 40)
        angle = np.random.randint(0, 180)

        # Draw main slick body
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(bbox, fill=255)

        # Draw slick drift tail
        tail_end_x = cx + int(rx * 1.8 * np.cos(np.radians(angle)))
        tail_end_y = cy + int(ry * 1.8 * np.sin(np.radians(angle)))
        draw.line([cx, cy, tail_end_x, tail_end_y], fill=255, width=np.random.randint(8, 16))

    mask = np.array(mask_pil, dtype=np.float32) / 255.0

    # 4. Suppress SAR backscatter where oil slicks exist
    slick_damping = 0.15 + 0.05 * np.random.randn(height, width)
    slick_damping = np.clip(slick_damping, 0.05, 0.3)

    sar_image = np.where(mask > 0.5, sar_image * slick_damping, sar_image)

    # 5. Inject hard negative look-alikes (do not update ground truth mask)
    if inject_hard_negative:
        cx = np.random.randint(100, width - 100)
        cy = np.random.randint(100, height - 100)
        rx = np.random.randint(80, 150)
        ry = np.random.randint(40, 80)

        hn_mask_pil = Image.new("L", (width, height), 0)
        hn_draw = ImageDraw.Draw(hn_mask_pil)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        hn_draw.ellipse(bbox, fill=255)
        hn_mask = np.array(hn_mask_pil, dtype=np.float32) / 255.0

        # Hard negative damping is slightly less intense or more uniform than oil
        hn_damping = 0.25 + 0.1 * np.random.randn(height, width)
        hn_damping = np.clip(hn_damping, 0.1, 0.4)
        sar_image = np.where(hn_mask > 0.5, sar_image * hn_damping, sar_image)

    sar_image = np.clip(sar_image, 0.0, 2.5)

    return sar_image.astype(np.float32), (mask > 0.5).astype(np.uint8)


def _png_metadata(**fields: str) -> PngInfo:
    """Provenance stamped *inside* the PNG, so the file is self-describing."""
    info = PngInfo()
    info.add_text("provenance", PROVENANCE_SYNTHETIC)
    info.add_text("generator", "scripts/synthetic_train.py")
    info.add_text("warning", SYNTHETIC_WARNING.format(source="synthetic SAR training sample"))
    for key, value in fields.items():
        info.add_text(key, value)
    return info


def save_png(filename: str, img_data: np.ndarray, pnginfo: PngInfo | None = None) -> None:
    """Save numpy array as a PNG image, normalized to [0, 255]."""
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Normalize to [0, 1] then to [0, 255]
    img = np.clip(img_data, 0.0, 1.0)
    norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    pil = Image.fromarray((norm * 255).astype(np.uint8))
    pil.save(filename, pnginfo=pnginfo or _png_metadata())


def generate_benchmark_dataset(
    base_dir: str = "data/synthetic", n_train: int = 150, n_val: int = 20, n_test: int = 10
) -> dict:
    """
    Generates train, validation, test datasets for the sentinel detect service.
    Creates PNG images and masks in the expected directory structure.
    The SAROilSpillDataset can read these PNG files.

    Refuses to run unless ALLOW_SYNTHETIC_TRAINING=true. Returns a manifest that
    names every sample together with its provenance.
    """
    if not env_flag(ENV_ALLOW_SYNTHETIC_TRAINING):
        raise RuntimeError(
            f"synthetic training data refused [{ENV_ALLOW_SYNTHETIC_TRAINING}_not_enabled]: "
            f"set {ENV_ALLOW_SYNTHETIC_TRAINING}=true to generate it. Every sample is "
            f"stamped {PROVENANCE_SYNTHETIC} and must never be presented as real SAR."
        )

    splits = {"train": (n_train, "train"), "val": (n_val, "val"), "test": (n_test, "test")}
    samples: list[dict] = []

    for split_name, (count, split_dir) in splits.items():
        img_dir = os.path.join(base_dir, split_dir, "images")
        mask_dir = os.path.join(base_dir, split_dir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(count):
            seed = 100 if split_name == "train" else (200 if split_name == "val" else 300)
            # Inject hard negatives in ~30% of the training images
            inject_hn = split_name == "train" and np.random.rand() < 0.3
            sar_img, mask = generate_sar_scene(
                width=512,
                height=512,
                num_slicks=np.random.randint(1, 3),
                seed=seed + i,
                inject_hard_negative=inject_hn,
            )

            # Save SAR image as PNG (normalized)
            img_file = os.path.join(img_dir, f"sar_{split_name}_{i + 1:03d}.png")
            # SAR images have range ~0-2.5, normalize for PNG display
            norm = np.clip(sar_img, 0, 2.5) / 2.5
            save_png(
                img_file,
                norm,
                _png_metadata(split=split_name, seed=str(seed + i), sample=str(i + 1)),
            )

            # Save PNG ground truth mask
            mask_file = os.path.join(mask_dir, f"sar_{split_name}_{i + 1:03d}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                mask_file,
                pnginfo=_png_metadata(split=split_name, seed=str(seed + i), kind="mask"),
            )

            samples.append(
                {
                    "split": split_name,
                    "index": i + 1,
                    "image": img_file,
                    "mask": mask_file,
                    "seed": seed + i,
                    "hard_negative": bool(inject_hn),
                    "provenance": PROVENANCE_SYNTHETIC,
                }
            )

    # Save a dedicated sample image for quick CLI inference testing
    sample_dir = os.path.join(base_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    sample_sar, sample_mask = generate_sar_scene(width=512, height=512, num_slicks=2, seed=999)
    norm = np.clip(sample_sar, 0, 2.5) / 2.5
    save_png(
        os.path.join(sample_dir, "sample_sentinel1.png"),
        norm,
        _png_metadata(split="sample", seed="999"),
    )
    Image.fromarray((sample_mask * 255).astype(np.uint8)).save(
        os.path.join(sample_dir, "sample_sentinel1_mask.png"),
        pnginfo=_png_metadata(split="sample", seed="999", kind="mask"),
    )
    samples.append(
        {
            "split": "sample",
            "index": 1,
            "image": os.path.join(sample_dir, "sample_sentinel1.png"),
            "mask": os.path.join(sample_dir, "sample_sentinel1_mask.png"),
            "seed": 999,
            "hard_negative": False,
            "provenance": PROVENANCE_SYNTHETIC,
        }
    )

    manifest = {
        "provenance": PROVENANCE_SYNTHETIC,
        "is_synthetic": True,
        "warning": SYNTHETIC_WARNING.format(source="synthetic SAR training set"),
        "reason": (
            "No labelled real oil-spill masks are available on this machine. "
            "These samples are simulated and must never back a real-data claim."
        ),
        "generator": "scripts/synthetic_train.py",
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "base_dir": base_dir,
        "count": len(samples),
        "samples": samples,
    }
    manifest_path = os.path.join(base_dir, "provenance.json")
    os.makedirs(base_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Synthetic SAR dataset generated in '{base_dir}/' ({len(samples)} samples)")
    print(f"Provenance manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    generate_benchmark_dataset()
