"""Extract the real oil-spill masks from the local 7z archive.

Why this is a separate, one-time step
-------------------------------------
The image half of the Zenodo Part-I archive is 40.7 GB and lives on an external
disk; it is read **in place** and never copied. The mask half is 6.2 MB
compressed (1200 x 4.2 MB binary GeoTIFFs, ~5 GB expanded) and the archive is
already on the local machine, so it is extracted locally rather than written
back to the external disk.

Nothing is moved off the external disk by this script. It only unpacks a file
that was already local.

Idempotent: if the destination already holds one mask per image, the script
reports that and exits without touching anything. ``--force`` re-extracts.

Usage::

    python scripts/prepare_real_masks.py
    python scripts/prepare_real_masks.py --check
    python scripts/prepare_real_masks.py --force --dest data/oil_spill_masks
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from sentinel_core import configure_logging, get_settings  # noqa: E402
from sentinel_core.errors import DataSourceUnavailableError, SentinelError  # noqa: E402

DEFAULT_ARCHIVE = "01_Train_Val_Oil_Spill_mask.7z"
RASTER_SUFFIXES = (".tif", ".tiff")
_APPLEDOUBLE_PREFIX = "._"

#: Homebrew's p7zip is not on the default PATH of a non-login shell.
_SEVEN_ZIP_CANDIDATES = ("7z", "7zz", "7za", "/opt/homebrew/bin/7z", "/usr/local/bin/7z")


def find_7z() -> str:
    for candidate in _SEVEN_ZIP_CANDIDATES:
        found = shutil.which(candidate) if not candidate.startswith("/") else candidate
        if found and Path(found).exists():
            return found
    raise SentinelError(
        "no 7-Zip binary found. Install it with `brew install p7zip`, then retry.",
        reason="seven_zip_missing",
    )


def list_raster_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        p.stem.lower()
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in RASTER_SUFFIXES
        and not p.name.startswith(_APPLEDOUBLE_PREFIX)
    }


def extract(archive: Path, dest: Path, seven_zip: str) -> int:
    """Extract ``archive`` into ``dest`` with paths flattened.

    ``7z e`` (not ``x``) drops the ``Mask_oil/`` prefix so the masks land flat
    next to each other and pair with the images by stem.
    """
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        seven_zip,
        "e",
        str(archive),
        f"-o{dest}",
        "-y",
        "-bso0",
        "-bsp0",
        # Without this macOS writes an AppleDouble sidecar per file, which then
        # looks like a second, 4 KB "mask" for every real one.
        "-xr!._*",
    ]
    logger.info("extracting {} -> {} (this takes a few minutes)", archive.name, dest)
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SentinelError(
            f"7-Zip failed with exit code {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}",
            reason="extract_failed",
            context={"archive": str(archive), "dest": str(dest)},
        )
    elapsed = time.monotonic() - started
    logger.info("extraction finished in {:.1f}s", elapsed)
    return len(list_raster_stems(dest))


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archive",
        type=Path,
        default=settings.paths.data_dir / DEFAULT_ARCHIVE,
        help="mask archive to extract",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=settings.datasources.real_masks_dir,
        help="where the masks should live (default: the configured masks dir)",
    )
    parser.add_argument("--force", action="store_true", help="re-extract even if present")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether masks and images pair up, extract nothing",
    )
    args = parser.parse_args()

    configure_logging(service="prepare-masks", level=settings.logging.level)

    images_dir = settings.datasources.real_images_dir
    image_stems = list_raster_stems(images_dir)
    mask_stems = list_raster_stems(args.dest)

    if args.check:
        logger.info("images: {} at {}", len(image_stems), images_dir)
        logger.info("masks : {} at {}", len(mask_stems), args.dest)
        paired = image_stems & mask_stems
        logger.info("paired: {}", len(paired))
        if image_stems - mask_stems:
            logger.error("{} image(s) without a mask", len(image_stems - mask_stems))
            return 1
        return 0

    if mask_stems and not args.force:
        paired = image_stems & mask_stems
        if paired and not (image_stems - mask_stems):
            logger.info(
                "masks already present and complete: {} files at {} (use --force to redo)",
                len(mask_stems),
                args.dest,
            )
            return 0
        logger.warning(
            "masks present but incomplete ({} of {} images paired); re-extracting",
            len(paired),
            len(image_stems),
        )

    if not args.archive.is_file():
        raise DataSourceUnavailableError(
            f"mask archive not found: {args.archive}",
            reason="mask_archive_missing",
            context={"archive": str(args.archive)},
        )

    seven_zip = find_7z()
    written = extract(args.archive, args.dest, seven_zip)

    mask_stems = list_raster_stems(args.dest)
    paired = image_stems & mask_stems
    missing = image_stems - mask_stems
    extra = mask_stems - image_stems

    logger.info("extracted {} mask(s)", written)
    logger.info("paired {} of {} image(s)", len(paired), len(image_stems))
    if extra:
        logger.warning("{} mask(s) have no matching image", len(extra))
    if missing:
        logger.error(
            "{} image(s) still have no mask (e.g. {})",
            len(missing),
            ", ".join(sorted(missing)[:5]),
        )
        return 1

    logger.success("real masks ready at {}", args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
