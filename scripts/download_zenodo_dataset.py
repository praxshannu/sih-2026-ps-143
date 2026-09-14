"""Safely download the large Zenodo Sentinel-1 oil-spill archives.

Zenodo stores the images in monolithic 7z files, so individual image selection
is not possible over HTTP. This script queries the record metadata first,
checks free disk space, downloads one archive, verifies its MD5, and writes a
manifest. It deliberately refuses Part I/II by default on small disks.

Examples::

    python scripts/download_zenodo_dataset.py --list
    python scripts/download_zenodo_dataset.py --record part3 --download
    python scripts/download_zenodo_dataset.py --record part3 --download --extract

The extracted archive still needs a dataset-layout preparation step before
training. Never point the trainer at an empty directory: it now fails loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://zenodo.org/api/records/{}"
RECORDS = {
    "part1": (8346860, "01_Train_Val_Oil_Spill_images.7z"),
    "part2": (8253899, "01_Train_Val_Lookalike_images.7z"),
    "part3": (13761290, "02_Test_images_and_ground_truth.7z"),
}


@dataclass(frozen=True)
class Archive:
    record: str
    record_id: int
    name: str
    size: int
    checksum: str
    url: str


def _metadata(record_id: int) -> dict[str, object]:
    with urllib.request.urlopen(API.format(record_id), timeout=30) as response:
        return json.load(response)


def _archive(record: str) -> Archive:
    record_id, expected_name = RECORDS[record]
    metadata = _metadata(record_id)
    files = metadata.get("files", [])
    if not isinstance(files, list):
        raise TypeError(f"Zenodo record {record_id} returned no file list")
    for item in files:
        if not isinstance(item, dict) or item.get("key") != expected_name:
            continue
        checksum = str(item.get("checksum", ""))
        links = item.get("links", {})
        url = links.get("self") if isinstance(links, dict) else None
        if not checksum or not isinstance(url, str):
            break
        return Archive(
            record, record_id, expected_name, int(item["size"]), checksum, url
        )
    raise RuntimeError(
        f"Expected archive {expected_name!r} not found in record {record_id}"
    )


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(archive: Archive, output: Path, min_free_gb: float, extract: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    required = archive.size + (5 * 1024**3 if extract else 0)
    if free < required + int(min_free_gb * 1024**3):
        raise RuntimeError(
            f"Refusing {archive.record}: needs about {_human(required)} plus "
            f"{min_free_gb:g} GiB safety margin, but only {_human(free)} is free."
        )

    target = output / archive.name
    print(f"Downloading {archive.name} ({_human(archive.size)})", flush=True)
    request = urllib.request.Request(
        archive.url, headers={"User-Agent": "sentinel-dataset-fetch/1.0"}
    )
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        target.open("wb") as stream,
    ):
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    actual = _md5(target)
    if actual != archive.checksum.removeprefix("md5:"):
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {target.name}: {actual} != {archive.checksum}"
        )

    manifest = {
        "record": archive.record,
        "zenodo_record": archive.record_id,
        "source": f"https://doi.org/10.5281/zenodo.{archive.record_id}",
        "file": archive.name,
        "bytes": archive.size,
        "md5": actual,
        "downloaded_to": str(target),
    }
    (output / f"{archive.record}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Verified {target} (md5 {actual})")

    if extract:
        if shutil.which("7z") is None:
            raise RuntimeError("--extract requires 7z; install p7zip before retrying")
        extract_dir = output / archive.record
        extract_dir.mkdir(exist_ok=True)
        import subprocess

        subprocess.run(["7z", "x", str(target), f"-o{extract_dir}", "-y"], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="show authoritative archive sizes"
    )
    parser.add_argument("--record", choices=sorted(RECORDS))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    args = parser.parse_args()

    if args.list:
        for name in RECORDS:
            item = _archive(name)
            print(f"{name}: {_human(item.size)} — {item.name}")
        return 0
    if not args.record or not args.download:
        parser.error("use --list, or provide --record NAME --download")
    try:
        download(_archive(args.record), args.output_dir, args.min_free_gb, args.extract)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
