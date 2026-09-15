"""Generate a synthetic SAR dataset that matches the real archive's distribution.

Four steps, and the fourth is the point:

1. resolve the **real** data source (read in place from the external disk);
2. **profile** it — per-band dB statistics, sea level, oil contrast, oil
   coverage, blob geometry, effective looks;
3. **generate** a synthetic dataset by sampling those measurements;
4. **profile the synthetic dataset with the same code** and compare the two
   with explicit tolerances, writing ``match_report.json``.

Step 4 is what makes "matches the real distribution" a claim someone can check
instead of an adjective. Both sides are measured by one function, so a
difference in the report is a difference in the data.

Usage::

    python scripts/generate_synthetic.py --scenes 120
    python scripts/generate_synthetic.py --scenes 40 --sample 16 --force
    python scripts/generate_synthetic.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from ml.synth import (  # noqa: E402
    GenerationConfig,
    compare_profiles,
    generate_dataset,
    profile_from_spec,
)
from sentinel_core import configure_logging, get_settings  # noqa: E402
from sentinel_core.datasource import resolve_source  # noqa: E402
from sentinel_core.errors import SentinelError  # noqa: E402
from sentinel_core.provenance import DataProvenance, synthetic_warning  # noqa: E402


def _profile_path(root: Path) -> Path:
    return root / "distribution.json"


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", type=int, default=120, help="number of synthetic scenes")
    parser.add_argument("--size", type=int, default=2048, help="scene edge in pixels")
    parser.add_argument("--sample", type=int, default=32, help="real scenes to profile")
    parser.add_argument("--seed", type=int, default=settings.training.seed)
    parser.add_argument("--out", type=Path, default=settings.datasources.synthetic_root)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--force", action="store_true", help="overwrite an existing dataset")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-profile the existing synthetic dataset and compare; generate nothing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any statistic falls outside tolerance",
    )
    args = parser.parse_args()

    configure_logging(
        service="generate-synthetic",
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        log_dir=settings.resolved_log_dir(),
    )

    out: Path = args.out
    profile_file = _profile_path(out)
    summary_file = out / "source.json"

    if not args.verify_only:
        if out.is_dir() and (out / "manifest.jsonl").is_file() and not args.force:
            logger.warning(
                "{} already holds a dataset; use --force to regenerate or "
                "--verify-only to check the existing one",
                out,
            )
            return 1

        logger.info("resolving the real data source (read in place, nothing is copied)")
        real = resolve_source("real", settings=settings)
        if real.is_synthetic:
            raise SentinelError(
                "the configured real source resolved to synthetic data; refusing to "
                "fit synthetic data to synthetic data",
                reason="real_source_is_synthetic",
            )
        logger.info(
            "real archive: {} pairs at {} ({})",
            real.n_pairs,
            real.images_dir,
            real.provenance.value,
        )

        logger.info("step 1/3 — profiling {} real scene(s)", args.sample)
        started = time.monotonic()
        profile = profile_from_spec(real, sample=args.sample, seed=args.seed)
        logger.info("profiled in {:.1f}s", time.monotonic() - started)

        logger.info("step 2/3 — generating {} synthetic scene(s)", args.scenes)
        started = time.monotonic()
        config = GenerationConfig(
            n_scenes=args.scenes,
            train_fraction=args.train_fraction,
            size=args.size,
            seed=args.seed,
        )
        summary = generate_dataset(profile, out, config)
        logger.info(
            "generated {} scene(s) in {:.1f}s ({} train / {} val / {} test)",
            args.scenes,
            time.monotonic() - started,
            summary["split_counts"]["train"],
            summary["split_counts"]["val"],
            summary["split_counts"]["test"],
        )
        if not profile_file.is_file():
            profile.save(profile_file)
    else:
        if not profile_file.is_file():
            raise SentinelError(
                f"{profile_file} is missing; run a full generation first",
                reason="profile_missing",
            )
        if summary_file.is_file():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        else:
            summary = {}

    logger.info("step 3/3 — verifying the synthetic dataset against the real profile")
    from ml.synth.profile import DistributionProfile

    real_profile = DistributionProfile.load(profile_file)
    synthetic_spec = resolve_source("synthetic", settings=settings, rebuild=True)
    synthetic_profile = profile_from_spec(
        synthetic_spec, sample=min(args.sample, synthetic_spec.n_pairs), seed=args.seed
    )

    report = compare_profiles(real_profile, synthetic_profile)
    report["real_source"] = {
        "n_pairs": real_profile.source.get("n_pairs_total"),
        "sampled": real_profile.n_scenes_sampled,
        "images_dir": real_profile.source.get("images_dir"),
    }
    report["synthetic_dataset"] = {
        "root": str(out),
        "n_pairs": synthetic_spec.n_pairs,
        "bands": synthetic_spec.bands,
        "provenance": synthetic_spec.provenance.value,
    }
    report["provenance"] = DataProvenance.SYNTHETIC.value
    report["warning"] = synthetic_warning("synthetic dataset")

    report_path = out / "match_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "match: {}/{} checks within tolerance (fraction {:.2f})",
        report["n_passed"],
        report["n_checks"],
        report["match_fraction"],
    )
    for check in report["failed"]:
        logger.warning(
            "  outside tolerance: {} real={} synthetic={} delta={} (tol {})",
            check["quantity"],
            check["real"],
            check["synthetic"],
            check["delta"],
            check["tolerance"],
        )

    logger.info("report: {}", report_path)
    if args.strict and report["n_failed"]:
        logger.error("{} statistic(s) outside tolerance", report["n_failed"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
