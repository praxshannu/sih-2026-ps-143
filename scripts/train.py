"""Train the UNet++ oil-spill detector. The documented entry point.

Everything comes from configuration: this script parses flags, maps them onto
:class:`sentinel_core.config.Settings`, sets up logging, and hands off to
:func:`ml.training.train_detector.train`. It contains no training logic, so a
scheduled run and a hand-typed run cannot diverge.

The dataset is read in place. ``--data-source real`` reads the 1200-scene
archive from the external disk and never writes to it; the run fails if the
source directory changed while training. Nothing is copied to the local machine
at any point.

Examples::

    # a bounded real-data run: 24 scenes, 3 epochs, one batch per epoch
    python scripts/train.py --data-source real --epochs 3 \\
        --max-scenes 24 --max-steps-per-epoch 1

    # the synthetic set, full length
    python scripts/train.py --data-source synthetic --epochs 10

    # continue an interrupted run
    python scripts/train.py --resume runs/<run_id>/../checkpoints/.../detector_last.pth

Exit codes: 0 success, 2 a usage problem, 78 configuration, 1 anything else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from ml.training.train_detector import train  # noqa: E402
from sentinel_core import configure_logging, get_settings  # noqa: E402
from sentinel_core.config import TrainingSettings  # noqa: E402
from sentinel_core.errors import SentinelError  # noqa: E402
from sentinel_core.provenance import DataProvenance  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-source",
        choices=["real", "synthetic"],
        default=None,
        help="which dataset to train on; overrides SENTINEL_DATA_SOURCE",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default=None,
        help="auto prefers CUDA, then Apple MPS, then CPU",
    )
    parser.add_argument("--encoder", default=None)
    parser.add_argument(
        "--encoder-weights",
        choices=["imagenet", "none"],
        default=None,
        help="use none when the machine is offline",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="cap the number of pairs; 0 means no cap",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="cap optimiser steps per epoch; for smoke runs",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="disable shuffle and stochastic augmentation",
    )
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="rebuild the data-source index first (slow on the external disk)",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--json-logs", action="store_true", default=None)
    parser.add_argument("--log-every", type=int, default=10, help="log a batch every N steps")
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    """Translate flags into validated Settings fields.

    Only flags the user actually passed are included, so a flag left at its
    default never overrides the environment. ``--max-scenes 0`` is the explicit
    way to clear a cap that came from the environment.
    """
    training: dict[str, object] = {}
    mapping = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
        "encoder": args.encoder,
        "encoder_weights": args.encoder_weights,
        "max_steps_per_epoch": args.max_steps_per_epoch,
        "deterministic": args.deterministic,
    }
    for key, value in mapping.items():
        if value is not None:
            training[key] = value
    if args.max_scenes is not None:
        training["max_scenes"] = args.max_scenes or None

    overrides: dict[str, object] = {}
    if training:
        overrides["training"] = training
    if args.data_source is not None:
        overrides["data_source"] = args.data_source
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.log_level is not None:
        overrides["logging"] = {"level": args.log_level}
    if args.json_logs:
        overrides["logging"] = {**overrides.get("logging", {}), "json_output": True}
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    overrides = _overrides(args)

    # Nested groups are re-validated rather than patched in place: model_copy
    # skips validation, and a bad --image-size must fail here, not inside a
    # convolution four hours into the run.
    if "training" in overrides:
        merged = {**settings.training.model_dump(), **overrides["training"]}
        settings = settings.model_copy(update={"training": TrainingSettings(**merged)})
    if "logging" in overrides:
        merged = {**settings.logging.model_dump(), **overrides["logging"]}
        settings = settings.model_copy(update={"logging": type(settings.logging)(**merged)})
    for key in ("data_source", "run_name"):
        if key in overrides:
            settings = settings.model_copy(update={key: overrides[key]})

    configure_logging(
        service="train",
        level=settings.logging.level,
        json_output=settings.logging.json_output,
        log_dir=settings.resolved_log_dir(),
    )

    logger.info("configuration: {}", settings.describe())
    if not settings.datasources.allow_synthetic_fallback and settings.data_source == "real":
        logger.info("real data source is mandatory for this run; no fallback is configured")

    try:
        report = train(
            settings,
            data_source=args.data_source,
            epochs=args.epochs,
            max_scenes=args.max_scenes if args.max_scenes is not None else None,
            max_steps_per_epoch=args.max_steps_per_epoch,
            resume=args.resume,
            rebuild_index=args.rebuild_index,
            log_every=args.log_every,
        )
    except SentinelError as exc:
        logger.error("{}: {}", exc.reason, exc)
        for key, value in exc.context.items():
            logger.error("  {}={}", key, value)
        return exc.exit_code
    except KeyboardInterrupt:
        logger.warning("interrupted; the last checkpoint is complete and resumable")
        return 130

    marker = (
        "\u25a0" if report.provenance == DataProvenance.REAL.value else "\u25b2"
    )
    logger.info(
        "{} {} | best val IoU {:.4f} at epoch {} | {}",
        marker,
        report.provenance,
        report.best_val_iou,
        report.best_epoch + 1 if report.best_epoch >= 0 else -1,
        report.claim,
    )
    if not report.scientifically_valid:
        logger.warning(
            "these metrics are not a scientific result; provenance is {}",
            report.provenance,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
