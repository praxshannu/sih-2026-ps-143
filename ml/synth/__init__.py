"""Synthetic dataset generation for SENTINEL.

The package exists so the real and synthetic data paths are interchangeable at
runtime (``sentinel_core.datasource``), which requires the synthetic data to
match the real data's *structure* (2-band float32 dB GeoTIFFs at 2048x2048,
paired masks, same manifest contract) and its *distribution* (per-band
statistics, sea level, oil-to-sea contrast, oil coverage, blob geometry).

``profile.py`` measures the real archive; ``generate.py`` samples those
measurements. The fitted parameters are written next to the generated data so
the match can be checked rather than asserted.
"""

from __future__ import annotations

from ml.synth.generate import GenerationConfig, generate_dataset, synthesize_scene
from ml.synth.profile import (
    DistributionProfile,
    compare_profiles,
    profile_from_spec,
    profile_scenes,
)

__all__ = [
    "DistributionProfile",
    "GenerationConfig",
    "compare_profiles",
    "generate_dataset",
    "profile_from_spec",
    "profile_scenes",
    "synthesize_scene",
]
