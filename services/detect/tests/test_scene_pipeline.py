"""End-to-end inference tests against REAL Sentinel-1 scenes.

The unit tests around the individual operators all passed while the pipeline as
a whole was broken: `geometry.measure()` constructed `Measurement(elongation=…)`
against a dataclass field named `elongation_rotated_rect`, so *every valid
scene* died with a TypeError that `DetectPipeline.run` correctly swallowed into
an `invalid_scene` result. Nothing caught it because nothing ran a real scene
through the real geometry path.

That is what this file is for. The acceptance gate for this project is "a real
Sentinel-1 TIFF completes inference", so there is a test that asserts exactly
that, on the real GeoTIFFs in `data/sar/`.

`data/sar/` is gitignored, so these tests skip with a clear reason when the
scenes are absent. A skip here means "the scenes were not present", never
"it passed".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DETECT_DIR = ROOT / "services" / "detect"
SAR_DIR = ROOT / "data" / "sar"


def _purge_app_namespace() -> None:
    """Every service ships a top-level `app` package; purge before importing."""
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]


_purge_app_namespace()
if str(DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(DETECT_DIR))

pipeline_module = pytest.importorskip("app.pipeline")

DetectPipeline = pipeline_module.DetectPipeline
PipelineOptions = pipeline_module.PipelineOptions


def _real_scenes() -> list[Path]:
    if not SAR_DIR.is_dir():
        return []
    return sorted(SAR_DIR.glob("*.tif"))


_REAL_SCENES = _real_scenes()

requires_scenes = pytest.mark.skipif(
    not _REAL_SCENES,
    reason=(
        f"no real Sentinel-1 GeoTIFFs under {SAR_DIR} — that directory is "
        "gitignored, so this environment has no scenes to run. The live "
        "acceptance check must be re-run where data/sar is populated."
    ),
)


@pytest.fixture(scope="module")
def pipeline():
    return DetectPipeline()


# ── the acceptance gate ───────────────────────────────────────────────────


@requires_scenes
def test_real_sentinel1_scene_completes_inference(pipeline):
    """A real scene must reach a real verdict — not `invalid_scene`.

    `invalid_scene` is the state the pipeline uses for "something went wrong
    internally", so it is the specific thing to assert against: a scene that
    validates must never land there.
    """
    scene = next(
        (s for s in _REAL_SCENES if "20200810" in s.name),
        _REAL_SCENES[0],
    )

    result = pipeline.run(scene, PipelineOptions(with_evidence=True))

    assert result["state"] != "invalid_scene", (
        f"{scene.name} failed inference: {result['state_reason']} "
        f"(flags={result['flags']})"
    )
    assert result["state"] in {
        "ok",
        "no_detection",
        "low_confidence",
        "out_of_distribution",
        "missing_forcing_data",
    }
    assert result["validation"]["valid"] is True
    assert result["detector"]
    assert result["schema"] == "sentinel.detect.inference/v1"


@requires_scenes
def test_real_detections_carry_geometry_confidence_and_ci(pipeline):
    """Every detection needs the full evidence set, CI included.

    AGENTS.md: attribution and detection show Wilson 95% CIs, never a point
    estimate alone. That rule is enforced here rather than trusted.
    """
    scene = next((s for s in _REAL_SCENES if "20200810" in s.name), _REAL_SCENES[0])
    result = pipeline.run(scene, PipelineOptions(with_evidence=True))

    if result["count"] == 0:
        pytest.skip(f"{scene.name} produced no detections in this environment")

    for det in result["detections"]:
        for key in (
            "id",
            "geometry",
            "area_km2",
            "perimeter_km",
            "centroid",
            "length_km",
            "width_km",
            "orientation_deg",
            "confidence",
        ):
            assert key in det, f"detection missing {key}"

        assert det["area_km2"] > 0
        assert det["perimeter_km"] > 0
        assert det["length_km"] >= det["width_km"] > 0
        assert 0.0 <= det["orientation_deg"] <= 360.0

        conf = det["confidence"]
        assert "value" in conf
        # The CI is the point: a bare value is not an acceptable answer.
        assert conf.get("low") is not None and conf.get("high") is not None
        assert 0.0 <= conf["low"] <= conf["value"] <= conf["high"] <= 1.0
        assert conf.get("interval") == "wilson_95"

    # Georeferencing survives: the GeoJSON is WGS84 and geometrically sane.
    gj = result["geojson"]
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == result["count"]
    for feature in gj["features"]:
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
        coords = feature["geometry"]["coordinates"]
        assert coords, "empty geometry emitted"


@requires_scenes
def test_explanation_is_labelled_honestly(pipeline):
    """The evidence map must not be passed off as Grad-CAM.

    There is no trained checkpoint on this host, so no gradient exists to
    weight. The response has to say that rather than shipping a heatmap under
    a name that implies a neural attribution method.
    """
    scene = next((s for s in _REAL_SCENES if "20200810" in s.name), _REAL_SCENES[0])
    result = pipeline.run(scene, PipelineOptions(with_evidence=True))
    explanation = result["explanation"]

    assert explanation, "with_evidence=True must produce an explanation"
    assert explanation["method"] == "deterministic_evidence_map"
    assert explanation["gradcam"] is False
    assert explanation["not_gradcam_reason"]
    assert "grad" in explanation["not_gradcam_reason"].lower()


@requires_scenes
def test_missing_wind_is_flagged_not_guessed(pipeline):
    """Without wind the detector still runs, but must not imply it validated
    look-alike rejection — wind viability is the gate that separates oil from
    a wind shadow."""
    scene = next((s for s in _REAL_SCENES if "20200810" in s.name), _REAL_SCENES[0])
    result = pipeline.run(scene, PipelineOptions(wind_speed_ms=None))

    if result["count"] == 0:
        pytest.skip(f"{scene.name} produced no detections in this environment")

    assert "missing_wind_forcing" in result["flags"]
    for det in result["detections"]:
        assert det["wind_viability"] == "LOW_CONFIDENCE_NO_WIND"


@requires_scenes
def test_wind_in_viability_band_clears_the_flag(pipeline):
    """5 m/s sits inside the 2..10 m/s band where slicks are separable."""
    scene = next((s for s in _REAL_SCENES if "20200810" in s.name), _REAL_SCENES[0])
    result = pipeline.run(scene, PipelineOptions(wind_speed_ms=5.0))

    if result["count"] == 0:
        pytest.skip(f"{scene.name} produced no detections in this environment")

    assert "missing_wind_forcing" not in result["flags"]
    assert all(det["wind_viability"] == "OK" for det in result["detections"])


# ── failure states, which need no real data ───────────────────────────────


def test_missing_scene_is_invalid_not_a_traceback(pipeline, tmp_path: Path):
    result = pipeline.run(tmp_path / "does_not_exist.tif")
    assert result["state"] == "invalid_scene"
    assert result["count"] == 0
    assert result["state_reason"]
    assert result["validation"]["valid"] is False


def test_non_georeferenced_raster_is_rejected(pipeline, tmp_path: Path):
    """A raster with no CRS cannot be measured in metres or placed on a map,
    so it must be refused before inference rather than after."""
    import numpy as np
    import rasterio

    path = tmp_path / "no_crs.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=2,
        dtype="float32",
    ) as dst:
        dst.write(np.full((2, 64, 64), -14.0, dtype="float32"))

    result = pipeline.run(path)
    assert result["state"] == "invalid_scene"
    assert result["validation"]["valid"] is False
    assert result["validation"]["reason_code"]


def test_wrong_band_count_is_rejected(pipeline, tmp_path: Path):
    """Two bands (VV/VH) is the contract; a single-band raster is a different
    product and must not be silently accepted."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "one_band.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(57.6, -20.4, 0.0003, 0.0003),
    ) as dst:
        dst.write(np.full((1, 64, 64), -14.0, dtype="float32"))

    result = pipeline.run(path)
    assert result["state"] == "invalid_scene"
    assert result["validation"]["valid"] is False
