"""Provenance tests, including the contract check against the ingest module.

The interesting test here is :func:`test_wire_values_match_the_ingest_contract`.
``sentinel_core.provenance`` re-declares four strings that
``services/ingest/app/provenance.py`` already owns, because that module lives
inside a service package that is only importable with the service directory on
``sys.path``. Duplicated contracts drift; this test is what stops it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from sentinel_core.config import repo_root
from sentinel_core.provenance import (
    PROVENANCE_NO_COVERAGE,
    PROVENANCE_REAL,
    PROVENANCE_SYNTHETIC,
    PROVENANCE_UNAVAILABLE,
    DataProvenance,
    synthetic_warning,
)


def _load_ingest_provenance():
    """Import the ingest service's provenance module without importing the app."""
    path = repo_root() / "services" / "ingest" / "app" / "provenance.py"
    if not path.is_file():  # pragma: no cover - only if the service is removed
        pytest.skip("ingest provenance module not present")
    spec = importlib.util.spec_from_file_location("_ingest_provenance_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves `cls.__module__` through sys.modules when processing
    # a frozen dataclass, so the module must be registered before it executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_wire_values_match_the_ingest_contract() -> None:
    ingest = _load_ingest_provenance()
    assert DataProvenance.REAL.value == ingest.PROVENANCE_REAL
    assert DataProvenance.SYNTHETIC.value == ingest.PROVENANCE_SYNTHETIC
    assert DataProvenance.UNAVAILABLE.value == ingest.PROVENANCE_UNAVAILABLE
    assert DataProvenance.NO_REAL_COVERAGE.value == ingest.PROVENANCE_NO_COVERAGE
    assert PROVENANCE_REAL == ingest.PROVENANCE_REAL
    assert PROVENANCE_SYNTHETIC == ingest.PROVENANCE_SYNTHETIC
    assert PROVENANCE_UNAVAILABLE == ingest.PROVENANCE_UNAVAILABLE
    assert PROVENANCE_NO_COVERAGE == ingest.PROVENANCE_NO_COVERAGE


def test_only_real_data_supports_a_scientific_claim() -> None:
    assert DataProvenance.REAL.supports_scientific_claim is True
    for other in (
        DataProvenance.SYNTHETIC,
        DataProvenance.UNAVAILABLE,
        DataProvenance.NO_REAL_COVERAGE,
    ):
        assert other.supports_scientific_claim is False, other


def test_only_synthetic_is_flagged_synthetic() -> None:
    assert DataProvenance.SYNTHETIC.is_synthetic is True
    assert DataProvenance.REAL.is_synthetic is False


def test_glyphs_match_the_ui_legend() -> None:
    assert DataProvenance.REAL.glyph == "■"
    assert DataProvenance.SYNTHETIC.glyph == "▲"
    assert DataProvenance.UNAVAILABLE.glyph == "□"


def test_enum_serialises_as_its_wire_value() -> None:
    # It subclasses str, so json.dumps must not need a custom encoder.
    import json

    assert json.dumps({"p": DataProvenance.SYNTHETIC}) == '{"p": "synthetic_mock"}'


def test_describe_carries_the_verdict() -> None:
    described = DataProvenance.SYNTHETIC.describe()
    assert described["provenance"] == "synthetic_mock"
    assert described["is_synthetic"] is True
    assert described["supports_scientific_claim"] is False


def test_synthetic_warning_names_the_source_and_refuses_evidence_use() -> None:
    text = synthetic_warning("synthetic dataset")
    assert "synthetic dataset" in text
    assert "MUST NOT" in text
    assert text.startswith("SYNTHETIC DATA")
