from __future__ import annotations

import copy

import pytest
from histoforge.cloud.training.runs.request import training_run_request_digest
from histoforge.cloud.training.runs.source_snapshot import (
    validate_training_source_ready_marker,
)
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import STARTING_WEIGHT_DIGEST
from tests.test_training_run_request import run_request

RUN_ID = "44444444-4444-4444-8444-444444444444"


def source_ready_marker() -> dict:
    request = run_request()
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "request_digest": training_run_request_digest(request, training_api_limits()),
        "source_module": request["source_module"],
        "starting_weight_path": "source/model.safetensors",
        "starting_weight_digest": STARTING_WEIGHT_DIGEST,
        "source_files": [
            {
                "relative_path": "source/model.safetensors",
                "size_bytes": 16,
                "sha256": STARTING_WEIGHT_DIGEST,
            },
            {
                "relative_path": "source/selected-model.json",
                "size_bytes": 2,
                "sha256": (
                    "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
                ),
            },
        ],
        "report_digest": "b" * 64,
    }


def test_ready_marker_round_trip_binds_starting_weights() -> None:
    marker = source_ready_marker()

    assert validate_training_source_ready_marker(marker, 8 * 1024**2) == marker

    wrong = copy.deepcopy(marker)
    wrong["starting_weight_digest"] = "c" * 64
    with pytest.raises(ValueError, match="starting weights do not match"):
        validate_training_source_ready_marker(wrong, 8 * 1024**2)


def test_ready_marker_requires_sorted_contained_source_paths() -> None:
    marker = source_ready_marker()
    marker["source_files"].reverse()
    with pytest.raises(ValueError, match="sorted"):
        validate_training_source_ready_marker(marker, 8 * 1024**2)

    marker = source_ready_marker()
    marker["source_files"][0]["relative_path"] = "model.safetensors"
    marker["starting_weight_path"] = "model.safetensors"
    with pytest.raises(ValueError, match="below the source directory"):
        validate_training_source_ready_marker(marker, 8 * 1024**2)


def test_ready_marker_is_bounded() -> None:
    marker = source_ready_marker()
    with pytest.raises(ValueError, match="byte limit"):
        validate_training_source_ready_marker(marker, 64)
