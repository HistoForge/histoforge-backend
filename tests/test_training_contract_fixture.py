import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_digest
from histoforge.cloud.training.contract import (
    validate_model_training_capability,
    validate_training_settings,
    validate_training_settings_for_capability,
)
from histoforge.cloud.training.datasets.manifest import (
    validate_training_dataset_manifest,
)
from histoforge.cloud.training.datasets.split import (
    validate_training_split_plan,
    validate_training_split_sidecar,
)
from histoforge.cloud.training.datasets.transfer import (
    training_dataset_chunk_range,
    training_dataset_start_digest,
    validate_training_dataset_chunk_receipt,
    validate_training_dataset_cleanup_record,
    validate_training_dataset_page,
    validate_training_dataset_record,
    validate_training_dataset_seal_request,
    validate_training_dataset_start_request,
    validate_training_dataset_start_result,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request_against_manifest_capability,
)


def test_shared_flow_setting_presence_and_exact_canonical_bytes():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/contracts/training-datasets-v1.json"
        ).read_text()
    )
    for case in fixture["flowSettingCases"]:
        capability = copy.deepcopy(fixture["capability"]["value"])
        if not case["advertised"]:
            capability.pop("flow_loss_weighting_options")
            capability["default_training_settings"].pop("flow_loss_weighting")
        settings = case["value"]
        if case["valid_syntax"]:
            assert validate_training_settings(settings) == settings, case["name"]
            assert canonical_digest(settings) == case["sha256"], case["name"]
        else:
            with pytest.raises(ValueError):
                validate_training_settings(settings)
        if case["allowed"]:
            assert (
                validate_training_settings_for_capability(settings, capability, 4096)
                == settings
            )
        else:
            with pytest.raises(ValueError):
                validate_training_settings_for_capability(settings, capability, 4096)


def test_training_dataset_fixture_freezes_every_cross_language_digest():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/contracts/training-datasets-v1.json"
        ).read_text()
    )
    limits = validate_training_api_limits(fixture["trainingApiLimits"])
    assert canonical_digest(limits) == fixture["trainingApiLimitsSha256"]

    capability = validate_model_training_capability(
        fixture["capability"]["value"], limits["report"]["max_text_bytes"]
    )
    assert canonical_digest(capability) == fixture["capability"]["sha256"]

    manifest = validate_training_dataset_manifest(fixture["manifest"]["value"], limits)
    assert manifest["schema_version"] == 1
    assert canonical_digest(manifest) == fixture["manifest"]["sha256"]

    split = validate_training_split_plan(fixture["split"]["value"], manifest, limits)
    assert canonical_digest(split) == fixture["split"]["sha256"]

    request = validate_training_run_request_against_manifest_capability(
        fixture["runRequest"]["value"], manifest, capability, limits
    )
    assert (
        training_run_request_digest(request, limits) == fixture["runRequest"]["sha256"]
    )

    start = validate_training_dataset_start_request(fixture["start"]["value"], limits)
    assert training_dataset_start_digest(start, limits) == fixture["start"]["sha256"]
    assert (
        training_dataset_chunk_range(start, 2, limits)
        == fixture["start"]["chunkRange2"]
    )

    retained = validate_training_dataset_record(fixture["retained"]["value"], limits)
    assert canonical_digest(retained) == fixture["retained"]["sha256"]

    for name in ("uploadAccepted", "existing"):
        start_result = validate_training_dataset_start_result(
            fixture[name]["value"], limits
        )
        assert canonical_digest(start_result) == fixture[name]["sha256"]

    seal_request = validate_training_dataset_seal_request(
        fixture["sealRequest"]["value"]
    )
    assert canonical_digest(seal_request) == fixture["sealRequest"]["sha256"]

    chunk_receipt = validate_training_dataset_chunk_receipt(
        fixture["chunkReceipt"]["value"]
    )
    assert canonical_digest(chunk_receipt) == fixture["chunkReceipt"]["sha256"]

    cleanup = validate_training_dataset_cleanup_record(
        fixture["cleanup"]["value"], limits
    )
    assert canonical_digest(cleanup) == fixture["cleanup"]["sha256"]

    page = validate_training_dataset_page(fixture["datasetPage"]["value"], limits)
    assert canonical_digest(page) == fixture["datasetPage"]["sha256"]

    sidecar = validate_training_split_sidecar(
        fixture["splitSidecar"]["value"], manifest, limits
    )
    assert canonical_digest(sidecar) == fixture["splitSidecar"]["sha256"]


def test_schema_two_manifest_uses_the_same_package_graph_and_digest_contract():
    directory = Path(__file__).parent / "fixtures/contracts"
    legacy = json.loads((directory / "training-datasets-v1.json").read_text())
    fixture = json.loads((directory / "training-dataset-manifest-v2.json").read_text())
    manifest = validate_training_dataset_manifest(
        fixture["value"], legacy["trainingApiLimits"]
    )
    assert manifest["schema_version"] == 2
    assert canonical_digest(manifest) == fixture["sha256"]
    assert manifest["dataset_digest"] != legacy["manifest"]["value"]["dataset_digest"]
