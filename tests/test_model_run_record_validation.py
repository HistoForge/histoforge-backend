"""Growing run records share validation work without weakening recovery checks."""

import copy
from unittest.mock import patch

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure, build_gateway_error
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.runs import publications
from histoforge.cloud.inference.runs.publications import (
    model_run_execution_identity_digest,
    model_run_execution_identity_digests,
    validate_model_input_result,
    validate_model_run_checkpoint,
    validate_model_run_publications,
    validate_model_run_unit_failure,
    validate_model_run_unit_result,
)
from histoforge.cloud.inference.runs.records import validate_model_run_record
from histoforge.cloud.inference.runs.reuse import select_reusable_model_run_state
from histoforge.cloud.inference.runs.store import ModelRunStore

from test_model_run_source_delivery import (
    _binding,
    _captured_request,
    _input_result,
    _unit,
)
from test_model_run_store import (
    _Clock,
    _finite_publications,
    _limits,
    _request,
    _reserve,
)


@pytest.mark.parametrize("kind", ["finite", "core-halo", "global", "captured"])
def test_collection_validation_matches_single_publication_validators(tmp_path, kind):
    request = (
        _captured_request()
        if kind == "captured"
        else _request({"finite": 0, "core-halo": 1, "global": 2}[kind])
    )
    files = ArtifactFileStore(tmp_path)
    bindings = (
        [_binding(request, unit_id) for unit_id in (0, 1)] if kind == "captured" else []
    )
    values = {
        "source_bindings": bindings,
        "completed_units": [],
        "checkpoints": [],
        "failed_units": [],
        "completed_inputs": [],
    }
    if kind == "captured":
        values["completed_units"] = [_unit(request, files, bindings, 0)]
        values["completed_inputs"] = [_input_result(request, files, bindings)]
    elif kind == "global":
        input_id = request["inputs"][0]["input_id"]
        values["checkpoints"] = [
            {
                "input_id": input_id,
                "unit_id": 0,
                "sequence": 1,
                "state": files.publish_artifact_bytes(
                    b"checkpoint", media_type="application/octet-stream"
                ),
                "execution_identity_digest": model_run_execution_identity_digest(
                    request, input_id, 0, _limits()
                ),
            }
        ]
    else:
        unit, result = _finite_publications(request, files)
        values["completed_units" if kind == "core-halo" else "completed_inputs"] = [
            unit if kind == "core-halo" else result
        ]
    values["failed_units"] = [
        {
            "input_id": request["inputs"][0]["input_id"],
            "unit_id": 0,
            "failure": build_gateway_error("RUNNER_FAILED", "prediction failed"),
        }
    ]
    validators = {
        "completed_units": lambda value: validate_model_run_unit_result(
            value, request, _limits(), bindings
        ),
        "checkpoints": lambda value: validate_model_run_checkpoint(
            value, request, _limits()
        ),
        "failed_units": lambda value: validate_model_run_unit_failure(
            value, request, _limits()
        ),
        "completed_inputs": lambda value: validate_model_input_result(
            value, request, _limits(), bindings
        ),
    }
    expected = {
        "source_bindings": bindings,
        **{
            name: [validator(value) for value in values[name]]
            for name, validator in validators.items()
        },
    }
    with patch.object(
        publications, "_publication_context", wraps=publications._publication_context
    ) as context:
        with patch.object(
            publications,
            "validate_run_source_bindings",
            wraps=publications.validate_run_source_bindings,
        ) as sources:
            actual = validate_model_run_publications(values, request, _limits())
    assert canonical_bytes(actual) == canonical_bytes(expected)
    assert context.call_count == sources.call_count == 1


@pytest.mark.parametrize("kind", ["finite", "core-halo", "global", "captured"])
def test_batch_unit_identities_match_exact_single_unit_preimages(kind):
    request = (
        _captured_request()
        if kind == "captured"
        else _request({"finite": 0, "core-halo": 1, "global": 2}[kind])
    )
    bindings = [_binding(request, 0)] if kind == "captured" else []
    input_id = request["inputs"][0]["input_id"]
    keys = [(input_id, 0), (input_id, 1), (input_id + 1, 0)]
    actual = model_run_execution_identity_digests(request, keys, _limits(), bindings)
    if kind == "captured":
        assert actual == {}
        for unit in (0, 1):
            with pytest.raises(ValueError, match="every input source"):
                model_run_execution_identity_digest(
                    request, input_id, unit, _limits(), bindings
                )
        complete = bindings + [_binding(request, 1)]
        assert model_run_execution_identity_digests(
            request, keys, _limits(), complete
        ) == {
            (input_id, unit): model_run_execution_identity_digest(
                request, input_id, unit, _limits(), complete
            )
            for unit in (0, 1)
        }
        malformed = copy.deepcopy(bindings)
        malformed[0]["source"]["capture_digest"] = "0" * 64
        with pytest.raises(ValueError):
            model_run_execution_identity_digests(request, keys, _limits(), malformed)
    else:
        assert actual == {
            (input_id, 0): model_run_execution_identity_digest(
                request, input_id, 0, _limits(), bindings
            )
        }


@pytest.mark.parametrize(
    "corruption",
    [
        "identity",
        "missing-source",
        "coverage",
        "capture",
        "duplicate",
        "order",
        "tuple",
        "over-budget",
        "completed-and-failed",
        "queued",
        "request-artifact",
    ],
)
def test_private_record_checks_still_reject_corruption(tmp_path, corruption):
    request = _captured_request()
    files = ArtifactFileStore(tmp_path)
    values = {}
    store = ModelRunStore(values, max_active_runs=4, clock=_Clock())
    _reserve(store, request, _limits(), 600, files)
    record = copy.deepcopy(next(iter(values.values())))
    bindings = [_binding(request, unit_id) for unit_id in (0, 1)]
    record.update(
        phase="RUNNING",
        source_bindings=bindings,
        completed_units=[
            _unit(request, files, bindings, unit_id) for unit_id in (0, 1)
        ],
    )
    assert canonical_bytes(validate_model_run_record(record)) == canonical_bytes(record)
    if corruption == "identity":
        record["completed_units"][0]["execution_identity_digest"] = "0" * 64
    elif corruption == "missing-source":
        record["source_bindings"].pop()
    elif corruption == "coverage":
        record["source_bindings"][0]["source"]["source_rect_at_level"]["width"] = 1
    elif corruption == "capture":
        record["source_bindings"][0]["source"]["capture_digest"] = "0" * 64
    elif corruption == "duplicate":
        record["completed_units"].append(record["completed_units"][-1])
    elif corruption == "order":
        record["completed_units"].reverse()
    elif corruption == "tuple":
        record["completed_units"] = tuple(record["completed_units"])
    elif corruption == "over-budget":
        record["limits"]["max_recovery_state_bytes_per_run"] = 1
    elif corruption == "completed-and-failed":
        record["failed_units"] = [
            {
                "input_id": bindings[0]["input_id"],
                "unit_id": 0,
                "failure": build_gateway_error("RUNNER_FAILED", "failed"),
            }
        ]
    elif corruption == "queued":
        record["phase"] = "QUEUED"
    else:
        record["request_artifact"]["sha256"] = "0" * 64
    with pytest.raises(GatewayFailure) as rejected:
        validate_model_run_record(record)
    assert rejected.value.error["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize("completed_input", [False, True])
@pytest.mark.parametrize("change", ["backend", "halo", "generation", "bad-capture"])
def test_replacement_preserves_incompatible_skip_and_source_conflict(
    tmp_path, completed_input, change
):
    request = _captured_request()
    files = ArtifactFileStore(tmp_path)
    values = {}
    store = ModelRunStore(values, max_active_runs=4, clock=_Clock())
    _reserve(store, request, _limits(), 600, files)
    bindings = [_binding(request, unit_id) for unit_id in (0, 1)]
    for binding in bindings:
        store.bind_source(
            request["model_run_id"], binding, files, require_source=lambda *_: None
        )
    for unit_id in (0, 1) if completed_input else (0,):
        store.publish_unit(
            request["model_run_id"], _unit(request, files, bindings, unit_id), files
        )
    if completed_input:
        store.publish_input(
            request["model_run_id"], _input_result(request, files, bindings), 0, files
        )
    if completed_input:
        store.complete(request["model_run_id"], files)
    else:
        store.fail(
            request["model_run_id"],
            build_gateway_error("RUNNER_FAILED", "failed"),
            files,
        )
    previous = copy.deepcopy(next(iter(values.values())))
    replacement = copy.deepcopy(request)
    replacement["model_run_id"] = "99999999-9999-4999-8999-999999999999"
    replacement["resume_from"] = {
        "previous_run_id": request["model_run_id"],
        "previous_manifest_digest": previous["manifest"]["manifest_digest"],
    }
    if change == "backend":
        replacement["expected_backend"]["preprocessing_contract_digest"] = "b" * 64
    elif change == "halo":
        replacement["halo_px_override"] = 0
    elif change == "generation":
        replacement["inputs"][0]["output_generation_id"] = (
            "88888888-8888-4888-8888-888888888888"
        )
    else:
        previous["source_bindings"][0]["source"]["capture_digest"] = "0" * 64
        with pytest.raises(GatewayFailure) as conflict:
            select_reusable_model_run_state(previous, replacement, _limits(), files)
        assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"
        return
    selected = select_reusable_model_run_state(previous, replacement, _limits(), files)
    assert selected["completed_units"] == selected["completed_inputs"] == []
    assert selected["source_bindings"] == bindings
