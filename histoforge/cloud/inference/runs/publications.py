"""Immutable durable model-run publication and manifest contracts.

The backend publishes reusable unit state, global checkpoints, complete input
feature streams, and one terminal manifest. This module owns their
canonical shape and cross-record identity. It performs no storage, dispatch,
cleanup, range IO, or terminal compare-and-set.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_nonnegative_safe_integer,
    require_positive_safe_integer,
    require_sha256,
)
from histoforge.cloud._errors import validate_gateway_error
from histoforge.cloud.architectures.registry import uses_target_normalization
from histoforge.cloud.artifacts.refs import validate_artifact_ref
from histoforge.cloud.execution_limits import validate_model_run_limits
from histoforge.cloud.inference.contract import (
    build_backend_input_identity,
    validate_backend_input_identity,
)
from histoforge.cloud.inference.identity import validate_backend_model_identity
from histoforge.cloud.inference.plans import (
    build_expected_model_execution_plan,
    validate_model_execution_plan,
)
from histoforge.cloud.inference.runs.request import (
    model_run_request_digest,
    validate_model_run_input,
    validate_model_run_resource_limits,
)
from histoforge.cloud.inference.runs.source_bindings import (
    input_source_bindings,
    unit_source_binding,
    validate_run_source_bindings,
)
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION

FEATURE_STREAM_MEDIA_TYPE = "application/geo+json-seq"
FEATURE_STREAM_CONTENT_ENCODING = "gzip"

_UNIT_RESULT_FIELDS = {
    "input_id",
    "unit_id",
    "reusable_state",
    "execution_identity_digest",
}
_CHECKPOINT_FIELDS = {
    "input_id",
    "unit_id",
    "sequence",
    "state",
    "execution_identity_digest",
}
_UNIT_FAILURE_FIELDS = {"input_id", "unit_id", "failure"}
_INPUT_RESULT_FIELDS = {
    "input_id",
    "input_identity",
    "execution_plan",
    "feature_stream_artifact",
    "feature_count",
    "vertex_count",
}
_MANIFEST_PREIMAGE_FIELDS = {
    "schema_version",
    "model_run_id",
    "action_id",
    "backend",
    "request_digest",
    "completed_units",
    "checkpoints",
    "failed_units",
    "completed_inputs",
    "complete",
}
_MANIFEST_FIELDS = _MANIFEST_PREIMAGE_FIELDS | {"manifest_digest"}
_MANIFEST_OPTIONAL_FIELDS = {"failure", "source_bindings"}


def preprocessing_context_digest(model_input: Any, backend: Any) -> str:
    """Bind preprocessing behavior to one exact immutable raster input."""

    item = validate_model_run_input(model_input)
    normalized_backend = validate_backend_model_identity(backend)
    raster_input = item["raster_input"]
    # The complete source set does not exist at admission. Planning depends only
    # on the captured image/grid, never on when its files arrive.
    input_identity = (
        {
            "source_view_digest": raster_input["source_view"]["view_digest"],
            "model_raster_input_digest": raster_input["input_digest"],
        }
        if "capture" in raster_input["source_view"]
        else build_backend_input_identity(raster_input)
    )
    return canonical_digest(
        {
            "input_identity": input_identity,
            "preprocessing_contract_digest": normalized_backend[
                "preprocessing_contract_digest"
            ],
        }
    )


def build_expected_model_execution_plans(
    request: Any, model_run_limits: Any
) -> dict[int, dict[str, Any]]:
    """Derive every bounded plan from the accepted request and runner limits."""

    normalized = validate_model_run_resource_limits(request, model_run_limits)
    limits = validate_model_run_limits(model_run_limits)
    profile = normalized.get("expected_profile")
    plans: dict[int, dict[str, Any]] = {}
    for item in normalized["inputs"]:
        raster = item["raster_input"]["frame"]["transform"]["raster"]
        plans[item["input_id"]] = build_expected_model_execution_plan(
            profile,
            raster,
            preprocessing_context_digest(item, normalized["expected_backend"]),
            limits["max_units"],
            halo_px_override=normalized.get("halo_px_override"),
        )
    return plans


def model_run_execution_identity_digest(
    request: Any,
    input_id: Any,
    unit_id: Any,
    model_run_limits: Any,
    source_bindings=(),
) -> str:
    """Return the exact-compatible reusable-state identity for one plan unit."""

    normalized = validate_model_run_resource_limits(request, model_run_limits)
    requested_input_id = require_nonnegative_int32(input_id, "input_id")
    requested_unit_id = require_nonnegative_int32(unit_id, "unit_id")
    inputs = {item["input_id"]: item for item in normalized["inputs"]}
    if requested_input_id not in inputs:
        raise ValueError("execution identity input_id is absent from model run request")
    plans = build_expected_model_execution_plans(normalized, model_run_limits)
    plan = plans[requested_input_id]
    if requested_unit_id not in _plan_unit_ids(plan):
        raise ValueError("execution identity unit_id is absent from generated plan")
    digest = _execution_identity_digester(
        normalized,
        inputs,
        plans,
        validate_run_source_bindings(source_bindings, inputs, plans),
    )
    return digest(requested_input_id, requested_unit_id)


def model_run_execution_identity_digests(
    request: Any,
    unit_keys: Sequence[tuple[int, int]],
    model_run_limits: Any,
    source_bindings=(),
) -> dict[tuple[int, int], str]:
    """Compute requested unit identities from one validated request and source set.

    Replacement runs can have a different plan or fewer accepted source files.
    Omit a requested unit when it is absent from that plan or lacks its required
    file; malformed requests and source assignments still fail validation.
    """

    submitted, inputs, plans = _publication_context(request, model_run_limits)
    bindings = validate_run_source_bindings(source_bindings, inputs, plans)
    bound_keys = {(item["input_id"], item["unit_id"]) for item in bindings}
    digest = _execution_identity_digester(submitted, inputs, plans, bindings)
    identities = {}
    for input_id, unit_id in unit_keys:
        input_id = require_nonnegative_int32(input_id, "input_id")
        unit_id = require_nonnegative_int32(unit_id, "unit_id")
        if input_id not in plans or unit_id not in _plan_unit_ids(plans[input_id]):
            continue
        key = (input_id, unit_id)
        model_input = inputs[input_id]
        if "capture" in model_input["raster_input"]["source_view"] and any(
            (input_id, required) not in bound_keys
            for required in (
                _plan_unit_ids(plans[input_id])
                if uses_target_normalization(
                    submitted["expected_backend"]["architecture"],
                    plans[input_id]["type"],
                )
                else (unit_id,)
            )
        ):
            continue
        identities[key] = digest(input_id, unit_id)
    return identities


def build_model_result_manifest(
    value: Any, request: Any, model_run_limits: Any
) -> dict[str, Any]:
    """Build one canonical self-digested complete or partial terminal manifest."""

    preimage = _require_exact_object(
        value,
        _MANIFEST_PREIMAGE_FIELDS,
        "model result manifest preimage",
        optional_fields=_MANIFEST_OPTIONAL_FIELDS,
    )
    normalized = _normalize_manifest_preimage(preimage, request, model_run_limits)
    return canonical_tree(
        {**normalized, "manifest_digest": canonical_digest(normalized)}
    )


def validate_model_result_manifest(
    value: Any, request: Any, model_run_limits: Any
) -> dict[str, Any]:
    """Validate a terminal manifest against its exact request, plans, and caps."""

    manifest = _require_exact_object(
        value,
        _MANIFEST_FIELDS,
        "model result manifest",
        optional_fields=_MANIFEST_OPTIONAL_FIELDS,
    )
    expected = build_model_result_manifest(
        {field: item for field, item in manifest.items() if field != "manifest_digest"},
        request,
        model_run_limits,
    )
    digest = require_sha256(manifest["manifest_digest"], "manifest_digest")
    if digest != expected["manifest_digest"]:
        raise ValueError("manifest_digest does not match model result manifest fields")
    return expected


def model_result_manifest_completed_unit_count(manifest: Any) -> int:
    """Project accepted plan-unit progress from one validated manifest tree."""

    value = _require_object(manifest, "model result manifest")
    completed = len(_require_array(value.get("completed_units"), "completed_units"))
    for item in _require_array(value.get("completed_inputs"), "completed_inputs"):
        result = _require_object(item, "completed input")
        plan = validate_model_execution_plan(result["execution_plan"])
        completed += len(_plan_unit_ids(plan))
    return completed


def model_result_manifest_digest(
    value: Any, request: Any, model_run_limits: Any
) -> str:
    """Return the verified self-digest of one terminal manifest."""

    return validate_model_result_manifest(value, request, model_run_limits)[
        "manifest_digest"
    ]


def validate_model_run_unit_result(
    value: Any,
    request: Any,
    model_run_limits: Any,
    source_bindings=(),
) -> dict[str, Any]:
    """Validate one core/halo reusable-state publication in request context."""

    submitted, inputs, plans = _publication_context(request, model_run_limits)
    bindings = validate_run_source_bindings(source_bindings, inputs, plans)
    return _validate_unit_result(
        value, plans, _execution_identity_digester(submitted, inputs, plans, bindings)
    )


def validate_model_run_checkpoint(
    value: Any, request: Any, model_run_limits: Any
) -> dict[str, Any]:
    """Validate one global accumulator checkpoint in request context."""

    submitted, inputs, plans = _publication_context(request, model_run_limits)
    return _validate_checkpoint(
        value, plans, _execution_identity_digester(submitted, inputs, plans)
    )


def validate_model_run_unit_failure(
    value: Any, request: Any, model_run_limits: Any
) -> dict[str, Any]:
    """Validate one coded plan-unit failure in request context."""

    _submitted, _inputs, plans = _publication_context(request, model_run_limits)
    return _validate_unit_failure(value, plans)


def validate_model_input_result(
    value: Any,
    request: Any,
    model_run_limits: Any,
    source_bindings=(),
) -> dict[str, Any]:
    """Validate one complete logical-input feature publication in context."""

    _submitted, inputs, plans = _publication_context(request, model_run_limits)
    bindings = validate_run_source_bindings(source_bindings, inputs, plans)
    return _validate_input_result(value, inputs, plans, bindings)


def validate_model_run_publications(
    value: Mapping[str, Any], request: Any, model_run_limits: Any
) -> dict[str, list[dict[str, Any]]]:
    """Validate a run's publications without rebuilding its plan for each item.

    Every item uses the same checked request, processing plans and uploaded-file
    assignments. The record or manifest owner still checks collection order,
    incompatible combinations, lifecycle state and aggregate resource limits.
    """

    submitted, inputs, plans = _publication_context(request, model_run_limits)
    bindings = validate_run_source_bindings(value["source_bindings"], inputs, plans)
    return {
        "source_bindings": bindings,
        **_normalize_publication_collections(value, submitted, inputs, plans, bindings),
    }


def _publication_context(
    request: Any, model_run_limits: Any
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    submitted = validate_model_run_resource_limits(request, model_run_limits)
    return (
        submitted,
        {item["input_id"]: item for item in submitted["inputs"]},
        build_expected_model_execution_plans(submitted, model_run_limits),
    )


def _normalize_manifest_preimage(
    value: Mapping[str, Any], request: Any, model_run_limits: Any
) -> dict[str, Any]:
    submitted = validate_model_run_resource_limits(request, model_run_limits)
    limits = validate_model_run_limits(model_run_limits)
    plans = build_expected_model_execution_plans(submitted, limits)
    inputs = {item["input_id"]: item for item in submitted["inputs"]}
    captures = any(
        "capture" in item["raster_input"]["source_view"] for item in inputs.values()
    )
    if captures != ("source_bindings" in value):
        raise ValueError("captured inputs require source_bindings in their manifest")
    bindings = validate_run_source_bindings(
        value.get("source_bindings", ()), inputs, plans
    )

    _require_current_schema(value["schema_version"])
    model_run_id = require_canonical_uuid(value["model_run_id"], "model_run_id")
    action_id = require_canonical_uuid(value["action_id"], "action_id")
    backend = validate_backend_model_identity(value["backend"])
    request_digest = require_sha256(value["request_digest"], "request_digest")
    if model_run_id != submitted["model_run_id"]:
        raise ValueError("manifest model_run_id does not match model run request")
    if action_id != submitted["action_id"]:
        raise ValueError("manifest action_id does not match model run request")
    if backend != submitted["expected_backend"]:
        raise ValueError("manifest backend does not match model run request")
    if request_digest != model_run_request_digest(submitted):
        raise ValueError("manifest request_digest does not match model run request")

    publications = _normalize_publication_collections(
        value, submitted, inputs, plans, bindings
    )
    completed_units = publications["completed_units"]
    checkpoints = publications["checkpoints"]
    failed_units = publications["failed_units"]
    completed_inputs = publications["completed_inputs"]
    _require_canonical_unique_order(completed_units, "completed_units")
    _require_canonical_unique_order(checkpoints, "checkpoints")
    _require_canonical_unique_order(failed_units, "failed_units")
    _require_canonical_unique_input_order(completed_inputs)

    completed_input_ids = {item["input_id"] for item in completed_inputs}
    for collection_name, collection in (
        ("completed_units", completed_units),
        ("checkpoints", checkpoints),
        ("failed_units", failed_units),
    ):
        if any(item["input_id"] in completed_input_ids for item in collection):
            raise ValueError(f"{collection_name} cannot name a completed input")

    completed_unit_keys = {
        (item["input_id"], item["unit_id"]) for item in completed_units
    }
    failed_unit_keys = {(item["input_id"], item["unit_id"]) for item in failed_units}
    if completed_unit_keys & failed_unit_keys:
        raise ValueError("one core/halo unit cannot be both completed and failed")

    complete = _require_bool(value["complete"], "complete")
    failure = validate_gateway_error(value["failure"]) if "failure" in value else None
    request_input_ids = set(inputs)
    if complete:
        if failure is not None:
            raise ValueError("complete manifest cannot carry failure")
        if completed_input_ids != request_input_ids:
            raise ValueError("complete manifest must cover every requested input")
        if completed_units or checkpoints or failed_units:
            raise ValueError("complete manifest cannot retain unit state or failures")
    else:
        if completed_input_ids == request_input_ids:
            raise ValueError("manifest covering every requested input must be complete")
        if not (completed_inputs or completed_units or checkpoints):
            raise ValueError("partial manifest must preserve reusable state")

    recovery_sizes = [
        item["reusable_state"]["size_bytes"] for item in completed_units
    ] + [item["state"]["size_bytes"] for item in checkpoints]
    if any(
        size > limits["max_recovery_state_bytes_per_unit"] for size in recovery_sizes
    ):
        raise ValueError("manifest recovery state exceeds per-unit byte limit")
    if sum(recovery_sizes) > limits["max_recovery_state_bytes_per_run"]:
        raise ValueError("manifest recovery state exceeds run byte limit")

    result_budget = submitted["result_budget"]
    if (
        sum(item["feature_stream_artifact"]["size_bytes"] for item in completed_inputs)
        > result_budget["max_transferred_bytes"]
    ):
        raise ValueError("manifest feature streams exceed max_transferred_bytes")
    if (
        sum(item["feature_count"] for item in completed_inputs)
        > result_budget["max_features"]
    ):
        raise ValueError("manifest completed inputs exceed max_features")
    if (
        sum(item["vertex_count"] for item in completed_inputs)
        > result_budget["max_vertices"]
    ):
        raise ValueError("manifest completed inputs exceed max_vertices")

    return canonical_tree(
        {
            "schema_version": MODEL_RUN_SCHEMA_VERSION,
            "model_run_id": model_run_id,
            "action_id": action_id,
            "backend": backend,
            "request_digest": request_digest,
            "completed_units": completed_units,
            "checkpoints": checkpoints,
            "failed_units": failed_units,
            "completed_inputs": completed_inputs,
            "complete": complete,
            **({"source_bindings": bindings} if captures else {}),
            **({"failure": failure} if failure is not None else {}),
        }
    )


def _normalize_publication_collections(value, request, inputs, plans, bindings):
    digest = _execution_identity_digester(request, inputs, plans, bindings)
    return {
        "completed_units": [
            _validate_unit_result(item, plans, digest)
            for item in _require_array(value["completed_units"], "completed_units")
        ],
        "checkpoints": [
            _validate_checkpoint(item, plans, digest)
            for item in _require_array(value["checkpoints"], "checkpoints")
        ],
        "failed_units": [
            _validate_unit_failure(item, plans)
            for item in _require_array(value["failed_units"], "failed_units")
        ],
        "completed_inputs": [
            _validate_input_result(item, inputs, plans, bindings)
            for item in _require_array(value["completed_inputs"], "completed_inputs")
        ],
    }


def _validate_unit_result(
    value: Any,
    plans: Mapping[int, Mapping[str, Any]],
    execution_digest: Callable[[int, int], str],
) -> dict[str, Any]:
    item = _require_exact_object(value, _UNIT_RESULT_FIELDS, "model run unit result")
    input_id, unit_id, plan = _require_plan_unit(item, plans)
    if plan["type"] != "core-halo":
        raise ValueError("completed unit state is legal only for core-halo plans")
    state = _require_nonempty_artifact(item["reusable_state"], "reusable_state")
    digest = require_sha256(
        item["execution_identity_digest"], "execution_identity_digest"
    )
    expected = execution_digest(input_id, unit_id)
    if digest != expected:
        raise ValueError(
            "unit result execution_identity_digest does not match its plan"
        )
    return canonical_tree(
        {
            "input_id": input_id,
            "unit_id": unit_id,
            "reusable_state": state,
            "execution_identity_digest": digest,
        }
    )


def _validate_checkpoint(
    value: Any,
    plans: Mapping[int, Mapping[str, Any]],
    execution_digest: Callable[[int, int], str],
) -> dict[str, Any]:
    item = _require_exact_object(value, _CHECKPOINT_FIELDS, "model run checkpoint")
    input_id, unit_id, plan = _require_plan_unit(item, plans)
    if plan["type"] != "global-decode" or unit_id != 0:
        raise ValueError("checkpoint is legal only for global-decode unit zero")
    sequence = require_positive_safe_integer(item["sequence"], "sequence")
    state = _require_nonempty_artifact(item["state"], "state")
    digest = require_sha256(
        item["execution_identity_digest"], "execution_identity_digest"
    )
    expected = execution_digest(input_id, unit_id)
    if digest != expected:
        raise ValueError("checkpoint execution_identity_digest does not match its plan")
    return canonical_tree(
        {
            "input_id": input_id,
            "unit_id": unit_id,
            "sequence": sequence,
            "state": state,
            "execution_identity_digest": digest,
        }
    )


def _validate_unit_failure(
    value: Any, plans: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    item = _require_exact_object(value, _UNIT_FAILURE_FIELDS, "model run unit failure")
    input_id, unit_id, _ = _require_plan_unit(item, plans)
    return canonical_tree(
        {
            "input_id": input_id,
            "unit_id": unit_id,
            "failure": validate_gateway_error(item["failure"]),
        }
    )


def _validate_input_result(
    value: Any,
    inputs: Mapping[int, Mapping[str, Any]],
    plans: Mapping[int, Mapping[str, Any]],
    source_bindings=(),
) -> dict[str, Any]:
    item = _require_exact_object(value, _INPUT_RESULT_FIELDS, "model input result")
    input_id = require_nonnegative_int32(item["input_id"], "input_id")
    if input_id not in inputs:
        raise ValueError("model input result input_id is absent from request")
    input_identity = validate_backend_input_identity(item["input_identity"])
    bindings = input_source_bindings(source_bindings, input_id)
    raster_input = inputs[input_id]["raster_input"]
    if "capture" in raster_input["source_view"] and len(bindings) != len(
        _plan_unit_ids(plans[input_id])
    ):
        raise ValueError(
            "complete input result lacks sources for every processing unit"
        )
    expected_identity = build_backend_input_identity(raster_input, bindings)
    if input_identity != expected_identity:
        raise ValueError("model input result input_identity does not match request")
    plan = validate_model_execution_plan(item["execution_plan"])
    if plan != plans[input_id]:
        raise ValueError("model input result execution_plan does not match request")
    artifact = _require_feature_stream_artifact(item["feature_stream_artifact"])
    feature_count = require_nonnegative_safe_integer(
        item["feature_count"], "feature_count"
    )
    vertex_count = require_nonnegative_safe_integer(
        item["vertex_count"], "vertex_count"
    )
    if (feature_count == 0) != (vertex_count == 0):
        raise ValueError("feature_count and vertex_count must be zero together")
    return canonical_tree(
        {
            "input_id": input_id,
            "input_identity": input_identity,
            "execution_plan": plan,
            "feature_stream_artifact": artifact,
            "feature_count": feature_count,
            "vertex_count": vertex_count,
        }
    )


def _execution_identity_digester(
    request: Mapping[str, Any],
    inputs: Mapping[int, Mapping[str, Any]],
    plans: Mapping[int, Mapping[str, Any]],
    source_bindings=(),
) -> Callable[[int, int], str]:
    """Reuse common canonical bytes within one already-validated operation.

    A locally normalized input appends its own ``source_binding`` and ``unit_id``
    after the common keys in JCS order. Target-normalized inputs instead include
    their complete ``source_bindings`` in the common object and append only the
    unit ID. Replacing the final brace therefore preserves the canonical preimage.
    Only hash states are retained, not the encoded mask/plan or source metadata.
    The caller owns the validated inputs and bindings for this operation only.
    """

    prefixes = {}

    def digest(input_id: int, unit_id: int) -> str:
        model_input = inputs[input_id]
        shared_target = uses_target_normalization(
            request["expected_backend"]["architecture"], plans[input_id]["type"]
        )
        if input_id not in prefixes:
            common = {
                "action_id": request["action_id"],
                "producer": request["producer"],
                "expected_backend": request["expected_backend"],
                "input": model_input,
                "execution_plan": plans[input_id],
            }
            if "expected_profile" in request:
                common["expected_profile"] = request["expected_profile"]
            if (
                shared_target
                and "capture" in model_input["raster_input"]["source_view"]
            ):
                bindings = input_source_bindings(source_bindings, input_id)
                if [item["unit_id"] for item in bindings] != list(
                    _plan_unit_ids(plans[input_id])
                ):
                    raise ValueError(
                        "target normalization requires every input source binding"
                    )
                common["source_bindings"] = bindings
            prefixes[input_id] = sha256(canonical_bytes(common)[:-1])
        result = prefixes[input_id].copy()
        if (
            not shared_target
            and "capture" in model_input["raster_input"]["source_view"]
        ):
            binding = unit_source_binding(source_bindings, input_id, unit_id)
            if binding is None:
                raise ValueError("processing unit has no accepted source binding")
            result.update(b',"source_binding":')
            result.update(canonical_bytes(binding))
        result.update(b',"unit_id":')
        result.update(canonical_bytes(unit_id))
        result.update(b"}")
        return result.hexdigest()

    return digest


def _require_plan_unit(
    value: Mapping[str, Any], plans: Mapping[int, Mapping[str, Any]]
) -> tuple[int, int, Mapping[str, Any]]:
    input_id = require_nonnegative_int32(value["input_id"], "input_id")
    unit_id = require_nonnegative_int32(value["unit_id"], "unit_id")
    if input_id not in plans:
        raise ValueError("publication input_id is absent from model run request")
    plan = plans[input_id]
    if unit_id not in _plan_unit_ids(plan):
        raise ValueError("publication unit_id is absent from generated plan")
    return input_id, unit_id, plan


def _plan_unit_ids(plan: Mapping[str, Any]) -> range:
    if plan["type"] == "core-halo":
        return range(len(plan["windows"]))
    return range(1)


def _require_feature_stream_artifact(value: Any) -> dict[str, Any]:
    artifact = _require_nonempty_artifact(value, "feature_stream_artifact")
    if artifact["media_type"] != FEATURE_STREAM_MEDIA_TYPE:
        raise ValueError("feature stream artifact has the wrong media_type")
    if artifact.get("content_encoding") != FEATURE_STREAM_CONTENT_ENCODING:
        raise ValueError("feature stream artifact must use gzip content encoding")
    return artifact


def _require_nonempty_artifact(value: Any, path: str) -> dict[str, Any]:
    artifact = validate_artifact_ref(value)
    if artifact["size_bytes"] == 0:
        raise ValueError(f"{path} must name nonempty immutable bytes")
    return artifact


def _require_canonical_unique_order(
    values: Sequence[Mapping[str, Any]], path: str
) -> None:
    keys = [(item["input_id"], item["unit_id"]) for item in values]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path} contains duplicate input/unit keys")
    if keys != sorted(keys):
        raise ValueError(f"{path} must use canonical input/unit order")


def _require_canonical_unique_input_order(
    values: Sequence[Mapping[str, Any]],
) -> None:
    keys = [item["input_id"] for item in values]
    if len(set(keys)) != len(keys):
        raise ValueError("completed_inputs contains duplicate input_id")
    if keys != sorted(keys):
        raise ValueError("completed_inputs must use canonical input order")


def _require_current_schema(value: Any) -> None:
    if require_int32(value, "schema_version") != MODEL_RUN_SCHEMA_VERSION:
        raise ValueError("unsupported model run publication schema_version")


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_array(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be an array")
    return value


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_object(
    value: Any,
    fields: set[str],
    path: str,
    *,
    optional_fields: set[str] | None = None,
) -> Mapping[str, Any]:
    result = _require_object(value, path)
    actual = set(result)
    allowed = fields | (optional_fields or set())
    if not fields <= actual or not actual <= allowed:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - allowed)}"
        )
    return result
