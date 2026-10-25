"""Private persisted durable-run records over one Modal Dict-compatible map.

The public request, publication, manifest, and status codecs remain in their
shared contract modules.  This owner stores one bounded mutable control record
per run and validates it whenever it crosses the persistence boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_canonical_uuid,
    require_positive_int32,
    require_rfc3339_millis,
    sha256_bytes,
)
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    validate_gateway_error,
)
from histoforge.cloud.artifacts.refs import validate_artifact_ref
from histoforge.cloud.execution_limits import validate_model_run_limits

MODEL_RUN_RECORD_SCHEMA_VERSION = 2
_KEY_PREFIX = "model-run:v1:"
_RECORD_FIELDS = {
    "schema_version",
    "request",
    "request_digest",
    "request_artifact",
    "limits",
    "terminal_retention_seconds",
    "accepted_at",
    "execution_deadline",
    "phase",
    "phase_deadline",
    "source_bindings",
    "completed_units",
    "checkpoints",
    "failed_units",
    "completed_inputs",
    "completed_input_decoded_bytes",
    "released_artifact_ids",
    "terminal_manifest",
    "manifest",
    "failure",
}
_ACTIVE_PHASES = {"QUEUED", "RUNNING"}
_TERMINAL_PHASES = {"PARTIAL", "COMPLETE", "FAILED", "CANCELLED", "TIMED_OUT"}


class ModelRunRecords:
    """Validated whole-record reads and atomic replacement writes."""

    def __init__(self, values: Any) -> None:
        self._values = values

    def get(self, model_run_id: str) -> dict[str, Any] | None:
        """Return one validated record, or ``None`` when absent."""

        canonical_id = require_canonical_uuid(model_run_id, "model_run_id")
        value = self._values.get(_key(canonical_id))
        return None if value is None else validate_model_run_record(value, canonical_id)

    def all(self) -> list[dict[str, Any]]:
        """Return every valid run record in deterministic ID order."""

        records = []
        for key, value in list(self._values.items()):
            if not isinstance(key, str) or not key.startswith(_KEY_PREFIX):
                continue
            model_run_id = key.removeprefix(_KEY_PREFIX)
            try:
                canonical_id = require_canonical_uuid(model_run_id, "model_run_id")
                records.append(validate_model_run_record(value, canonical_id))
            except (TypeError, ValueError) as error:
                raise _invalid_store(str(model_run_id)) from error
        return sorted(records, key=lambda item: item["request"]["model_run_id"])

    def active_ids(self) -> list[str]:
        """List cancellation targets without loading the cloud geometry stack.

        The local deployment helper needs only the persisted identity and phase.
        Validate that projection here, beside its schema/key owner; the existing
        cancellation API still validates the complete record before changing it.
        """

        active = []
        for key, value in list(self._values.items()):
            if not isinstance(key, str) or not key.startswith(_KEY_PREFIX):
                continue
            try:
                run_id = require_canonical_uuid(
                    key.removeprefix(_KEY_PREFIX), "model_run_id"
                )
                item = _exact_object(value, _RECORD_FIELDS, "model run record")
                if item["schema_version"] != MODEL_RUN_RECORD_SCHEMA_VERSION:
                    raise ValueError("unsupported private model-run record schema")
                if item["request"]["model_run_id"] != run_id:
                    raise ValueError("model-run record key does not match its request")
                phase = item["phase"]
                if phase not in _ACTIVE_PHASES | _TERMINAL_PHASES:
                    raise ValueError("unsupported private model-run phase")
                if phase in _ACTIVE_PHASES:
                    active.append(run_id)
            except (KeyError, TypeError, ValueError) as error:
                raise _invalid_store(key.removeprefix(_KEY_PREFIX)) from error
        return sorted(active)

    def reserve(self, record: Any) -> tuple[dict[str, Any], bool]:
        """Publish one absent run record and return ``(winner, created)``."""

        normalized = validate_model_run_record(record)
        model_run_id = normalized["request"]["model_run_id"]
        key = _key(model_run_id)
        created = _put_if_absent(self._values, key, normalized)
        if created:
            return normalized, True
        winner = self.get(model_run_id)
        if winner is None:
            raise _invalid_store(model_run_id)
        return winner, False

    def replace(self, record: Any) -> dict[str, Any]:
        """Atomically replace one existing record inside the lifecycle lane."""

        normalized = validate_model_run_record(record)
        model_run_id = normalized["request"]["model_run_id"]
        key = _key(model_run_id)
        if self._values.get(key) is None:
            raise _invalid_store(model_run_id)
        _put(self._values, key, normalized)
        return normalized

    def remove(self, model_run_id: str) -> None:
        """Remove one full control record after retention expires."""

        _pop(self._values, _key(model_run_id))


def validate_model_run_record(
    value: Any, expected_model_run_id: str | None = None
) -> dict[str, Any]:
    """Validate one private control record and all embedded public contracts."""

    from histoforge.cloud.inference.runs.publications import (
        validate_model_result_manifest,
        validate_model_run_publications,
    )
    from histoforge.cloud.inference.runs.request import (
        model_run_request_digest,
        validate_model_run_resource_limits,
    )

    try:
        item = _exact_object(value, _RECORD_FIELDS, "model run record")
        if item["schema_version"] != MODEL_RUN_RECORD_SCHEMA_VERSION:
            raise ValueError("unsupported private model-run record schema")
        limits = validate_model_run_limits(item["limits"])
        request = validate_model_run_resource_limits(item["request"], limits)
        model_run_id = request["model_run_id"]
        if expected_model_run_id is not None and model_run_id != require_canonical_uuid(
            expected_model_run_id, "model_run_id"
        ):
            raise ValueError("model-run record key does not match its request")
        request_digest = model_run_request_digest(request)
        if item["request_digest"] != request_digest:
            raise ValueError("model-run record request digest does not match request")
        request_artifact = _canonical_json_artifact(
            item["request_artifact"], request, "request_artifact"
        )
        retention = require_positive_int32(
            item["terminal_retention_seconds"], "terminal_retention_seconds"
        )
        accepted_at = require_rfc3339_millis(item["accepted_at"], "accepted_at")
        execution_deadline = require_rfc3339_millis(
            item["execution_deadline"], "execution_deadline"
        )
        phase_deadline = require_rfc3339_millis(
            item["phase_deadline"], "phase_deadline"
        )
        if accepted_at >= execution_deadline:
            raise ValueError("execution deadline must follow acceptance")

        phase = item["phase"]
        if phase not in _ACTIVE_PHASES | _TERMINAL_PHASES:
            raise ValueError("unsupported private model-run phase")
        if phase in _ACTIVE_PHASES and phase_deadline != execution_deadline:
            raise ValueError(
                "active model-run phase deadline must be execution deadline"
            )
        if phase in _TERMINAL_PHASES and phase_deadline <= accepted_at:
            raise ValueError("terminal retention deadline must follow acceptance")

        collections = {
            name: _array(item[name], name)
            for name in (
                "completed_units",
                "checkpoints",
                "failed_units",
                "completed_inputs",
            )
        }
        publications = validate_model_run_publications(
            {"source_bindings": item["source_bindings"], **collections}, request, limits
        )
        source_bindings = publications["source_bindings"]
        completed_units = publications["completed_units"]
        checkpoints = publications["checkpoints"]
        failed_units = publications["failed_units"]
        completed_inputs = publications["completed_inputs"]
        completed_input_decoded_bytes = _decoded_input_bytes(
            item["completed_input_decoded_bytes"], completed_inputs
        )
        released_artifact_ids = _artifact_id_list(
            item["released_artifact_ids"], "released_artifact_ids"
        )
        _require_order(completed_units, "completed_units", include_unit=True)
        _require_order(checkpoints, "checkpoints", include_unit=True)
        _require_order(failed_units, "failed_units", include_unit=True)
        _require_order(completed_inputs, "completed_inputs", include_unit=False)
        _require_open_state_consistency(
            request,
            limits,
            completed_units,
            checkpoints,
            failed_units,
            completed_inputs,
            completed_input_decoded_bytes,
        )
        if phase == "QUEUED":
            has_seeded_state = bool(completed_units or checkpoints or completed_inputs)
            if failed_units or (has_seeded_state and "resume_from" not in request):
                raise ValueError(
                    "QUEUED publications require exact replacement-run state"
                )

        terminal_manifest = item["terminal_manifest"]
        manifest = item["manifest"]
        failure = item["failure"]
        if phase in {"PARTIAL", "COMPLETE"}:
            if terminal_manifest is None or manifest is None:
                raise ValueError("manifest terminal phase has invalid payloads")
            normalized_manifest = validate_model_result_manifest(
                manifest, request, limits
            )
            expected_complete = phase == "COMPLETE"
            if normalized_manifest["complete"] != expected_complete:
                raise ValueError("terminal phase disagrees with manifest completeness")
            if failure is not None:
                failure = validate_gateway_error(failure)
            if failure != normalized_manifest.get("failure"):
                raise ValueError("record failure differs from manifest")
            terminal_ref = _canonical_json_artifact(
                terminal_manifest, normalized_manifest, "terminal_manifest"
            )
            if completed_units != normalized_manifest["completed_units"]:
                raise ValueError("record completed units differ from manifest")
            if checkpoints != normalized_manifest["checkpoints"]:
                raise ValueError("record checkpoints differ from manifest")
            if failed_units != normalized_manifest["failed_units"]:
                raise ValueError("record failed units differ from manifest")
            if completed_inputs != normalized_manifest["completed_inputs"]:
                raise ValueError("record completed inputs differ from manifest")
            if source_bindings != normalized_manifest.get("source_bindings", []):
                raise ValueError("record source bindings differ from manifest")
        elif phase == "FAILED":
            if terminal_manifest is not None or manifest is not None or failure is None:
                raise ValueError("FAILED record requires only failure")
            failure = validate_gateway_error(failure)
            terminal_ref = None
            normalized_manifest = None
            if completed_units or checkpoints or completed_inputs:
                raise ValueError("FAILED record cannot retain reusable state")
        else:
            if (
                terminal_manifest is not None
                or manifest is not None
                or failure is not None
            ):
                raise ValueError(
                    "non-result model-run phase cannot carry terminal payloads"
                )
            terminal_ref = None
            normalized_manifest = None
            failure = None
            if phase == "TIMED_OUT" and (
                completed_units or checkpoints or failed_units or completed_inputs
            ):
                raise ValueError("TIMED_OUT record cannot retain computation state")

        return canonical_tree(
            {
                "schema_version": MODEL_RUN_RECORD_SCHEMA_VERSION,
                "request": request,
                "request_digest": request_digest,
                "request_artifact": request_artifact,
                "limits": limits,
                "terminal_retention_seconds": retention,
                "accepted_at": accepted_at,
                "execution_deadline": execution_deadline,
                "phase": phase,
                "phase_deadline": phase_deadline,
                "source_bindings": source_bindings,
                "completed_units": completed_units,
                "checkpoints": checkpoints,
                "failed_units": failed_units,
                "completed_inputs": completed_inputs,
                "completed_input_decoded_bytes": completed_input_decoded_bytes,
                "released_artifact_ids": released_artifact_ids,
                "terminal_manifest": terminal_ref,
                "manifest": normalized_manifest,
                "failure": failure,
            }
        )
    except GatewayFailure:
        raise
    except (KeyError, TypeError, ValueError) as error:
        resource = expected_model_run_id or "unknown-model-run"
        raise _invalid_store(resource) from error


def model_run_record_artifact_ids(record: Mapping[str, Any]) -> set[str]:
    """Return every live or deferred-cleanup artifact owned by one record."""

    normalized = validate_model_run_record(record)
    result = {normalized["request_artifact"]["artifact_id"]}
    if normalized["terminal_manifest"] is not None:
        result.add(normalized["terminal_manifest"]["artifact_id"])
    for item in normalized["completed_units"]:
        result.add(item["reusable_state"]["artifact_id"])
    for item in normalized["checkpoints"]:
        result.add(item["state"]["artifact_id"])
    for item in normalized["completed_inputs"]:
        result.add(item["feature_stream_artifact"]["artifact_id"])
    result.update(normalized["released_artifact_ids"])
    return result


def _require_open_state_consistency(
    request: Mapping[str, Any],
    limits: Mapping[str, Any],
    completed_units: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    failed_units: list[dict[str, Any]],
    completed_inputs: list[dict[str, Any]],
    completed_input_decoded_bytes: list[dict[str, int]],
) -> None:
    completed_input_ids = {item["input_id"] for item in completed_inputs}
    for values in (completed_units, checkpoints, failed_units):
        if any(item["input_id"] in completed_input_ids for item in values):
            raise ValueError("completed input cannot retain lower-level state")
    unit_keys = {(item["input_id"], item["unit_id"]) for item in completed_units}
    failure_keys = {(item["input_id"], item["unit_id"]) for item in failed_units}
    if unit_keys & failure_keys:
        raise ValueError("one unit cannot be both completed and failed")
    checkpoint_keys = [(item["input_id"], item["unit_id"]) for item in checkpoints]
    if len(checkpoint_keys) != len(set(checkpoint_keys)):
        raise ValueError("only the highest checkpoint may be retained per unit")
    require_model_run_publication_budgets(
        request,
        limits,
        completed_units,
        checkpoints,
        completed_inputs,
        completed_input_decoded_bytes,
    )


def require_model_run_publication_budgets(
    request: Mapping[str, Any],
    limits: Mapping[str, Any],
    completed_units: list[Mapping[str, Any]],
    checkpoints: list[Mapping[str, Any]],
    completed_inputs: list[Mapping[str, Any]],
    completed_input_decoded_bytes: list[Mapping[str, int]],
) -> None:
    """Enforce aggregate recovery and result budgets for one persisted cut.

    The private-record validator and the live publication path share this one
    arithmetic owner. The live path translates a violation into the public
    ``RESULT_LIMIT_EXCEEDED`` outcome before persistence validation can
    correctly classify an impossible stored cut as ``INTERNAL_ERROR``.
    """

    recovery_sizes = [item["reusable_state"]["size_bytes"] for item in completed_units]
    recovery_sizes += [item["state"]["size_bytes"] for item in checkpoints]
    if any(
        size > limits["max_recovery_state_bytes_per_unit"] for size in recovery_sizes
    ):
        raise ValueError("recovery state exceeds per-unit limit")
    if sum(recovery_sizes) > limits["max_recovery_state_bytes_per_run"]:
        raise ValueError("recovery state exceeds per-run limit")

    result_budget = request["result_budget"]
    if (
        sum(item["feature_stream_artifact"]["size_bytes"] for item in completed_inputs)
        > result_budget["max_transferred_bytes"]
    ):
        raise ValueError("completed input streams exceed transferred-byte budget")
    if (
        sum(item["feature_count"] for item in completed_inputs)
        > result_budget["max_features"]
    ):
        raise ValueError("completed inputs exceed feature budget")
    if (
        sum(item["vertex_count"] for item in completed_inputs)
        > result_budget["max_vertices"]
    ):
        raise ValueError("completed inputs exceed vertex budget")
    if (
        sum(item["decoded_bytes"] for item in completed_input_decoded_bytes)
        > result_budget["max_decoded_bytes"]
    ):
        raise ValueError("completed input streams exceed decoded-byte budget")


def _canonical_json_artifact(value: Any, body: Any, path: str) -> dict[str, Any]:
    artifact = validate_artifact_ref(value)
    encoded = canonical_bytes(body)
    if artifact["media_type"] != "application/json" or "content_encoding" in artifact:
        raise ValueError(f"{path} must be unencoded application/json")
    if artifact["size_bytes"] != len(encoded) or artifact["sha256"] != sha256_bytes(
        encoded
    ):
        raise ValueError(f"{path} does not bind its canonical JSON bytes")
    return artifact


def _require_order(
    values: list[Mapping[str, Any]], path: str, *, include_unit: bool
) -> None:
    keys = [
        (item["input_id"], item["unit_id"]) if include_unit else item["input_id"]
        for item in values
    ]
    if len(keys) != len(set(keys)) or keys != sorted(keys):
        raise ValueError(f"{path} must be unique and canonically ordered")


def _decoded_input_bytes(
    value: Any, completed_inputs: list[Mapping[str, Any]]
) -> list[dict[str, int]]:
    values = _array(value, "completed_input_decoded_bytes")
    normalized = []
    for item in values:
        exact = _exact_object(
            item, {"input_id", "decoded_bytes"}, "completed input decoded bytes"
        )
        input_id = exact["input_id"]
        decoded_bytes = exact["decoded_bytes"]
        if (
            isinstance(input_id, bool)
            or not isinstance(input_id, int)
            or input_id < 0
            or input_id > 2**31 - 1
        ):
            raise ValueError("decoded input input_id must be nonnegative int32")
        if (
            isinstance(decoded_bytes, bool)
            or not isinstance(decoded_bytes, int)
            or decoded_bytes < 0
            or decoded_bytes > 2**53 - 1
        ):
            raise ValueError("decoded_bytes must be a nonnegative safe integer")
        normalized.append({"input_id": input_id, "decoded_bytes": decoded_bytes})
    expected_ids = [item["input_id"] for item in completed_inputs]
    if [item["input_id"] for item in normalized] != expected_ids:
        raise ValueError(
            "decoded input byte counts must exactly cover completed inputs"
        )
    return normalized


def _artifact_id_list(value: Any, path: str) -> list[str]:
    values = [
        require_canonical_uuid(item, f"{path} artifact_id")
        for item in _array(value, path)
    ]
    if values != sorted(set(values)):
        raise ValueError(f"{path} must be unique and canonically ordered")
    return values


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value


def _key(model_run_id: str) -> str:
    return _KEY_PREFIX + require_canonical_uuid(model_run_id, "model_run_id")


def _put_if_absent(store: Any, key: str, value: Any) -> bool:
    put = getattr(store, "put", None)
    if callable(put):
        return bool(put(key, value, skip_if_exists=True))
    if key in store:
        return False
    store[key] = value
    return True


def _put(store: Any, key: str, value: Any) -> None:
    put = getattr(store, "put", None)
    if callable(put):
        put(key, value)
    else:
        store[key] = value


def _pop(store: Any, key: str) -> Any:
    try:
        return store.pop(key, None)
    except TypeError:
        try:
            value = store[key]
        except KeyError:
            return None
        del store[key]
        return value


def _invalid_store(model_run_id: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.INTERNAL_ERROR,
        "the durable model-run store contains an invalid record",
        resource_id=model_run_id,
    )
