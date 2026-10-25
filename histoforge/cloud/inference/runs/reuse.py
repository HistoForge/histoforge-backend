"""Exact-compatible publication selection for explicit replacement runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
    ArtifactFileStore,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digests,
    validate_model_input_result,
)
from histoforge.cloud.inference.runs.source_bindings import validate_run_source_bindings


def select_reusable_model_run_state(
    previous: Mapping[str, Any],
    request: Mapping[str, Any],
    limits: Mapping[str, Any],
    files: ArtifactFileStore,
) -> dict[str, Any]:
    """Select only exact-compatible immutable state from the named manifest."""

    resume = request["resume_from"]
    previous_id = resume["previous_run_id"]
    if previous["request"]["model_run_id"] != previous_id:
        raise _conflict(previous_id, "replacement source record has the wrong run ID")
    manifest = previous.get("manifest")
    if previous["phase"] not in {"PARTIAL", "COMPLETE"} or manifest is None:
        raise _conflict(previous_id, "replacement source has no terminal manifest")
    if manifest["manifest_digest"] != resume["previous_manifest_digest"]:
        raise _conflict(previous_id, "replacement source manifest digest differs")

    # Accepted sources are immutable even when their units never completed.
    # Restore them before selecting reusable computation, not from that subset.
    inputs = {item["input_id"]: item for item in request["inputs"]}
    plans = build_expected_model_execution_plans(request, limits)
    try:
        bindings = validate_run_source_bindings(
            previous["source_bindings"], inputs, plans
        )
    except ValueError as error:
        raise _conflict(
            previous_id, "replacement cannot retain its accepted sources"
        ) from error

    complete_unit_keys = {
        (item["input_id"], unit_id)
        for item in previous["completed_inputs"]
        for unit_id in range(
            len(item["execution_plan"]["windows"])
            if item["execution_plan"]["type"] == "core-halo"
            else 1
        )
    }
    publication_keys = {
        (item["input_id"], item["unit_id"])
        for item in [*previous["completed_units"], *previous["checkpoints"]]
    }
    replacement_identities = model_run_execution_identity_digests(
        request, sorted(complete_unit_keys | publication_keys), limits, bindings
    )
    try:
        prior_identities = (
            model_run_execution_identity_digests(
                previous["request"],
                sorted(complete_unit_keys),
                previous["limits"],
                previous["source_bindings"],
            )
            if complete_unit_keys
            else {}
        )
    except ValueError:
        prior_identities = {}

    completed_inputs = []
    completed_input_decoded_bytes = []
    decoded_by_input = {
        item["input_id"]: item for item in previous["completed_input_decoded_bytes"]
    }
    completed_input_ids = set()
    for item in previous["completed_inputs"]:
        input_id = item["input_id"]
        plan = item["execution_plan"]
        unit_count = len(plan["windows"]) if plan["type"] == "core-halo" else 1
        if not all(
            prior_identities.get((input_id, unit_id)) is not None
            and prior_identities[(input_id, unit_id)]
            == replacement_identities.get((input_id, unit_id))
            for unit_id in range(unit_count)
        ):
            continue
        validate_model_input_result(item, request, limits, bindings)
        _require_artifact(item["feature_stream_artifact"], files, previous_id)
        completed_inputs.append(item)
        completed_input_decoded_bytes.append(decoded_by_input[input_id])
        completed_input_ids.add(input_id)

    completed_units = []
    for item in previous["completed_units"]:
        input_id = item["input_id"]
        if input_id in completed_input_ids or item[
            "execution_identity_digest"
        ] != replacement_identities.get((input_id, item["unit_id"])):
            continue
        _require_artifact(item["reusable_state"], files, previous_id)
        completed_units.append(item)

    checkpoints = []
    for item in previous["checkpoints"]:
        input_id = item["input_id"]
        if input_id in completed_input_ids or item[
            "execution_identity_digest"
        ] != replacement_identities.get((input_id, item["unit_id"])):
            continue
        _require_artifact(item["state"], files, previous_id)
        checkpoints.append(item)

    return canonical_tree(
        {
            "source_bindings": bindings,
            "completed_units": completed_units,
            "checkpoints": checkpoints,
            "failed_units": [],
            "completed_inputs": completed_inputs,
            "completed_input_decoded_bytes": completed_input_decoded_bytes,
        }
    )


def _require_artifact(
    reference: Mapping[str, Any], files: ArtifactFileStore, previous_id: str
) -> None:
    try:
        observed = files.load_artifact_ref(reference["artifact_id"])
    except (ArtifactBytesCorrupt, ArtifactBytesMissing, OSError) as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "replacement source contains missing or corrupt reusable bytes",
            resource_id=previous_id,
        ) from error
    if observed != reference:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "replacement source artifact descriptor differs",
            resource_id=previous_id,
        )


def _conflict(resource_id: str, message: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.IDEMPOTENCY_CONFLICT,
        message,
        resource_id=resource_id,
    )
