"""Store completed-core geometry and assemble one bounded final result."""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Any

from histoforge.cloud.architectures.registry import processing_result_limits
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt, ArtifactFileStore
from histoforge.cloud.inference.contract import (
    validate_client_result_budget,
    validate_inference_features,
)
from histoforge.cloud.inference.feature_stream import (
    DecodedFeatureStream,
    EncodedFeatureStream,
    collection_budget_for_stream,
    decode_feature_stream,
    encode_canonical_feature_records,
    encode_canonical_feature_stream,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
    encode_feature_record,
    feature_id_from_encoded_coordinates,
)
from histoforge.cloud.inference.runs.core_halo import InsufficientCoreHaloContext
from histoforge.cloud.inference.runs.publications import (
    FEATURE_STREAM_CONTENT_ENCODING,
    FEATURE_STREAM_MEDIA_TYPE,
    validate_model_run_publications,
)
from histoforge.cloud.rasterization.inputs import (
    geometry_vertex_count,
    require_geometry_inside_raster,
    validate_raster_descriptor,
)

_LOGGER = logging.getLogger(__name__)


def publish_core_halo_unit_state(
    encoded: EncodedFeatureStream,
    files: ArtifactFileStore,
) -> dict[str, Any]:
    """Publish complete bounded bytes before returning their immutable reference."""
    return files.publish_artifact_bytes(
        encoded.data,
        media_type=FEATURE_STREAM_MEDIA_TYPE,
        content_encoding=FEATURE_STREAM_CONTENT_ENCODING,
    )


def merge_core_halo_unit_states(
    unit_publications: Sequence[Any],
    *,
    request: Mapping[str, Any],
    input_id: int,
    plan: Mapping[str, Any],
    model_run_limits: Any,
    declared_classes: Collection[str],
    output_type: str,
    result_budget: Mapping[str, int],
    target_raster: Mapping[str, Any],
    files: ArtifactFileStore,
    source_bindings=(),
) -> EncodedFeatureStream:
    """Load every completed core within intermediate limits, then bound the result."""

    units = validate_model_run_publications(
        {
            "source_bindings": source_bindings,
            "completed_units": unit_publications,
            "checkpoints": [],
            "failed_units": [],
            "completed_inputs": [],
        },
        request,
        model_run_limits,
    )["completed_units"]
    expected_ids = list(range(len(plan["windows"])))
    if [item["unit_id"] for item in units] != expected_ids or any(
        item["input_id"] != input_id for item in units
    ):
        raise ValueError("completed CoreHalo units do not exactly cover the input plan")

    intermediate_limits = model_run_limits["intermediate_result"]
    processing_limits = processing_result_limits(
        request["expected_backend"]["architecture"]
    )
    if (
        sum(item["reusable_state"]["size_bytes"] for item in units)
        > (intermediate_limits["max_transferred_bytes"])
    ):
        raise ResultLimitExceeded(
            "completed-core artifacts exceed intermediate max_transferred_bytes"
        )

    semantic = output_type == "dense_classification"
    hovernet = request["expected_backend"]["architecture"] == "dinov3-adapter-hovernet"
    if not semantic and output_type not in {
        "instance_segmentation",
        "instance_segmentation_and_classification",
    }:
        raise ValueError("unsupported CoreHalo output type")
    raster = validate_raster_descriptor(target_raster)
    budget = validate_client_result_budget(result_budget)
    fragments: list[tuple[str, Mapping[str, Any]]] = []
    window_collections = []
    # Other instance families retain exact coordinate bytes and multiplicities,
    # releasing decoded geometry after each core.
    groups: dict[tuple[str, bytes], tuple[str, int, int]] = {}
    decoded_bytes = 0
    feature_count = 0
    vertex_count = 0
    for publication in units:
        reference = publication["reusable_state"]
        _require_unit_state_ref(reference)
        if reference["size_bytes"] > min(
            processing_limits["max_transferred_bytes"],
            model_run_limits["max_recovery_state_bytes_per_unit"],
        ):
            raise ResultLimitExceeded("completed-core artifact exceeds per-unit limit")
        observed = files.load_artifact_ref(reference["artifact_id"])
        if observed != reference:
            raise ArtifactBytesCorrupt(
                "CoreHalo unit descriptor differs from its publication"
            )
        data = files.read_artifact_block(
            reference["artifact_id"], 0, reference["size_bytes"]
        )
        # Restrict each decode to the remaining aggregate budget. At zero, one
        # byte permits an empty gzip stream but rejects any nonempty record.
        remaining_limits = {
            **intermediate_limits,
            "max_decoded_bytes": max(
                1, intermediate_limits["max_decoded_bytes"] - decoded_bytes
            ),
            "max_features": max(1, intermediate_limits["max_features"] - feature_count),
            "max_vertices": max(1, intermediate_limits["max_vertices"] - vertex_count),
        }
        remaining_limits = {
            field: min(value, processing_limits[field])
            for field, value in remaining_limits.items()
        }
        decoded = decode_feature_stream(data, declared_classes, remaining_limits)
        decoded_bytes += decoded.decoded_bytes
        feature_count += decoded.feature_count
        vertex_count += decoded.vertex_count
        _require_aggregate_counts(
            decoded_bytes,
            feature_count,
            vertex_count,
            intermediate_limits,
        )
        if hovernet:
            # The aggregate budgets above bound these local observations. Other
            # instance families retain their existing streaming coordinate merge.
            window_collections.append(decoded.collection)
        elif semantic:
            fragments.extend(
                (feature["properties"]["path_class"], feature["geometry"])
                for feature in decoded.collection["features"]
            )
        else:
            _collect_instance_coordinates(decoded, groups, raster)
        # Release bytes/coordinate buffers. HoVer alone retains its bounded
        # local collection for correspondence across the complete window cover.
        del decoded, data

    if hovernet:
        from histoforge.cloud.architectures.hovernet.reconciliation import (
            reconcile_window_predictions,
        )

        collection, decisions = reconcile_window_predictions(
            plan, window_collections, declared_classes
        )
        _LOGGER.info(
            "stage=hovernet_reconciliation candidates=%d selected=%s "
            "spatial_query_hits=%d",
            decisions["decoded_candidate_count"],
            decisions["output_count"],
            decisions["spatial_query_hits"],
        )
        if collection is None:
            _LOGGER.warning(
                "HoVer window observations are unresolved: %s",
                decisions["failures"][0]["reason"],
            )
            raise InsufficientCoreHaloContext(
                "Neighbouring processing areas produced conflicting object outlines. "
                "This region could not be completed. Increase the context in "
                "Inference settings and start a new analysis, "
                "or analyse a smaller region."
            )
        complete = validate_inference_features(
            collection, raster, collection_budget_for_stream(budget), declared_classes
        )
        return encode_canonical_feature_stream(complete, budget)

    # The aggregate intermediate checks above bound these geometries. Final
    # byte limits apply to the encoded stream, not a FeatureCollection wrapper
    # that is never transferred or stored in the result artifact.
    if semantic:
        from histoforge.cloud.inference.semantic_features import (
            assemble_semantic_fragments,
        )

        assembled = assemble_semantic_fragments(fragments)
        collection = build_feature_collection(assembled, declared_classes)
        complete = validate_inference_features(
            collection, raster, budget, declared_classes
        )
        return encode_canonical_feature_stream(complete, budget)

    _require_aggregate_counts(decoded_bytes, feature_count, vertex_count, budget)
    encoded = encode_canonical_feature_records(
        _instance_feature_records(groups), budget
    )
    if (
        encoded.decoded_bytes != decoded_bytes
        or encoded.feature_count != feature_count
        or encoded.vertex_count != vertex_count
    ):
        raise ValueError("merged CoreHalo feature counts changed")
    return encoded


def _collect_instance_coordinates(
    decoded: DecodedFeatureStream,
    groups: dict[tuple[str, bytes], tuple[str, int, int]],
    raster: Mapping[str, Any],
) -> None:
    """Keep one coordinate byte string per exact class/shape, preserving duplicates."""

    for feature, coordinates in zip(
        decoded.collection["features"], decoded.encoded_coordinates, strict=True
    ):
        geometry = feature["geometry"]
        require_geometry_inside_raster(geometry, raster, "inference feature geometry")
        key = (
            feature["properties"]["path_class"],
            coordinates,
        )
        if key in groups:
            geometry_type, vertices, count = groups[key]
            groups[key] = (geometry_type, vertices, count + 1)
        else:
            groups[key] = (geometry["type"], geometry_vertex_count(geometry), 1)


def _instance_feature_records(
    groups: Mapping[tuple[str, bytes], tuple[str, int, int]],
) -> Iterator[tuple[bytes, int]]:
    """Assign global occurrence IDs, then encode records in the existing ID order."""

    ordered = [
        (feature_id_from_encoded_coordinates(*key, occurrence), key)
        for key, (_geometry_type, _vertices, count) in groups.items()
        for occurrence in range(count)
    ]
    ordered.sort()
    for feature_id, (path_class, coordinates) in ordered:
        geometry_type, vertices, _count = groups[(path_class, coordinates)]
        yield (
            encode_feature_record(feature_id, path_class, geometry_type, coordinates),
            vertices,
        )


def _require_unit_state_ref(reference: Mapping[str, Any]) -> None:
    if (
        reference["media_type"] != FEATURE_STREAM_MEDIA_TYPE
        or reference.get("content_encoding") != FEATURE_STREAM_CONTENT_ENCODING
    ):
        raise ValueError("CoreHalo reusable state has the wrong media contract")


def _require_aggregate_counts(
    decoded_bytes: int,
    feature_count: int,
    vertex_count: int,
    budget: Mapping[str, int],
) -> None:
    if decoded_bytes > budget["max_decoded_bytes"]:
        raise ResultLimitExceeded("feature stream exceeds max_decoded_bytes")
    if feature_count > budget["max_features"]:
        raise ResultLimitExceeded("feature stream exceeds max_features")
    if vertex_count > budget["max_vertices"]:
        raise ResultLimitExceeded("feature stream exceeds max_vertices")
