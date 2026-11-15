"""Pure CPU conversion of decoded windows into bounded feature streams.

Instance labels retain the complete processing-image geometry until ownership
selection. HoVer retains all local observations for final reconciliation.
Semantic labels already cover the core and retain their local offset.
These functions never read source pixels, access mounted stores or publish data.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from os import getpid
from time import perf_counter
from typing import Any

import numpy as np

from histoforge.cloud.inference.feature_stream import (
    EncodedFeatureStream,
    collection_budget_for_stream,
    encode_canonical_feature_records,
    encode_canonical_feature_stream,
)
from histoforge.cloud.inference.mask_features import feature_collection_from_mask
from histoforge.cloud.inference.runs.core_halo import (
    select_canonical_core_halo_unit_features,
    translate_canonical_core_regions,
)
from histoforge.cloud.inference.semantic_features import (
    feature_collection_from_class_map,
)

_LOGGER = logging.getLogger(__name__)


def initialize_geometry_worker() -> int:
    """Confirm the spawned worker imported CPU conversion code before admission."""

    # Spawn does not inherit the parent's stage logging. Configure only this
    # CPU module, without importing the Modal app or changing host/root logging.
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    if not _LOGGER.handlers:
        _LOGGER.addHandler(logging.StreamHandler())
    return getpid()


def encode_instance_window(
    prediction: tuple[np.ndarray, Mapping[int, str] | str],
    window: Mapping[str, Any],
    declared_classes: Collection[str],
    target_raster: Mapping[str, int],
    processing_limits: Mapping[str, int],
) -> EncodedFeatureStream:
    """Convert complete local labels, retain owned instances, and encode them."""

    labels, classifications = prediction
    started = perf_counter()
    collection = feature_collection_from_mask(
        labels,
        classifications,
        declared_classes,
        collection_budget_for_stream(processing_limits),
    )
    _LOGGER.info(
        "stage=geometry_conversion elapsed_seconds=%.6f", perf_counter() - started
    )
    return encode_core_halo_collection(
        collection,
        window,
        declared_classes,
        "instance_segmentation",
        target_raster,
        processing_limits,
    )


def encode_semantic_window(
    prediction: tuple[np.ndarray, Mapping[int, str]],
    window: Mapping[str, Any],
    declared_classes: Collection[str],
    target_raster: Mapping[str, int],
    processing_limits: Mapping[str, int],
) -> EncodedFeatureStream:
    """Polygonize core class labels in the existing processing-image coordinates."""

    class_map, class_by_channel = prediction
    rect, core = window["input_rect"], window["core_rect"]
    started = perf_counter()
    collection = feature_collection_from_class_map(
        class_map,
        class_by_channel,
        declared_classes,
        collection_budget_for_stream(processing_limits),
        x_offset=core["x"] - rect["x"],
        y_offset=core["y"] - rect["y"],
    )
    _LOGGER.info(
        "stage=geometry_conversion elapsed_seconds=%.6f", perf_counter() - started
    )
    return encode_core_halo_collection(
        collection,
        window,
        declared_classes,
        "dense_classification",
        target_raster,
        processing_limits,
    )


def encode_hovernet_window(
    prediction: tuple[np.ndarray, Mapping[int, str] | str],
    window: Mapping[str, Any],
    declared_classes: Collection[str],
    target_raster: Mapping[str, int],
    processing_limits: Mapping[str, int],
) -> EncodedFeatureStream:
    """Keep bounded local HoVer observations for final cross-window selection.

    A locally incomplete core fails before state publication. Halo observations
    remain available to the finalizer; they cannot independently become output.
    """

    from histoforge.cloud.architectures.hovernet.reconciliation import (
        validate_window_observations,
    )

    labels, classifications = prediction
    started = perf_counter()
    collection = feature_collection_from_mask(
        labels,
        classifications,
        declared_classes,
        collection_budget_for_stream(processing_limits),
    )
    _LOGGER.info(
        "stage=geometry_conversion elapsed_seconds=%.6f", perf_counter() - started
    )
    validate_window_observations(window, collection, declared_classes, target_raster)
    return encode_canonical_feature_stream(collection, processing_limits)


def encode_core_halo_collection(
    collection: Mapping[str, Any],
    window: Mapping[str, Any],
    declared_classes: Collection[str],
    output_type: str,
    target_raster: Mapping[str, int],
    processing_limits: Mapping[str, int],
) -> EncodedFeatureStream:
    """Apply unchanged ownership/context rules and bound the canonical bytes."""

    started = perf_counter()
    select = (
        translate_canonical_core_regions
        if output_type == "dense_classification"
        else select_canonical_core_halo_unit_features
    )
    owned = select(
        window,
        collection,
        declared_classes,
        target_width=target_raster["width"],
        target_height=target_raster["height"],
        result_budget=collection_budget_for_stream(processing_limits),
    )
    _LOGGER.info(
        "stage=ownership_and_validation elapsed_seconds=%.6f", perf_counter() - started
    )
    started = perf_counter()
    encoded = encode_canonical_feature_records(
        owned.encoded_records(), processing_limits
    )
    _LOGGER.info(
        "stage=unit_state_encode elapsed_seconds=%.6f", perf_counter() - started
    )
    return encoded
