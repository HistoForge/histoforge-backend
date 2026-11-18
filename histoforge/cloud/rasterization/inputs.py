"""Canonical source-view and target-raster input contracts.

The extension constructs these values and the backend independently validates
them before source decoding.  This module owns only normalized wire identity
and cross-record invariants; it does not select sources, rasterize pixels, or
apply QuPath scope acceptance.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from histoforge.canonical_geometry import (
    canonical_geometry,
    canonical_geometry_digest,
)
from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_nonblank_string,
    require_nonnegative_int32,
    require_positive_finite_number,
    require_positive_int32,
    require_sha256,
)
from histoforge.cloud.inference.identity import validate_rasterizer_identity
from histoforge.cloud.sources.capture import (
    validate_captured_source_binding,
    validate_captured_source_view,
)
from histoforge.cloud.sources.identity import (
    validate_image_plane_index,
    validate_image_source_identity,
    validate_int_rect,
    validate_scale_translate_2d,
)
from histoforge.cloud.sources.publication import validate_source_artifact_ref
from histoforge.execution_scope import validate_region_key

_IMAGE_REVISION_FIELDS = {
    "image_id",
    "image_source_digest",
    "image_data_content_digest",
}
_ACTION_SOURCE_REVISION_FIELDS = {"image_data_revision", "pixel_source"}
_RASTER_DESCRIPTOR_FIELDS = {
    "width",
    "height",
    "plane",
    "target_mpp",
    "resampling",
    "pixel_format",
}
_RASTER_TRANSFORM_FIELDS = {"raster", "raster_to_slide"}
_SOURCE_VIEW_PREIMAGE_FIELDS = {
    "artifact",
    "series_index",
    "resolution_level",
    "plane",
    "pixel_rect",
    "view_pixels_to_slide",
}
_SOURCE_VIEW_FIELDS = _SOURCE_VIEW_PREIMAGE_FIELDS | {"view_digest"}
_RASTER_FRAME_PREIMAGE_FIELDS = {
    "source_revision",
    "region",
    "transform",
    "inclusion_mask_raster",
    "inclusion_mask_digest",
}
_RASTER_FRAME_BUILD_FIELDS = _RASTER_FRAME_PREIMAGE_FIELDS - {"inclusion_mask_digest"}
_RASTER_FRAME_FIELDS = _RASTER_FRAME_PREIMAGE_FIELDS | {"frame_digest"}
_MODEL_RASTER_INPUT_FIELDS = {"source_view", "frame", "rasterizer", "input_digest"}


def validate_image_revision_token(value: Any) -> dict[str, str]:
    """Validate one complete semantic image revision used for conflict detection."""

    revision = _require_exact_object(value, _IMAGE_REVISION_FIELDS, "image revision")
    return {
        "image_id": require_nonblank_string(revision["image_id"], "image_id"),
        "image_source_digest": require_sha256(
            revision["image_source_digest"], "image_source_digest"
        ),
        "image_data_content_digest": require_sha256(
            revision["image_data_content_digest"], "image_data_content_digest"
        ),
    }


def validate_action_source_revision(value: Any) -> dict[str, Any]:
    """Bind one semantic ImageData revision to the independently captured source."""

    source_revision = _require_exact_object(
        value, _ACTION_SOURCE_REVISION_FIELDS, "action source revision"
    )
    image_revision = validate_image_revision_token(
        source_revision["image_data_revision"]
    )
    pixel_source = validate_image_source_identity(source_revision["pixel_source"])
    expected_source_digest = canonical_digest(pixel_source)
    if image_revision["image_source_digest"] != expected_source_digest:
        raise ValueError(
            "image_data_revision.image_source_digest does not match pixel_source"
        )
    return canonical_tree(
        {"image_data_revision": image_revision, "pixel_source": pixel_source}
    )


def validate_raster_descriptor(value: Any) -> dict[str, Any]:
    """Validate one exact square-pixel target-grid descriptor."""

    raster = _require_exact_object(
        value, _RASTER_DESCRIPTOR_FIELDS, "raster descriptor"
    )
    if raster["resampling"] != "BILINEAR":
        raise ValueError("resampling must be BILINEAR")
    if raster["pixel_format"] != "SRGB_UINT8":
        raise ValueError("pixel_format must be SRGB_UINT8")
    return canonical_tree(
        {
            "width": require_positive_int32(raster["width"], "width"),
            "height": require_positive_int32(raster["height"], "height"),
            "plane": validate_image_plane_index(raster["plane"]),
            "target_mpp": require_positive_finite_number(
                raster["target_mpp"], "target_mpp"
            ),
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        }
    )


def validate_raster_transform(value: Any) -> dict[str, Any]:
    """Validate a target grid and its exact axis-aligned slide mapping."""

    transform = _require_exact_object(
        value, _RASTER_TRANSFORM_FIELDS, "raster transform"
    )
    return canonical_tree(
        {
            "raster": validate_raster_descriptor(transform["raster"]),
            "raster_to_slide": validate_scale_translate_2d(
                transform["raster_to_slide"]
            ),
        }
    )


def validate_source_view(value: Any) -> dict[str, Any]:
    """Validate and self-digest one exact view into a sealed source artifact."""

    view = _require_exact_object(value, _SOURCE_VIEW_FIELDS, "source view")
    plane = validate_image_plane_index(view["plane"])
    _require_supported_plane(plane)
    preimage = {
        "artifact": validate_source_artifact_ref(view["artifact"]),
        "series_index": require_nonnegative_int32(view["series_index"], "series_index"),
        "resolution_level": require_nonnegative_int32(
            view["resolution_level"], "resolution_level"
        ),
        "plane": plane,
        "pixel_rect": validate_int_rect(view["pixel_rect"]),
        "view_pixels_to_slide": validate_scale_translate_2d(
            view["view_pixels_to_slide"]
        ),
    }
    digest = require_sha256(view["view_digest"], "view_digest")
    if digest != canonical_digest(preimage):
        raise ValueError("view_digest does not match source view fields")
    return canonical_tree({**preimage, "view_digest": digest})


def validate_raster_frame(value: Any) -> dict[str, Any]:
    """Validate and self-digest one logical region's exact target raster frame."""

    frame = _require_exact_object(value, _RASTER_FRAME_FIELDS, "raster frame")
    source_revision = validate_action_source_revision(frame["source_revision"])
    transform = validate_raster_transform(frame["transform"])
    _require_target_mpp_scale(transform, source_revision["pixel_source"])
    mask = canonical_geometry(frame["inclusion_mask_raster"])
    mask_digest = require_sha256(
        frame["inclusion_mask_digest"], "inclusion_mask_digest"
    )
    if mask_digest != canonical_geometry_digest(mask):
        raise ValueError("inclusion_mask_digest does not match inclusion_mask_raster")
    require_geometry_inside_raster(mask, transform["raster"], "inclusion_mask_raster")
    _require_plane_inside_source(
        transform["raster"]["plane"], source_revision["pixel_source"]
    )
    preimage = {
        "source_revision": source_revision,
        "region": validate_region_key(frame["region"]),
        "transform": transform,
        "inclusion_mask_raster": mask,
        "inclusion_mask_digest": mask_digest,
    }
    digest = require_sha256(frame["frame_digest"], "frame_digest")
    if digest != canonical_digest(preimage):
        raise ValueError("frame_digest does not match raster frame fields")
    return canonical_tree({**preimage, "frame_digest": digest})


def validate_model_raster_input(value: Any) -> dict[str, Any]:
    """Validate the complete immutable source view and requested target grid."""

    model_input = _require_exact_object(
        value, _MODEL_RASTER_INPUT_FIELDS, "model raster input"
    )
    source_view = validate_raster_source_view(model_input["source_view"])
    frame = validate_raster_frame(model_input["frame"])
    rasterizer = validate_rasterizer_identity(model_input["rasterizer"])
    _require_source_frame(source_view, frame)
    digest = require_sha256(model_input["input_digest"], "input_digest")
    expected = _model_raster_input_digest(source_view, frame, rasterizer)
    if digest != expected:
        raise ValueError("input_digest does not match model raster input identity")
    return canonical_tree(
        {
            "source_view": source_view,
            "frame": frame,
            "rasterizer": rasterizer,
            "input_digest": digest,
        }
    )


def model_raster_input_digest(value: Any) -> str:
    """Return the canonical compact digest of a validated model-raster input."""

    model_input = validate_model_raster_input(value)
    return model_input["input_digest"]


def build_source_view(value: Any) -> dict[str, Any]:
    """Build a self-digested source view from its six identity fields."""

    preimage = _require_exact_object(value, _SOURCE_VIEW_PREIMAGE_FIELDS, "source view")
    plane = validate_image_plane_index(preimage["plane"])
    _require_supported_plane(plane)
    normalized = {
        "artifact": validate_source_artifact_ref(preimage["artifact"]),
        "series_index": require_nonnegative_int32(
            preimage["series_index"], "series_index"
        ),
        "resolution_level": require_nonnegative_int32(
            preimage["resolution_level"], "resolution_level"
        ),
        "plane": plane,
        "pixel_rect": validate_int_rect(preimage["pixel_rect"]),
        "view_pixels_to_slide": validate_scale_translate_2d(
            preimage["view_pixels_to_slide"]
        ),
    }
    return canonical_tree({**normalized, "view_digest": canonical_digest(normalized)})


def build_raster_frame(value: Any) -> dict[str, Any]:
    """Build a self-digested frame after validating all scientific input fields."""

    preimage = _require_exact_object(value, _RASTER_FRAME_BUILD_FIELDS, "raster frame")
    mask = canonical_geometry(preimage["inclusion_mask_raster"])
    normalized = {
        "source_revision": validate_action_source_revision(preimage["source_revision"]),
        "region": validate_region_key(preimage["region"]),
        "transform": validate_raster_transform(preimage["transform"]),
        "inclusion_mask_raster": mask,
        "inclusion_mask_digest": canonical_geometry_digest(mask),
    }
    _require_target_mpp_scale(
        normalized["transform"], normalized["source_revision"]["pixel_source"]
    )
    require_geometry_inside_raster(
        mask, normalized["transform"]["raster"], "inclusion_mask_raster"
    )
    _require_plane_inside_source(
        normalized["transform"]["raster"]["plane"],
        normalized["source_revision"]["pixel_source"],
    )
    return canonical_tree({**normalized, "frame_digest": canonical_digest(normalized)})


def build_model_raster_input(
    source_view: Any, frame: Any, rasterizer: Any
) -> dict[str, Any]:
    """Build the complete self-digested model-raster input from validated parts."""

    normalized_view = validate_raster_source_view(source_view)
    normalized_frame = validate_raster_frame(frame)
    normalized_rasterizer = validate_rasterizer_identity(rasterizer)
    _require_source_frame(normalized_view, normalized_frame)
    return canonical_tree(
        {
            "source_view": normalized_view,
            "frame": normalized_frame,
            "rasterizer": normalized_rasterizer,
            "input_digest": _model_raster_input_digest(
                normalized_view, normalized_frame, normalized_rasterizer
            ),
        }
    )


def validate_raster_source_view(value: Any) -> dict[str, Any]:
    """Accept a sealed-file view or immutable geometry awaiting bounded uploads."""

    item = _require_object(value, "source view")
    if "capture" in item:
        return validate_captured_source_view(item)
    return validate_source_view(item)


def model_input_source_artifact(raster_input, source_binding=None):
    """Resolve a file from a sealed view or an explicitly bound captured view."""

    view = raster_input["source_view"]
    if "capture" in view:
        return validate_captured_source_binding(source_binding, view["capture"])[
            "artifact"
        ]
    if source_binding is not None:
        raise ValueError(
            "sealed-file input must not carry an incremental source binding"
        )
    return view["artifact"]


def _require_source_frame(source_view, frame):
    source = source_view.get("capture", source_view)
    if source["plane"] != frame["transform"]["raster"]["plane"]:
        raise ValueError("source view plane does not match target raster plane")
    if (
        "capture" in source_view
        and source["source"] != frame["source_revision"]["pixel_source"]
    ):
        raise ValueError("source capture does not match the frame's pixel source")


def _model_raster_input_digest(
    source_view: Mapping[str, Any],
    frame: Mapping[str, Any],
    rasterizer: Mapping[str, Any],
) -> str:
    return canonical_digest(
        {
            "source_view_digest": source_view["view_digest"],
            "frame_digest": frame["frame_digest"],
            "rasterizer": rasterizer,
        }
    )


def require_geometry_inside_raster(
    geometry: Mapping[str, Any], raster: Mapping[str, Any], path: str
) -> None:
    """Bound polygons by pixel edges and points by the half-open pixel domain."""

    width = raster["width"]
    height = raster["height"]
    for x, y in _positions(geometry):
        if (
            x < 0.0
            or x > width
            or y < 0.0
            or y > height
            or (geometry["type"] == "Point" and (x == width or y == height))
        ):
            raise ValueError(f"{path} lies outside the target raster")


def geometry_vertex_count(geometry: Mapping[str, Any]) -> int:
    """Count one position per point and omit polygon closing duplicates."""

    if geometry["type"] == "Point":
        return 1
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    return sum(len(ring) - 1 for polygon in polygons for ring in polygon)


def _require_target_mpp_scale(
    transform: Mapping[str, Any], source: Mapping[str, Any]
) -> None:
    raster = transform["raster"]
    raster_to_slide = transform["raster_to_slide"]
    calibration = source["calibration"]
    expected_x = raster["target_mpp"] / calibration["mpp_x"]
    expected_y = raster["target_mpp"] / calibration["mpp_y"]
    if (
        raster_to_slide["scale_x"] != expected_x
        or raster_to_slide["scale_y"] != expected_y
    ):
        raise ValueError(
            "raster_to_slide scale does not match target_mpp and source calibration"
        )


def _positions(geometry: Mapping[str, Any]) -> Iterator[Sequence[float]]:
    coordinates = geometry["coordinates"]
    if geometry["type"] == "Point":
        yield coordinates
        return
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    for polygon in polygons:
        for ring in polygon:
            yield from ring


def _require_plane_inside_source(
    plane: Mapping[str, int], source: Mapping[str, Any]
) -> None:
    _require_supported_plane(plane)
    if plane["z"] >= source["z_slices"] or plane["t"] >= source["timepoints"]:
        raise ValueError("target raster plane lies outside the image source")


def _require_supported_plane(plane: Mapping[str, int]) -> None:
    if plane != {"z": 0, "t": 0}:
        raise ValueError("model raster inputs support only plane z=0, t=0")


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    result = _require_object(value, path)
    _require_fields(result, fields, path)
    return result


def _require_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
