"""Class populations use complete TRAIN ROI support, not sampled field exposure."""

import copy

import numpy as np
import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.artifacts.volume import artifact_volume_gate
from histoforge.cloud.training.contract import training_task_from_capability
from histoforge.cloud.training.datasets import class_weights as training_class_weights
from histoforge.cloud.training.datasets import io as training_io
from histoforge.cloud.training.datasets.class_weights import (
    count_training_class_pixels,
    smoothed_class_weights,
)
from histoforge.cloud.training.datasets.rasters import validate_training_raster_profile
from histoforge.cloud.training.datasets.split import expand_training_split_plan
from tests.test_training_batches import _profile
from tests.test_training_contract import training_capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import (
    _dataset,
    _object_feature,
    _original_dataset,
    _rectangle,
)
from tests.test_training_field_reader import _origin, _retain, _write_png, _write_roi


def test_class_role_counts_preserve_object_holes_and_boundary_crossing(
    tmp_path,
):
    capability = _capability()
    roi = _rectangle(10, 10, 20, 20)
    crossing = _rectangle(8, 12, 13, 18)
    crossing["coordinates"].append(_rectangle(10, 14, 12, 16)["coordinates"][0])
    objects = [
        _object(1, crossing),
        _object(2, _rectangle(17, 11, 19, 13), "Other"),
        _object(3, _rectangle(18.1, 18.1, 18.4, 18.4)),
    ]
    dataset = _retained(tmp_path, roi, objects, capability)

    # 100 ROI pixels, 14 in the crossing instance and four in Other.
    # The object's four hole pixels remain supervised background.
    # The third captured object has positive vector area but no target pixel.
    assert _count(dataset, capability) == (82, 14, 4)


@pytest.mark.parametrize("context", [False, True])
def test_valid_negative_roi_counts_background_without_context_or_tensor_padding(
    tmp_path, context
):
    capability = _capability()
    dataset = _retained(tmp_path, _rectangle(10, 10, 15, 15), [], capability)
    profile = {**capability["default_field_profile"], "smallRoiContext": context}

    assert _count(dataset, capability, profile=profile) == (25, 0, 0)


@pytest.mark.parametrize("offset", [10.1, 100.1])
def test_accepted_context_owns_grid_phase_not_seal_grid_or_source_envelope(
    tmp_path, offset
):
    capability = _capability()
    roi = _rectangle(offset, offset, offset + 5, offset + 5)
    objects = [
        _object(1, _rectangle(offset + 1.1, offset + 1.1, offset + 1.7, offset + 1.7))
    ]
    dataset = _retained(tmp_path, roi, objects, capability, width=128, height=128)
    profile = capability["default_field_profile"]

    assert _count(dataset, capability, profile=profile) == (24, 1, 0)
    contextual = {**profile, "smallRoiContext": True}
    assert _count(dataset, capability, profile=contextual) == (25, 0, 0)


def test_real_image_half_open_edge_excludes_selected_boundary_centers(tmp_path):
    capability = _capability()
    dataset = _retained(
        tmp_path,
        _rectangle(0.5, 0.5, 64, 64),
        [_object(1, _rectangle(62.5, 0.5, 65, 64))],
        capability,
    )

    # Centers 1..63 are real; centers at x=64 or y=64 are not. The foreground
    # has one real column, even though its complete geometry crosses that edge.
    assert _count(dataset, capability) == (63 * 62, 63, 0)


@pytest.mark.parametrize("window_side", [1, 7, 11, 1024])
def test_fractional_grid_counts_are_invariant_to_integer_window_partition(
    tmp_path, monkeypatch, window_side
):
    capability = _capability()
    offset, scale = 100.1, 0.3
    roi = _rectangle(offset, offset, offset + 32 * scale, offset + 32 * scale)
    objects = [
        _object(
            1,
            _rectangle(
                offset + 3.5 * scale,
                offset + 5.5 * scale,
                offset + 18.5 * scale,
                offset + 21.5 * scale,
            ),
        )
    ]
    dataset = _retained(
        tmp_path, roi, objects, capability, width=128, height=128, model_mpp=0.15
    )
    complete = _count(dataset, capability)
    monkeypatch.setattr(training_class_weights, "_COUNT_WINDOW_SIDE_PX", window_side)

    assert _count(dataset, capability) == complete
    assert sum(complete) == 32 * 32
    assert complete[1] > 0


def test_bounded_count_windows_ignore_execution_tile_size_and_overlap(
    tmp_path, monkeypatch
):
    capability = _capability()
    dataset = _retained(
        tmp_path,
        _rectangle(0, 0, 2050, 3),
        [_object(1, _rectangle(1023, 0, 1026, 3))],
        capability,
        width=2050,
        height=3,
    )
    rasterize = training_class_weights.rasterize_model_geometry
    windows = []

    def bounded_rasterize(geometry, rectangle):
        assert rectangle["width"] <= 1024
        assert rectangle["height"] <= 1024
        windows.append(dict(rectangle))
        return rasterize(geometry, rectangle)

    monkeypatch.setattr(
        training_class_weights, "rasterize_model_geometry", bounded_rasterize
    )
    for tile_side, overlap in ((16, 0.0), (32, 0.25), (64, 0.75)):
        selected = copy.deepcopy(capability)
        selected["default_field_profile"].update(
            preferredTileSideModelPx=tile_side, tileOverlap=overlap
        )
        assert _count(dataset, selected) == (2050 * 3 - 9, 9, 0)
    assert {rectangle["x"] for rectangle in windows} == {0, 1024, 2048}


def test_crop_padding_is_context_only(tmp_path):
    capability = _capability()
    capability.update(input_kind="crop", crop_padding_model_px=8)
    capability["small_roi_context_editable"] = False
    dataset = _retained(
        tmp_path,
        _rectangle(10, 10, 14, 14),
        [_object(1, _rectangle(8, 8, 12, 12))],
        capability,
    )

    assert _count(dataset, capability) == (12, 4, 0)


def test_validation_is_not_read_and_repeated_source_objects_count_per_train_roi(
    tmp_path, monkeypatch
):
    capability = _capability()
    extracted = _original_dataset(tmp_path / "source", resolution_level=0)
    extracted.manifest["task"] = training_task_from_capability(
        capability, _model_output(capability), 0.5, 4096
    )
    shared = _object(1, _rectangle(12, 10, 22, 14))
    for index, (roi, objects) in enumerate(
        [
            (_rectangle(10, 10, 14, 14), [shared]),
            (_rectangle(20, 10, 24, 14), [shared]),
            (
                _rectangle(30, 10, 34, 14),
                [_object(2, _rectangle(30, 10, 34, 14), "Other")],
            ),
        ]
    ):
        if index:
            record = copy.deepcopy(extracted.manifest["rois"][0])
            extracted.manifest["rois"].append(record)
        extracted.manifest["rois"][index]["source_roi_uuid"] = (
            f"{index + 1:08d}-1111-4111-8111-111111111111"
        )
        directory = extracted.root / f"rois/{index}"
        directory.mkdir(exist_ok=True)
        for name, value in (
            (
                "roi.geojson",
                {
                    "type": "Feature",
                    "geometry": canonical_geometry(roi),
                    "properties": {},
                },
            ),
            ("objects.geojson", {"type": "FeatureCollection", "features": objects}),
        ):
            (directory / name).write_bytes(canonical_bytes(value))
            if index:
                extracted.manifest["files"].append(
                    {
                        "path": f"rois/{index}/{name}",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    }
                )
    dataset = _retain(tmp_path, extracted, _origin(extracted), capability=capability)
    split = expand_training_split_plan(
        {
            "type": "split",
            "unit": "roi",
            "assignment_source": "custom",
            "units": [
                {
                    "unit_index": index,
                    "partition": "validation" if index == 2 else "training",
                }
                for index in range(3)
            ],
        },
        dataset.manifest,
        training_api_limits(),
    )
    read_roi = training_io.read_training_roi_geometry

    def training_only(root, roi_index, maximum_bytes):
        assert roi_index in (0, 1), (
            "validation geometry must not be read for class weights"
        )
        return read_roi(root, roi_index, maximum_bytes)

    monkeypatch.setattr(training_io, "read_training_roi_geometry", training_only)
    # This original's source MPP makes each ROI a 2 x 2 target grid. Each
    # occurrence contributes two foreground pixels, regardless of shared UUID.
    assert _count(dataset, capability, indices=split.training_roi_indices) == (4, 4, 0)


def test_dense_same_class_overlap_counts_union_once(tmp_path):
    capability = _capability("dense_classification")
    dataset = _retained(
        tmp_path,
        _rectangle(10, 10, 14, 14),
        [
            _object(1, _rectangle(10, 10, 12, 12)),
            _object(2, _rectangle(11, 11, 13, 13)),
        ],
        capability,
    )

    assert _count(dataset, capability) == (9, 7, 0)


@pytest.mark.parametrize(
    ("output_type", "second_class"),
    [
        ("instance_segmentation_and_classification", "Leydig cell"),
        ("instance_segmentation_and_classification", "Other"),
        ("dense_classification", "Other"),
    ],
)
def test_new_context_grid_rejects_supervised_overlap_absent_at_seal(
    tmp_path, output_type, second_class
):
    capability = _capability(output_type)
    dataset = _retained(
        tmp_path,
        _rectangle(10.1, 10.1, 15.1, 15.1),
        [
            _object(1, _rectangle(10.2, 10.2, 12.3, 15)),
            _object(2, _rectangle(11.9, 10.2, 14, 15), second_class),
        ],
        capability,
    )
    assert sum(_count(dataset, capability)) == 25
    profile = {**capability["default_field_profile"], "smallRoiContext": True}
    with pytest.raises(ValueError, match="share a supervised real pixel"):
        _count(dataset, capability, profile=profile)


def test_context_grid_with_no_supervised_real_pixel_fails(tmp_path):
    capability = _capability()
    dataset = _retained(tmp_path, _rectangle(10.1, 10.1, 10.7, 10.7), [], capability)
    assert _count(dataset, capability) == (1, 0, 0)
    profile = {**capability["default_field_profile"], "smallRoiContext": True}
    with pytest.raises(ValueError, match="no supervised real pixels"):
        _count(dataset, capability, profile=profile)


def test_counting_reads_only_metadata_then_releases_volume_gate(tmp_path, monkeypatch):
    capability = _capability()
    dataset = _retained(
        tmp_path,
        _rectangle(10, 10, 14, 14),
        [_object(1, _rectangle(10, 10, 12, 12))],
        capability,
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "class populations must not read RGB or construct flow/HoVer targets"
        )

    monkeypatch.setattr("histoforge.cloud.training.datasets.io.SourceReader", forbidden)
    monkeypatch.setattr(
        "histoforge.cloud.training.trainers.instance.field_targets.build_training_field_targets",
        forbidden,
    )
    rasterize = training_class_weights.rasterize_model_geometry
    read_roi = training_io.read_training_roi_geometry

    def gated_read(*args):
        assert artifact_volume_gate.locked()
        return read_roi(*args)

    def ungated_rasterize(*args):
        assert artifact_volume_gate.acquire(blocking=False)
        artifact_volume_gate.release()
        return rasterize(*args)

    monkeypatch.setattr(training_io, "read_training_roi_geometry", gated_read)
    monkeypatch.setattr(
        training_class_weights, "rasterize_model_geometry", ungated_rasterize
    )
    assert _count(dataset, capability) == (12, 4, 0)


def test_smoothed_class_weights_floor_zero_counts_and_use_even_numpy_median():
    # NumPy's even-population median is (1 + 9) / 2 = 5, not the lower middle 1.
    expected = np.sqrt(np.array([5, 5, 5 / 9, 5 / 25])).astype(np.float32).tolist()
    actual = smoothed_class_weights((0, 1, 9, 25))

    assert actual == expected
    assert all(type(value) is float for value in actual)
    assert smoothed_class_weights((0, 0, 0)) == [1.0, 1.0, 1.0]
    assert smoothed_class_weights((1, 4, 16)) == [2.0, 1.0, 0.5]


def _capability(output_type="instance_segmentation_and_classification"):
    capability = training_capability()
    capability.update(
        output_type=output_type,
        default_field_profile=_profile(),
        allowed_field_side_model_px={"minimum": 16, "maximum": 64},
        small_roi_context_editable=True,
    )
    if output_type == "dense_classification":
        capability.pop("flow_loss_weighting_options")
        capability["default_training_settings"].pop("flow_loss_weighting")
    return capability


def _model_output(capability):
    return {
        "type": capability["output_type"],
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "emitted", "pathClass": "Leydig cell"},
            {"channel": 2, "role": "training_only", "pathClass": "Other"},
        ],
    }


def _object(index, geometry, path_class="Leydig cell"):
    item = _object_feature(f"{index:08d}-1111-4111-8111-111111111111", geometry)
    item["properties"]["path_class"] = path_class
    return item


def _retained(
    tmp_path, roi, objects, capability, *, width=64, height=64, model_mpp=0.5
):
    extracted = _dataset(
        tmp_path / "source",
        objects=objects,
        capability=capability,
        model_output=_model_output(capability),
    )
    _write_roi(extracted, roi)
    origin = _origin(extracted)
    origin["source"].update(width=width, height=height)
    origin["source_rect_at_level"].update(width=width, height=height)
    extracted.manifest["rois"][0]["source_envelope"]["pixel_rect"].update(
        width=width, height=height
    )
    extracted.manifest["task"]["model_microns_per_pixel"] = model_mpp
    _write_png(
        extracted.root / "artifacts/0/source.png",
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    return _retain(tmp_path, extracted, origin, capability=capability)


def _count(dataset, capability, *, profile=None, indices=(0,)):
    selected = validate_training_raster_profile(
        dataset.manifest["task"],
        capability,
        profile or capability["default_field_profile"],
        4096,
    )
    return count_training_class_pixels(
        dataset,
        indices,
        capability=capability,
        field_profile=selected,
        max_json_member_bytes=8 * 1024**2,
    )
