"""Retained capture to held-out inference metrics on one exact raster grid."""

import copy
import weakref

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.artifacts import volume as artifact_volume
from histoforge.cloud.sources.reader import SourceReader
from histoforge.cloud.training.datasets import (
    validation as training_validation,
)
from histoforge.cloud.training.trainers.instance.metrics import InstanceMetricResult
from histoforge.cloud.training.trainers.instance.validation import (
    evaluate_bounded_training_validation,
)
from tests.test_training_batches import _profile
from tests.test_training_contract import training_capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _dataset, _object_feature, _rectangle
from tests.test_training_field_reader import (
    _origin,
    _pattern,
    _retain,
    _source_limits,
    _write_png,
    _write_roi,
)
from tests.test_training_rasters import _capability


@pytest.mark.parametrize("hole_prediction, expected", [(False, 1), (True, 2 / 3)])
def test_boundary_objects_context_and_object_holes_use_the_same_metric_grid(
    tmp_path, hole_prediction, expected
):
    """A complete crossing object is kept, but its outside pixels are not scored."""
    roi = _rectangle(10, 10, 30, 30)
    holed_object = _rectangle(18, 18, 26, 26)
    holed_object["coordinates"].append(_rectangle(20, 20, 24, 24)["coordinates"][0])
    dataset, split, capability = _capture(
        tmp_path,
        roi=roi,
        objects=[_object(1, _rectangle(8, 12, 14, 18)), _object(2, holed_object)],
    )
    mask = np.zeros((64, 64), np.int32)
    mask[12:18, 10:14] = 7  # The visible part of the crossing source object.
    mask[18:26, 18:26] = 8
    # An object hole is supervised background; a prediction there is a false positive.
    mask[20:24, 20:24] = 10 if hole_prediction else 0
    mask[:4, :4] = 9  # Real context is input, not supervised background.
    original = mask.copy()

    def predict(image):
        np.testing.assert_array_equal(image.pixels, _pattern(64, 64))
        assert image.real_pixel_mask.all()
        return {"instance_mask": mask}

    result = _evaluate(dataset, split, capability, predict)
    assert result.ap50 == pytest.approx(expected)
    assert result.msa == pytest.approx(expected)
    assert result.object_macro_f1 is None
    np.testing.assert_array_equal(mask, original)


def test_masking_preserves_one_instance_across_disconnected_object_fragments(tmp_path):
    roi = _rectangle(10, 10, 22, 14)
    # The complete U-shaped object is connected above the rectangular ROI.
    # Masking its connector leaves two fragments with one source-object identity.
    connected_object = {
        "type": "Polygon",
        "coordinates": [
            [
                [10, 8],
                [22, 8],
                [22, 14],
                [18, 14],
                [18, 10],
                [14, 10],
                [14, 14],
                [10, 14],
                [10, 8],
            ]
        ],
    }
    dataset, split, capability = _capture(
        tmp_path, roi=roi, objects=[_object(1, connected_object)]
    )
    mask = np.zeros((64, 64), np.int32)
    mask[10:14, 10:14] = 7
    mask[10:14, 18:22] = 42
    result = _evaluate(dataset, split, capability, lambda _: {"instance_mask": mask})
    assert result.ap50 == 0.5  # One true instance versus two predictions.
    assert result.msa == pytest.approx(0.05)


@pytest.mark.parametrize("channel, expected", [(0, 0.0), (1, 1.0), (2, 0.0)])
def test_class_votes_are_projected_without_revoting_after_roi_masking(
    tmp_path, channel, expected
):
    other = _object(2, _rectangle(22, 22, 26, 26))
    other["properties"]["path_class"] = "Other"
    dataset, split, capability = _capture(
        tmp_path,
        classified=True,
        objects=[_object(1, _rectangle(12, 12, 20, 20)), other],
    )
    mask = np.zeros((64, 64), np.int32)
    mask[12:20, 12:20] = 7
    mask[22:26, 22:26] = 42
    output = {"instance_mask": mask, "votes_per_instance": {7: channel, 42: 2}}
    assert _evaluate(
        dataset, split, capability, lambda _: output
    ) == InstanceMetricResult(expected, expected, expected, {1: expected})


def test_fractional_anisotropic_grid_is_not_recomputed_per_object(tmp_path):
    profile = training_capability()["default_field_profile"]
    profile["small_roi_context"] = False
    dataset, split, capability = _capture(
        tmp_path,
        roi=_rectangle(10.1, 10.4, 30.2, 30.3),
        objects=[_object(1, _rectangle(12, 12, 20, 20))],
        calibration={"mpp_x": 0.4, "mpp_y": 0.8},
        profile=profile,
    )
    y, x = np.mgrid[:32, :17]
    # Independent pixel-center oracle in the frozen model grid.
    inside = (
        (x + 0.5 >= (12 - 10.1) / 1.25)
        & (x + 0.5 < (20 - 10.1) / 1.25)
        & (y + 0.5 >= (12 - 10.4) / 0.625)
        & (y + 0.5 < (20 - 10.4) / 0.625)
    )
    mask = inside.astype(np.int32) * 73
    assert _evaluate(
        dataset, split, capability, lambda _: {"instance_mask": mask}
    ) == InstanceMetricResult(1, 1, None, {})


def test_physical_image_edge_is_not_supervised_background(tmp_path):
    dataset, split, capability = _capture(
        tmp_path,
        roi=_rectangle(0, 0, 64, 64),
        objects=[],
        calibration={"mpp_x": 0.4, "mpp_y": 0.4},
    )

    def predict(image):
        # Ceil preserves target MPP; the last center lies at 64.375, outside image.
        real = np.ones((52, 52), bool)
        real[-1] = False
        real[:, -1] = False
        np.testing.assert_array_equal(image.real_pixel_mask, real)
        return {"instance_mask": (~real).astype(np.int32)}

    assert _evaluate(dataset, split, capability, predict) == InstanceMetricResult(
        None, None, None, {}
    )


@pytest.mark.parametrize("fail_at", [None, "read", "predict"])
def test_sources_close_and_release_volume_gate_before_prediction(
    tmp_path, monkeypatch, fail_at
):
    dataset, split, capability = _capture(tmp_path, validation_rois=2)
    closed, pixels = [], []
    ordinary_close = SourceReader.close

    def close(reader):
        ordinary_close(reader)
        closed.append(True)

    def predict(image):
        assert len(closed) == len(pixels) + 1
        assert all(reference() is None for reference in pixels)
        gate = artifact_volume.artifact_volume_gate
        assert gate.acquire(blocking=False)
        gate.release()
        pixels.append(weakref.ref(image.pixels))
        if fail_at == "predict" and len(pixels) == 2:
            raise RuntimeError("injected prediction failure")
        mask = np.zeros((64, 64), np.int32)
        if len(pixels) == 1:
            mask[12:20, 12:20] = 7
        return {"instance_mask": mask}

    monkeypatch.setattr(SourceReader, "close", close)
    if fail_at == "read":

        def fail_read(*_args):
            raise OSError("injected pixel read failure")

        monkeypatch.setattr(SourceReader, "read_rgb_view", fail_read)
    if fail_at:
        with pytest.raises((OSError, RuntimeError), match="injected"):
            _evaluate(dataset, split, capability, predict)
        assert closed == [True] * (1 if fail_at == "read" else 2)
    else:
        # Exactly two held-out ROIs, one perfect and one missed; training is not read.
        assert _evaluate(dataset, split, capability, predict) == InstanceMetricResult(
            0.5, 0.5, None, {}
        )
        assert closed == [True, True]


@pytest.mark.parametrize("invalid", ["width", "height", "pixels"])
def test_all_validation_roi_bounds_are_checked_before_pixel_io(
    tmp_path, monkeypatch, invalid
):
    dataset, split, capability = _capture(tmp_path, validation_rois=2)
    limits = {"max_width": 128, "max_height": 128, "max_pixels": 10_000}
    limits[f"max_{invalid}"] = 4095 if invalid == "pixels" else 63

    def unexpected(*_args):
        pytest.fail("oversized validation plan reached pixel IO or prediction")

    monkeypatch.setattr(training_validation, "read_training_field_inputs", unexpected)
    with pytest.raises(ValueError, match="bounded inference"):
        _evaluate(dataset, split, capability, unexpected, raster_limits=limits)


def test_no_validation_is_distinct_from_undefined_validation_metrics(tmp_path):
    dataset, _, capability = _capture(tmp_path, objects=[])

    def unexpected(_image):
        pytest.fail("no-validation plan ran inference")

    assert _evaluate(dataset, {"type": "none"}, capability, unexpected) is None


@pytest.mark.parametrize(
    "output, message",
    [
        ({"instance_mask": np.zeros((32, 32), np.int32)}, "exact validation raster"),
        ({"instance_mask": np.zeros((64, 64), np.float32)}, "integer array"),
        ({"instance_mask": np.full((64, 64), -1, np.int32)}, "negative instance"),
    ],
)
def test_decoder_errors_propagate_without_resizing_or_partial_scores(
    tmp_path, output, message
):
    dataset, split, capability = _capture(tmp_path)
    with pytest.raises(ValueError, match=message):
        _evaluate(dataset, split, capability, lambda _: output)


def test_validation_grid_rechecks_overlap_even_after_dataset_admission(tmp_path):
    # The admission grid has centers x=10.5,11.5,... and sees no shared pixel.
    # The inference grid includes x=15, where these two complete objects overlap.
    dataset, split, capability = _capture(
        tmp_path,
        roi=_rectangle(10, 10, 29, 30),
        profile={**_profile(), "minSideModelPx": 32, "smallRoiContext": True},
        objects=[
            _object(1, _rectangle(12, 12, 15.1, 20)),
            _object(2, _rectangle(14.9, 12, 18, 20)),
        ],
    )
    with pytest.raises(ValueError, match="share a supervised real pixel"):
        _evaluate(
            dataset,
            split,
            capability,
            lambda _: {"instance_mask": np.zeros((32, 32), np.int32)},
        )


@pytest.mark.parametrize("channel", [True, 1.0, -1, 3])
def test_invalid_decoder_class_channels_are_not_silently_discarded(tmp_path, channel):
    dataset, split, capability = _capture(tmp_path, classified=True)
    mask = np.zeros((64, 64), np.int32)
    mask[12:20, 12:20] = 7
    with pytest.raises(ValueError, match="invalid instance class channel"):
        _evaluate(
            dataset,
            split,
            capability,
            lambda _: {"instance_mask": mask, "votes_per_instance": {7: channel}},
        )


def _capture(
    tmp_path,
    *,
    objects=None,
    roi=None,
    classified=False,
    calibration=None,
    validation_rois=1,
    profile=None,
    capability=None,
    model_output=None,
):
    """Publish and reverify a real PNG package with one training ROI and held-outs."""
    capability = capability or (
        _capability(profile) if profile else training_capability()
    )
    if model_output is not None:
        assert capability["output_type"] == model_output["type"]
    elif classified:
        capability["output_type"] = "instance_segmentation_and_classification"
        model_output = {
            "type": "instance_segmentation_and_classification",
            "classChannels": [
                {"channel": 0, "role": "background"},
                {
                    "channel": 1,
                    "role": "emitted",
                    "pathClass": "Leydig cell",
                },
                {
                    "channel": 2,
                    "role": "training_only",
                    "pathClass": "Other",
                },
            ],
        }
    else:
        model_output = {
            "type": "instance_segmentation",
            "classes": ["Leydig cell"],
        }
    extracted = _dataset(
        tmp_path / "capture",
        objects=objects,
        capability=capability,
        model_output=model_output,
    )
    if roi is not None:
        _write_roi(extracted, roi)
    _write_png(extracted.root / "artifacts/0/source.png", _pattern(64, 64))
    origin = _origin(extracted)
    if calibration is not None:
        origin["source"]["calibration"] = calibration
    (extracted.root / "artifacts/0/origin.json").write_bytes(canonical_bytes(origin))
    for index in range(1, validation_rois + 1):
        item = copy.deepcopy(extracted.manifest["rois"][0])
        item["source_roi_uuid"] = f"{index + 1:08x}-1111-4111-8111-111111111111"
        item["source_envelope"]["artifact_index"] = index
        extracted.manifest["rois"].append(item)
        extracted.manifest["artifacts"].append(
            copy.deepcopy(extracted.manifest["artifacts"][0])
        )
        for directory, names in (
            ("rois", ("roi.geojson", "objects.geojson")),
            ("artifacts", ("origin.json", "source.png")),
        ):
            for name in names:
                relative = f"{directory}/{index}/{name}"
                destination = extracted.root / relative
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(
                    (extracted.root / f"{directory}/0/{name}").read_bytes()
                )
                extracted.manifest["files"].append({"path": relative})
    # UUID sorting is part of the package contract, including the original ROI.
    extracted.manifest["rois"][0]["source_roi_uuid"] = (
        "00000001-1111-4111-8111-111111111111"
    )
    dataset = _retain(tmp_path, extracted, origin, capability=capability)
    split = {
        "type": "split",
        "unit": "roi",
        "assignment_source": "custom",
        "units": [
            {
                "unit_index": index,
                "partition": "training" if index == 0 else "validation",
            }
            for index in range(validation_rois + 1)
        ],
    }
    return dataset, split, capability


def _object(number, geometry):
    return _object_feature(f"{number:08x}-2222-4222-8222-222222222222", geometry)


def _evaluate(dataset, split, capability, predict, *, raster_limits=None):
    return evaluate_bounded_training_validation(
        dataset,
        split,
        predict,
        capability=capability,
        field_profile=capability["default_field_profile"],
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
        target_raster_limits=raster_limits
        or {
            "max_width": 128,
            "max_height": 128,
            "max_pixels": 10_000,
        },
        max_assignment_matrix_entries=100,
    )
