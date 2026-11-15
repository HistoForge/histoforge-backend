"""Shared field preparation uses arrays and grids, not instance-task surrogates."""

from itertools import product

import numpy as np
import pytest
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.cloud.sources.reader import SourceReader
from histoforge.cloud.training.contract import validate_instance_training_settings
from histoforge.cloud.training.datasets.field_reader import read_training_field_inputs
from histoforge.cloud.training.fields.augmentation import transform_spatial_array
from histoforge.cloud.training.fields.batches import (
    field_tensor,
    pad_field_tensors,
    prepare_field_batch,
)
from histoforge.cloud.training.fields.sampling import (
    bounded_field_read,
    plan_training_fields,
)
from tests.test_training_batches import _profile
from tests.test_training_contract import classified_training_model_output
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _dataset, _rectangle
from tests.test_training_field_reader import (
    _origin,
    _pattern,
    _retain,
    _source_limits,
    _write_png,
    _write_roi,
)
from tests.test_training_rasters import _capability
from tests.test_training_spatial_augmentation import _coordinate_map


@pytest.mark.parametrize("output_type", ["region_classification", "future-task"])
@pytest.mark.parametrize("settings", [{}, {"min_instance_pixels": 0}])
def test_unsupported_tasks_do_not_inherit_instance_thresholds(output_type, settings):
    with pytest.raises(ValueError, match="do not support this model output"):
        validate_instance_training_settings(settings, {"type": output_type})


@pytest.mark.parametrize("defect", ["shape", "dtype", "nonfinite", "artificial"])
def test_preprocessing_callable_cannot_change_the_scientific_tensor_contract(defect):
    image = _pattern(16, 16)
    real = np.ones((16, 16), dtype=bool)
    real[0] = False

    def preprocess(pixels, *, real_pixel_mask):
        result = np.zeros(pixels.shape, np.float32)
        if defect == "shape":
            return result[:1]
        if defect == "dtype":
            return result.astype(np.float64)
        result[0, 0] = np.nan if defect == "nonfinite" else 1
        return result

    with pytest.raises(ValueError, match="preprocessing must"):
        prepare_field_batch([image], [real], _profile(), preprocess=preprocess)


@pytest.mark.parametrize("negative", [False, True])
def test_dense_and_instance_tasks_share_pixel_reads_and_field_plans(
    tmp_path, monkeypatch, negative
):
    """Use admitted dense geometry directly, without building flow/HoVer targets."""

    plans = []
    pixels = _pattern(64, 64)
    for output_type in (
        "instance_segmentation_and_classification",
        "dense_classification",
    ):
        profile = {**_profile(), "maxSideModelPx": 32}
        capability = _capability(profile)
        capability["output_type"] = output_type
        if output_type == "dense_classification":
            capability.pop("flow_loss_weighting_options")
            capability["default_training_settings"].pop("flow_loss_weighting")
        output = {**classified_training_model_output(), "type": output_type}
        root = tmp_path / output_type
        extracted = _dataset(
            root / "capture",
            capability=capability,
            model_output=output,
            objects=[] if negative else None,
        )
        _write_png(extracted.root / "artifacts/0/source.png", pixels)
        _write_roi(extracted, _rectangle(0, 0, 60, 50))
        dataset = _retain(root, extracted, _origin(extracted), capability=capability)

        def unexpected(*_args):
            pytest.fail("field planning must not decode RGB")

        with monkeypatch.context() as patch:
            patch.setattr(SourceReader, "read_rgb_view", unexpected)
            plan = plan_training_fields(
                dataset,
                {"type": "none"},
                capability=capability,
                field_profile=profile,
                api_limits=training_api_limits(),
            )
        plans.append(plan)
        resolved, grids, locations = plan
        assert resolved == profile
        assert len(locations) == 6
        coverage = np.zeros((50, 60), dtype=bool)
        for roi, rectangle in locations:
            read, padding = bounded_field_read(rectangle, grids[roi]["raster"])
            assert not any(padding)
            actual, geometry, objects, bounds = read_training_field_inputs(
                dataset, roi, grids[roi], read, _source_limits(), 8 * 1024**2
            )
            x, y, width, height = rectangle
            np.testing.assert_array_equal(actual, pixels[y : y + height, x : x + width])
            coverage[y : y + height, x : x + width] = True
            assert geometry == _rectangle(0, 0, 60, 50)
            assert len(objects) == (0 if negative else 1)
            assert bounds == (0, 0, 64, 64)
        assert coverage.all()
    assert plans[0] == plans[1]


@pytest.mark.parametrize(
    "turns,horizontal,vertical", list(product(range(4), (False, True), (False, True)))
)
def test_spatial_array_transform_preserves_scalar_class_labels(
    turns, horizontal, vertical
):
    labels = np.arange(7 * 11).reshape(7, 11) % 5
    original = labels.copy()
    actual = transform_spatial_array(
        labels,
        quarter_turns=turns,
        flip_horizontal=horizontal,
        flip_vertical=vertical,
    )
    matrix, offset, shape = _coordinate_map(7, 11, turns, horizontal, vertical)
    assert actual.shape == shape
    for y, x in product(range(7), range(11)):
        out_y, out_x = matrix @ (y, x) + offset
        assert actual[out_y, out_x] == labels[y, x]
    actual[:] = 0
    np.testing.assert_array_equal(labels, original)


def test_class_arrays_use_the_same_normalization_offsets_and_batch_support():
    pixels = [_pattern(22, 9).astype(np.float32), _pattern(17, 33).astype(np.float32)]
    real = [np.ones(image.shape[:2], dtype=bool) for image in pixels]
    real[0][0] = False
    pixels[0][0] = np.nan
    labels = [np.indices(mask.shape).sum(axis=0).astype(np.int64) % 3 for mask in real]
    profile = {**_profile(), "minSideModelPx": 32}
    images, padding, shape = prepare_field_batch(
        pixels, real, profile, preprocess=normalize99
    )
    assert padding == [(5, 5, 11, 12), (7, 8, 7, 8)]
    assert shape == (48, 32)
    batched_images = pad_field_tensors(images, padding, *shape).numpy()
    batched_labels = pad_field_tensors(
        [field_tensor(a) for a in labels], padding, *shape
    ).numpy()
    batched_real = pad_field_tensors(
        [field_tensor(a) for a in real], padding, *shape
    ).numpy()
    for index, (image, mask, target) in enumerate(zip(pixels, real, labels)):
        left, _, top, _ = padding[index]
        height, width = mask.shape
        expected = np.zeros_like(image)
        for channel in range(3):
            lower, upper = np.percentile(image[..., channel][mask], (1, 99))
            if upper - lower > 0.001:
                expected[..., channel][mask] = (image[..., channel][mask] - lower) / (
                    upper - lower
                )
        np.testing.assert_array_equal(
            batched_images[index, :, top : top + height, left : left + width],
            expected.transpose(2, 0, 1),
        )
        np.testing.assert_array_equal(
            batched_labels[index, top : top + height, left : left + width], target
        )
        np.testing.assert_array_equal(
            batched_real[index, top : top + height, left : left + width], mask
        )
        outside = np.ones(shape, dtype=bool)
        outside[top : top + height, left : left + width] = False
        assert not batched_images[index, :, outside].any()
        assert not batched_labels[index, outside].any()
        assert not batched_real[index, outside].any()
