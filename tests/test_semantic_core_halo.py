"""Semantic processing windows preserve class selection, tiling and core geometry."""

import weakref

import numpy as np
import pytest
import torch
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.architectures.semantic.runner import SemanticRunner
from histoforge.cloud.architectures.semantic.runtime import SemanticModelRuntime
from histoforge.cloud.inference.plans import (
    build_core_halo_profile,
    build_expected_model_execution_plan,
)
from histoforge.cloud.inference.runs.core_halo import translate_core_regions
from histoforge.cloud.inference.runs.core_halo_output import encode_semantic_window
from histoforge.cloud.inference.semantic_features import (
    feature_collection_from_class_map,
)
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from shapely.geometry import shape

_VARIABLE_PROFILE = {
    "planner": "variable-field-v2",
    "minSideModelPx": 32,
    "maxSideModelPx": 64,
    "preferredTileSideModelPx": 32,
    "alignmentModelPx": 16,
    "outerArtificialPaddingModelPx": 8,
    "smallRoiContext": False,
    "tileOverlap": 0.25,
    "aggregation": "cellpose-tapered-mean-v1",
    "preprocessing": "normalize99-real-image-v1",
}

_CLASSES = {2: "Tubule", 3: "Lumen"}
_BUDGET = {
    "max_features": 1_000,
    "max_vertices": 100_000,
    "max_record_bytes": 1_000_000,
    "max_transferred_bytes": 10_000_000,
    "max_decoded_bytes": 10_000_000,
}


def _runner(*, network=None, field_options=None):
    runner = SemanticRunner._get_user_cls()()
    runner.ref = "owner/semantic"
    runner.runtime = SemanticModelRuntime(
        network=network,
        num_classes=4,
        field_options=field_options or {"bsize": 32, "tile_overlap": 0.25},
        warmup_side=32,
        batch_size=3,
        emitted_classes=tuple(_CLASSES.values()),
        class_by_channel=_CLASSES,
    )
    return runner


def _plan(width=96, height=96, core=32, halo=8):
    return build_expected_model_execution_plan(
        build_core_halo_profile(core, core, halo, "1" * 64),
        {"width": width, "height": height},
        "2" * 64,
        100,
    )


def _maps(labels):
    maps = np.zeros((*labels.shape, 7), np.float32)
    np.put_along_axis(maps, labels[..., None], 2, axis=-1)
    maps[..., 4:] = 1_000_000  # Unused segmentation channels must not win.
    return maps


def _window_collection(runner, image, plan, window, budget, *, percentiles=None):
    """Expose core geometry in local coordinates before separate translation checks."""

    from histoforge.cloud.inference import semantic_features

    class_map, class_by_channel = runner._predict_core_halo_window(
        lambda: image, plan, window, channel_percentiles=percentiles
    )
    rect, core = window["input_rect"], window["core_rect"]
    return semantic_features.feature_collection_from_class_map(
        class_map,
        class_by_channel,
        runner.runtime.emitted_classes,
        budget,
        x_offset=core["x"] - rect["x"],
        y_offset=core["y"] - rect["y"],
    )


@pytest.mark.parametrize("halo", [0, 8])
@pytest.mark.parametrize("unit_id", [0, 4, 8], ids=["top-left", "interior", "last"])
def test_all_channels_compete_before_core_crop_and_emission(monkeypatch, halo, unit_id):
    plan = _plan(width=85, height=81, halo=halo)
    window = plan["windows"][unit_id]
    rect, core = window["input_rect"], window["core_rect"]
    labels = np.full((rect["height"], rect["width"]), 3, np.int32)
    left, top = core["x"] - rect["x"], core["y"] - rect["y"]
    core_labels = labels[top : top + core["height"], left : left + core["width"]]
    core_labels[:] = 2
    core_labels[1:4, 1:4] = 0
    core_labels[5:8, 5:8] = 1
    core_labels[9:12, 9:12] = 3
    maps = _maps(labels)
    # Emitted channels score above each other, but not above non-emitted winners.
    maps[labels < 2, 2] = 1
    image = RasterizedImage(
        np.zeros((*labels.shape, 3), np.uint8), np.ones(labels.shape, bool)
    )
    calls = {}

    def predict(pixels, **kwargs):
        assert pixels is image.pixels
        calls.update(kwargs)
        return maps

    monkeypatch.setattr(
        "histoforge._vendor.histoforge_models.pathology.window_maps.predict_window_maps",
        predict,
    )
    runner = _runner()
    local = _window_collection(runner, image, plan, window, _BUDGET)
    actual = translate_core_regions(
        window,
        local,
        _CLASSES.values(),
        target_width=85,
        target_height=81,
        result_budget=_BUDGET,
    )
    expected = feature_collection_from_class_map(
        core_labels,
        _CLASSES,
        _CLASSES.values(),
        _BUDGET,
        x_offset=core["x"],
        y_offset=core["y"],
    )

    assert canonical_bytes(actual) == canonical_bytes(expected)
    assert len(actual["features"]) == 2
    assert sum(shape(item["geometry"]).area for item in actual["features"]) == (
        core["width"] * core["height"] - 18
    )
    assert calls["real_pixel_mask"] is image.real_pixel_mask
    assert (calls["target_width"], calls["target_height"]) == (85, 81)
    assert (calls["window_x"], calls["window_y"]) == (rect["x"], rect["y"])
    assert calls["field_options"] is runner.runtime.field_options
    assert calls["batch_size"] == 3


def test_semantic_regions_never_use_instance_interior_point_ownership(monkeypatch):
    # All pixels are one class. The region crosses every artificial input edge;
    # its core crop must survive, not fail the instance completeness check.
    plan = _plan()
    window = plan["windows"][4]
    labels = np.full((48, 48), 2, np.int32)
    monkeypatch.setattr(
        "histoforge._vendor.histoforge_models.pathology.window_maps.predict_window_maps",
        lambda *_args, **_kwargs: _maps(labels),
    )
    local = _window_collection(
        _runner(),
        RasterizedImage(np.zeros((48, 48, 3), np.uint8), np.ones((48, 48), bool)),
        plan,
        window,
        _BUDGET,
    )
    actual = translate_core_regions(
        window,
        local,
        _CLASSES.values(),
        target_width=96,
        target_height=96,
        result_budget=_BUDGET,
    )

    assert len(actual["features"]) == 1
    assert shape(actual["features"][0]["geometry"]).bounds == (32, 32, 64, 64)


def test_core_labels_do_not_retain_window_fields_or_full_window_labels(monkeypatch):
    from histoforge._vendor.histoforge_models.pathology import semantic_inference
    from histoforge.cloud.inference import semantic_features

    references = []
    class_labels = semantic_inference.class_labels_from_maps

    def fields(*_args, **_kwargs):
        output = _maps(np.full((48, 48), 2, np.int32))
        references.append(weakref.ref(output))
        return output

    def labels(output, count):
        result = class_labels(output, count)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(
        "histoforge._vendor.histoforge_models.pathology.window_maps.predict_window_maps",
        fields,
    )
    monkeypatch.setattr(semantic_inference, "class_labels_from_maps", labels)
    image = RasterizedImage(np.zeros((48, 48, 3), np.uint8), np.ones((48, 48), bool))
    plan = _plan()
    window = plan["windows"][4]
    runner = _runner()
    result = runner.runtime.predict_window(image, plan, window)
    assert set(result) == {"class_map"}
    assert result["class_map"].shape == (32, 32)
    assert result["class_map"].base is None
    assert all(reference() is None for reference in references)
    convert = semantic_features.feature_collection_from_class_map

    def geometry(core_labels, *args, **kwargs):
        assert all(reference() is None for reference in references)
        assert core_labels.base is None
        return convert(core_labels, *args, **kwargs)

    monkeypatch.setattr(
        semantic_features, "feature_collection_from_class_map", geometry
    )
    collection = _window_collection(runner, image, plan, window, _BUDGET)
    assert len(collection["features"]) == 1
    assert shape(collection["features"][0]["geometry"]).bounds == (8, 8, 40, 40)


class _SyntheticSemanticNet(torch.nn.Module):
    """Expose normalized RGB as class logits and record actual CPU tile batches."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, tiles):
        self.batches.append(tiles.numpy().copy())
        baseline = torch.full_like(tiles[:, 0], 0.2)
        logits = torch.stack([baseline, tiles[:, 0], tiles[:, 1], tiles[:, 2]], dim=1)
        tail = torch.full_like(tiles, 10)
        return torch.cat([logits, tail], dim=1), torch.zeros((tiles.shape[0], 256))


class _ContextSemanticNet(_SyntheticSemanticNet):
    """Use neighboring pixels and the complete neural field for class logits."""

    def forward(self, tiles):
        local = torch.nn.functional.avg_pool2d(tiles, 3, stride=1, padding=1)
        context = tiles.mean(dim=(-2, -1), keepdim=True)
        logits = torch.cat(
            [torch.full_like(tiles[:, :1], 0.25), local + context], dim=1
        )
        return torch.cat([logits, torch.zeros_like(tiles)], dim=1), torch.zeros(
            (tiles.shape[0], 256)
        )


@pytest.mark.parametrize(
    "field_options",
    [{"bsize": 32, "tile_overlap": 0.25}, {"field_profile": _VARIABLE_PROFILE}],
    ids=["fixed", "variable"],
)
def test_shared_target_scaling_preserves_contextual_neural_fields(field_options):
    pixels = np.random.default_rng(175).integers(0, 256, (96, 112, 3), np.uint8)
    pixels[:, 56:] //= 4
    support = np.ones((96, 112), bool)
    support[:3] = False
    image = RasterizedImage(pixels, support)
    runtime = _runner(
        network=_ContextSemanticNet(), field_options=field_options
    ).runtime
    expected = runtime.predict(image)["class_map"]
    pairs = np.percentile(pixels.astype(np.float32)[support], (1, 99), axis=0).T
    plan = _plan(width=112, height=96, core=32, halo=64)
    actual = np.empty_like(expected)
    for window in plan["windows"]:
        rect, core = window["input_rect"], window["core_rect"]
        region = (
            slice(rect["y"], rect["y"] + rect["height"]),
            slice(rect["x"], rect["x"] + rect["width"]),
        )
        predicted = runtime.predict_window(
            RasterizedImage(pixels[region], support[region]),
            plan,
            window,
            channel_percentiles=pairs,
        )["class_map"]
        actual[
            core["y"] : core["y"] + core["height"],
            core["x"] : core["x"] + core["width"],
        ] = predicted
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(expected >= 2) > 0


@pytest.mark.parametrize(
    "field_options",
    [{"bsize": 32, "tile_overlap": 0.25}, {"field_profile": _VARIABLE_PROFILE}],
    ids=["fixed", "variable"],
)
def test_complete_window_preserves_ordinary_tiles_normalization_and_regions(
    field_options,
):
    y, x = np.mgrid[:96, :112]
    pixels = np.stack([x, y, (x + y) / 2], axis=-1).astype(np.float32)
    support = np.ones((96, 112), bool)
    support[:3] = False
    pixels[:3] = 1_000_000  # Artificial pixels must not affect the percentiles.
    image = RasterizedImage(pixels, support)
    network = _SyntheticSemanticNet()
    runner = _runner(network=network, field_options=field_options)
    expected = runner._infer_features(image, _BUDGET)
    batches = network.batches.copy()
    network.batches.clear()
    plan = _plan(width=112, height=96, core=112)

    actual = _window_collection(runner, image, plan, plan["windows"][0], _BUDGET)

    assert expected["features"]
    assert canonical_bytes(actual) == canonical_bytes(expected)
    assert len(batches) > 1
    assert len(network.batches) == len(batches)
    assert batches[0].shape[0] == runner.runtime.batch_size
    for actual_batch, expected_batch in zip(network.batches, batches, strict=True):
        np.testing.assert_array_equal(actual_batch, expected_batch)


def test_modal_entry_points_use_existing_lifecycle_and_callbacks(monkeypatch):
    from histoforge.cloud.inference.runs import core_halo_runtime, runtime

    runner_type = SemanticRunner._get_user_cls()
    runner = _runner()
    runner.activation, runner.source_store = object(), object()
    runner.source_files, runner.artifact_files = object(), object()
    request, limits = {}, {}
    calls = []

    def capture(*args, source_bindings, channel_percentiles=None):
        assert source_bindings == ()
        calls.append(args)
        return {"complete": True}

    monkeypatch.setattr(runtime, "execute_durable_finite_input", capture)
    monkeypatch.setattr(core_halo_runtime, "execute_core_halo_unit", capture)
    for method, args in (
        ("infer_durable_input", (request, 2, limits)),
        ("infer_durable_core_halo_unit", (request, 2, 3, limits)),
    ):
        assert runner_type.__dict__[method]._get_raw_f()(runner, *args) == {
            "complete": True
        }
    assert calls == [
        (
            request,
            2,
            limits,
            runner.activation,
            runner.source_store,
            runner.source_files,
            runner.artifact_files,
            runner._infer_features,
        ),
        (
            request,
            2,
            3,
            limits,
            runner.activation,
            runner.source_store,
            runner.source_files,
            runner.artifact_files,
            runner._predict_core_halo_window,
            encode_semantic_window,
        ),
    ]
