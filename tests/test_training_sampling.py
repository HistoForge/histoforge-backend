"""Full retained-dataset coverage reaches real losses and optimizer updates."""

import copy
import hashlib
import json
from functools import partial

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models import core
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedObjective,
)
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
)
from histoforge.cloud.sources.reader import SourceReader
from histoforge.cloud.training.profiles import full_coverage_sampler_profile
from histoforge.cloud.training.trainers.instance.batches import collate_training_fields
from histoforge.cloud.training.trainers.instance.loop import train_instance_fields
from histoforge.cloud.training.trainers.instance.sampling import TrainingEpochDataset
from tests.test_training_augmentation import _appearance_configured, _component
from tests.test_training_batches import _profile
from tests.test_training_contract import (
    classified_training_model_output,
    training_capability,
    training_model_output,
)
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _dataset, _rectangle
from tests.test_training_field_reader import (
    _origin,
    _pattern,
    _retain,
    _source_limits,
    _target,
    _write_png,
)
from tests.test_training_rasters import _capability
from torch import nn
from torch.utils.data import DataLoader


def _retained(tmp_path, *, classified=False, roi0=None, profile=None):
    """Three ROIs from one image have one admitted derived artifact per ROI."""
    capability = _capability(profile or _profile())
    if classified:
        capability["output_type"] = "instance_segmentation_and_classification"
    model_output = (
        classified_training_model_output() if classified else training_model_output()
    )
    extracted = _dataset(
        tmp_path / "capture", capability=capability, model_output=model_output
    )
    origin = _origin(extracted)
    origin["source"].update(width=192, height=96)
    origin["source_rect_at_level"].update(width=192, height=96)
    _write_png(extracted.root / "artifacts/0/source.png", _pattern(192, 96))
    geometries = [
        _rectangle(0, 0, 96, 80) if roi0 is None else roi0,
        _rectangle(110, 5, 125, 16),
        _rectangle(140, 10, 170, 30),
    ]
    template = copy.deepcopy(extracted.manifest["rois"][0])
    extracted.manifest["rois"] = []
    for index, geometry in enumerate(geometries):
        record = copy.deepcopy(template)
        record["source_roi_uuid"] = f"{str(index + 1) * 8}-1111-4111-8111-111111111111"
        record["source_envelope"]["artifact_index"] = index
        record["source_envelope"]["pixel_rect"].update(width=192, height=96)
        extracted.manifest["rois"].append(record)
        root = extracted.root / f"rois/{index}"
        root.mkdir(exist_ok=True)
        (root / "roi.geojson").write_bytes(
            canonical_bytes(
                {
                    "type": "Feature",
                    "geometry": canonical_geometry(geometry),
                    "properties": {},
                }
            )
        )
        if index:
            artifact_root = extracted.root / f"artifacts/{index}"
            artifact_root.mkdir()
            (artifact_root / "origin.json").write_bytes(canonical_bytes(origin))
            (artifact_root / "source.png").write_bytes(
                (extracted.root / "artifacts/0/source.png").read_bytes()
            )
            extracted.manifest["artifacts"].append(
                {"image_index": 0, "representation": "png"}
            )
            for name in ("origin.json", "source.png"):
                extracted.manifest["files"].append(
                    {
                        "path": f"artifacts/{index}/{name}",
                        "sha256": "0" * 64,
                        "size_bytes": 1,
                    }
                )
            (root / "objects.geojson").write_bytes(
                canonical_bytes({"type": "FeatureCollection", "features": []})
            )
            for name in ("roi.geojson", "objects.geojson"):
                extracted.manifest["files"].append(
                    {
                        "path": f"rois/{index}/{name}",
                        "sha256": "0" * 64,
                        "size_bytes": 1,
                    }
                )
    dataset = _retain(tmp_path, extracted, origin, capability=capability)
    split = {
        "type": "split",
        "unit": "roi",
        "assignment_source": "custom",
        "units": [
            {"unit_index": 0, "partition": "training"},
            {"unit_index": 1, "partition": "training"},
            {"unit_index": 2, "partition": "validation"},
        ],
    }
    return dataset, split, capability


def _epoch(
    retained, *, epoch=0, seed=1937, profile=None, head="hovernet", augmentation=None
):
    dataset, split, capability = retained
    return TrainingEpochDataset(
        dataset,
        split,
        capability=capability,
        field_profile=profile or capability["default_field_profile"],
        sampler=full_coverage_sampler_profile(),
        augmentation=_component([]) if augmentation is None else augmentation,
        training_seed=seed,
        epoch_index=epoch,
        head_type=head,
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
    )


def test_epoch_covers_all_training_pixels_and_retains_negative_fields(
    tmp_path, monkeypatch
):
    retained = _retained(tmp_path)

    def no_pixels(*_args):
        raise AssertionError("planning must not decode image pixels")

    with monkeypatch.context() as patch:
        patch.setattr(SourceReader, "read_rgb_view", no_pixels)
        epoch = _epoch(retained)
    assert len(epoch) == 13  # Twelve fields from the large ROI, one from the small ROI.
    coverage = np.zeros((80, 96), dtype=np.int32)
    negatives = 0
    for index, (roi, (x, y, width, height)) in enumerate(epoch.inputs.field_locations):
        assert roi in {0, 1}
        field = epoch[index]
        assert field.pixels.shape == (height, width, 3)
        assert field.labels.supervision_mask.all()
        assert field.labels.real_pixel_mask.all()
        if roi == 0:
            coverage[y : y + height, x : x + width] += 1
        else:
            assert (x, y, width, height) == (0, 0, 15, 11)
        negatives += not field.labels.targets.foreground.any()
    assert coverage.min() == 1 and coverage.max() > 1
    assert negatives == 12


def test_shuffle_is_frozen_by_seed_and_epoch_not_batch_or_worker_rng(tmp_path):
    retained = _retained(tmp_path)
    epoch = _epoch(retained)
    ordered = [(0, (x, y, 32, 32)) for y in (0, 24, 48) for x in (0, 24, 48, 64)]
    ordered.append((1, (0, 0, 15, 11)))
    encoded = json.dumps(
        {
            "domain": "sha256-canonical-json-pcg64-epoch-v1",
            "training_seed": 1937,
            "epoch_index": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    seed = int(hashlib.sha256(encoded).hexdigest(), 16)
    order = np.random.Generator(np.random.PCG64(seed)).permutation(13)
    assert epoch.inputs.field_locations == tuple(ordered[int(i)] for i in order)
    assert _epoch(retained).inputs.field_locations == epoch.inputs.field_locations
    for changed in (_epoch(retained, epoch=1), _epoch(retained, seed=1938)):
        assert set(changed.inputs.field_locations) == set(epoch.inputs.field_locations)
        assert changed.inputs.field_locations != epoch.inputs.field_locations


def test_rectangular_roi_omits_only_fields_with_no_supervised_real_pixels(tmp_path):
    profile = training_capability()["default_field_profile"]
    profile.update(
        side_model_px=16,
        small_roi_context=True,
        tile_overlap=0.1,
        preprocessing=_profile()["preprocessing"],
    )
    epoch = _epoch(_retained(tmp_path, roi0=_rectangle(0, 12, 96, 13), profile=profile))
    assert epoch.inputs.grids[0] == _target(96, 16, 0, 4.5)
    locations = [rect for roi, rect in epoch.inputs.field_locations if roi == 0]
    # The fixed planner has three tile rows. Its y=8 row sees only real context.
    assert len(locations) == 18
    assert {rect[1] for rect in locations} == {-8, 0}
    coverage = np.zeros((16, 96), dtype=bool)
    negatives = 0
    for index, (roi, (x, y, width, height)) in enumerate(epoch.inputs.field_locations):
        field = epoch[index]
        support = field.labels.supervision_mask & field.labels.real_pixel_mask
        assert support.any()
        if roi == 0:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(96, x + width), min(16, y + height)
            coverage[y0:y1, x0:x1] |= support[y0 - y : y1 - y, x0 - x : x1 - x]
        else:
            assert roi == 1
            assert not field.labels.targets.foreground.any()
            negatives += 1
    assert negatives > 0
    expected = np.zeros_like(coverage)
    expected[7] = True
    np.testing.assert_array_equal(coverage, expected)


@pytest.mark.parametrize("fixed", [False, True])
def test_context_is_centered_and_clipped_without_shifting_at_image_edge(
    tmp_path, fixed
):
    """Real context is input only; fixed-tile overhang is artificial padding."""
    profile = {**_profile(), "minSideModelPx": 32, "smallRoiContext": True}
    if fixed:
        profile = training_capability()["default_field_profile"]
        profile.update(
            side_model_px=32,
            small_roi_context=True,
            tile_overlap=0.1,
            preprocessing=_profile()["preprocessing"],
        )
    epoch = _epoch(_retained(tmp_path, profile=profile))
    assert epoch.inputs.grids[1] == _target(32, 27, 101.5, 0)
    locations = [
        (i, rect)
        for i, (roi, rect) in enumerate(epoch.inputs.field_locations)
        if roi == 1
    ]
    assert len(locations) == (4 if fixed else 1)
    pixels = _pattern(192, 96)
    for index, (x, y, width, height) in locations:
        field = epoch[index]
        yy, xx = np.mgrid[y : y + height, x : x + width]
        real = (xx >= 0) & (xx < 32) & (yy >= 0) & (yy < 27)
        # Model centers land on integer source x coordinates. Rasterio's existing
        # exact-edge tie includes the right edge here, not the left edge.
        support = real & (xx > 8) & (xx <= 23) & (yy >= 5) & (yy < 16)
        np.testing.assert_array_equal(field.labels.real_pixel_mask, real)
        np.testing.assert_array_equal(field.labels.supervision_mask, support)
        np.testing.assert_array_equal(
            field.pixels[real],
            np.rint(
                (
                    pixels[yy[real], (xx + 101)[real]].astype(float)
                    + pixels[yy[real], (xx + 102)[real]].astype(float)
                )
                / 2
            ).astype(np.uint8),
        )
        assert (real & ~support).any()
        assert not field.labels.targets.foreground.any()


def test_source_failure_is_not_silently_skipped(tmp_path, monkeypatch):
    epoch = _epoch(_retained(tmp_path))

    def failed_read(*_args):
        raise OSError("injected source read failure")

    monkeypatch.setattr(SourceReader, "read_rgb_view", failed_read)
    with pytest.raises(OSError, match="injected source read failure"):
        epoch[0]


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.9])
def test_fixed_training_sampling_preserves_tile_geometry_and_sample_normalization(
    tmp_path, monkeypatch, overlap
):
    profile = training_capability()["default_field_profile"]
    profile.update(
        side_model_px=32,
        small_roi_context=False,
        tile_overlap=overlap,
        preprocessing=_profile()["preprocessing"],
    )
    epoch = _epoch(_retained(tmp_path, profile=profile))
    expected = []

    def forward(_net, batch):
        expected.extend(batch.copy())
        return batch, np.zeros((len(batch), 256), dtype=np.float32)

    monkeypatch.setattr(core, "_forward", forward)
    core.run_net(
        object(),
        _pattern(96, 80)[None],
        bsize=32,
        tile_overlap=overlap,
    )
    positions = sorted(
        (location[1], index)
        for index, location in enumerate(epoch.inputs.field_locations)
        if location[0] == 0
    )
    positions.sort(key=lambda item: (item[0][1], item[0][0]))
    assert len(positions) == len(expected)
    for (_, index), pixels in zip(positions, expected):
        field = epoch[index]
        np.testing.assert_array_equal(field.pixels.transpose(2, 0, 1), pixels)
        actual = collate_training_fields(
            [field], preprocess=normalize99, field_profile=profile, class_count=0
        )
        # TRAIN normalizes this sample, not the larger image it came from.
        values = pixels.transpose(1, 2, 0)[field.labels.real_pixel_mask]
        low, high = np.percentile(values.astype(np.float32), (1, 99), axis=0)
        normalized = np.zeros_like(pixels)
        for channel in range(3):
            if high[channel] - low[channel] > 0.001:
                normalized[channel][field.labels.real_pixel_mask] = (
                    values[:, channel] - low[channel]
                ) / (high[channel] - low[channel])
        np.testing.assert_array_equal(actual.images[0], normalized)
        assert not field.labels.supervision_mask[~field.labels.real_pixel_mask].any()


@pytest.mark.parametrize("defect", ["negative_rois", "range", "preprocessing"])
def test_mismatched_task_or_profile_is_rejected_before_pixels(
    tmp_path, monkeypatch, defect
):
    dataset, split, capability = _retained(tmp_path)
    profile = copy.deepcopy(capability["default_field_profile"])
    if defect == "negative_rois":
        capability["supports_negative_rois"] = False
    elif defect == "range":
        profile["maxSideModelPx"] = 128
    else:
        profile["preprocessing"] = "different-normalization"

    def unexpected(*_args):
        pytest.fail("mismatched run reached pixel IO")

    monkeypatch.setattr(SourceReader, "read_rgb_view", unexpected)
    with pytest.raises(ValueError, match="capability"):
        _epoch((dataset, split, capability), profile=profile)


def test_data_workers_return_identical_augmented_fields(tmp_path):
    capability, profile = _appearance_configured()
    augmentation = _component(
        [
            "rotate-v1",
            "flip-v1",
            "hed-v1",
            "brightness-contrast-v1",
            "blur-v1",
            "noise-v1",
        ],
        capability=capability,
        profile=profile,
    )
    epoch = _epoch(
        _retained(tmp_path, classified=True),
        head="flow",
        augmentation=augmentation,
    )
    collate = partial(
        collate_training_fields,
        preprocess=normalize99,
        field_profile=_profile(),
        class_count=2,
    )
    baseline = list(DataLoader(epoch, batch_size=2, collate_fn=collate, shuffle=False))
    parallel = list(
        DataLoader(
            epoch,
            batch_size=2,
            collate_fn=collate,
            shuffle=False,
            num_workers=2,
            multiprocessing_context="spawn",
        )
    )
    assert len(baseline) == len(parallel)
    assert all(batch.instance_flow_weights is not None for batch in baseline)
    weights = torch.cat([batch.instance_flow_weights for batch in baseline])
    assert torch.any(weights > 0)
    assert torch.any(weights == 0)
    for first, second in zip(baseline, parallel):
        for name in (
            "images",
            "directional_targets",
            "foreground_targets",
            "supervision_mask",
            "real_pixel_mask",
            "class_targets",
            "instance_flow_weights",
        ):
            left, right = getattr(first, name), getattr(second, name)
            assert (left is None) == (right is None)
            if left is not None:
                assert torch.equal(left, right)


@pytest.mark.parametrize(
    "head,flow_weighting",
    [("flow", "pixel"), ("flow", "equal_instance"), ("hovernet", "pixel")],
)
@pytest.mark.parametrize("classified", [False, True])
def test_full_epoch_reaches_both_heads_and_uneven_accumulation_without_lost_fields(
    tmp_path, head, flow_weighting, classified
):
    retained = _retained(tmp_path, classified=classified)
    epoch = _epoch(retained, head=head, augmentation=_component())
    class_count = 2 if classified else 0
    base = nn.Conv2d(3, class_count + (3 if head == "flow" else 4), 1)
    with torch.no_grad():
        base.weight.fill_(0.05)
        base.bias.fill_(0.1)
    results = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for batch_size, accumulation in ((13, 1), (4, 4), (1, 13), (4, 5)):
            model = copy.deepcopy(base)
            objective = ManagedObjective(
                class_count=class_count,
                classification_objective="cross_entropy+tversky",
                use_uncertainty_weighting=classified,
            )
            loader = DataLoader(
                epoch,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=partial(
                    collate_training_fields,
                    preprocess=normalize99,
                    field_profile=_profile(),
                    class_count=class_count,
                ),
            )
            outcome = train_instance_fields(
                model,
                torch.optim.SGD(
                    [*model.parameters(), *objective.parameters()], lr=0.01
                ),
                lambda _epoch: loader,
                training_forward=instance_training_predictions,
                head_type=head,
                class_count=class_count,
                learning_rate_scales=[1.0],
                gradient_accumulation_steps=accumulation,
                classification_objective="cross_entropy+tversky",
                class_weights=torch.tensor([0.25, 2.0]) if classified else None,
                weight_tversky_by_class=classified and head == "flow",
                objective=objective,
                flow_loss_weighting=flow_weighting,
            )[0]
            assert outcome.field_count == 13 and outcome.optimizer_steps == 1
            assert not torch.equal(model.weight, base.weight)
            results.append((model, outcome, objective))
    finally:
        torch.set_num_threads(previous)
    for model, outcome, objective in results[1:]:
        torch.testing.assert_close(
            model.weight, results[0][0].weight, atol=2e-7, rtol=2e-5
        )
        torch.testing.assert_close(model.bias, results[0][0].bias, atol=2e-7, rtol=2e-5)
        assert outcome.mean_loss == pytest.approx(results[0][1].mean_loss, abs=1e-6)
        if classified:
            assert torch.count_nonzero(objective.log_variances) == 3
            torch.testing.assert_close(
                objective.log_variances, results[0][2].log_variances
            )
