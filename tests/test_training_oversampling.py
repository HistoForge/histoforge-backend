"""Opt-in field draws retain original TRAIN evidence and pixel-exact visibility."""

import copy
import hashlib
import json
from dataclasses import fields
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.canonical_json import canonical_digest
from histoforge.cloud.artifacts.volume import artifact_volume_gate
from histoforge.cloud.sources.reader import SourceReader
from histoforge.cloud.training.contract import (
    validate_model_training_capability,
    validate_training_settings,
    validate_training_settings_for_capability,
)
from histoforge.cloud.training.datasets import io as training_io
from histoforge.cloud.training.engine import TrainingCancelled
from histoforge.cloud.training.profiles import instance_sampler_profile
from histoforge.cloud.training.runs.report import validate_training_report
from histoforge.cloud.training.trainers.instance import sampling
from histoforge.cloud.training.trainers.instance.batches import collate_training_fields
from tests.test_training_augmentation import _appearance_configured, _component
from tests.test_training_batches import single_thread_torch as single_thread_torch
from tests.test_training_class_weights import _capability, _count, _object, _retained
from tests.test_training_contract import classified_training_model_output
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _rectangle
from tests.test_training_execution import _execute, _setup
from tests.test_training_field_reader import _source_limits
from tests.test_training_report import queued_report
from torch.utils.data import DataLoader


def _population(tmp_path, *, negative=False):
    capability = _capability()
    capability["default_field_profile"].update(maxSideModelPx=32)
    objects = (
        []
        if negative
        else [
            _object(1, _rectangle(1, 1, 9, 9)),
            _object(2, _rectangle(27, 1, 31, 5)),
            _object(3, _rectangle(49, 2, 51, 4), "Other"),
        ]
    )
    dataset = _retained(
        tmp_path, _rectangle(0, 0, 96, 32), objects, capability, width=96, height=32
    )
    return dataset, capability


def test_shared_settings_and_sampler_fixture():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/contracts/training-oversampling-v1.json"
        ).read_text()
    )
    for case in fixture["cases"]:
        settings = validate_training_settings(case["settings"])
        assert canonical_digest(settings) == case["sha256"]
        assert instance_sampler_profile(settings["oversample"]) == case["sampler"]
        for invalid in fixture["invalid_values"]:
            with pytest.raises(ValueError):
                validate_training_settings({**settings, "oversample": invalid})
    report = validate_training_report(fixture["report"]["value"], training_api_limits())
    assert canonical_digest(report) == fixture["report"]["sha256"]


def _epoch(dataset, capability, *, enabled=True, augmentation=None, counts=None):
    return sampling.TrainingEpochDataset(
        dataset,
        {"type": "none"},
        capability=capability,
        field_profile=capability["default_field_profile"],
        sampler=instance_sampler_profile(enabled),
        augmentation=_component([]) if augmentation is None else augmentation,
        training_seed=1937,
        epoch_index=0,
        head_type="flow",
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
        class_pixel_counts=(
            dict(enumerate(_count(dataset, capability))) if counts is None else counts
        ),
    )


def test_probabilities_use_original_pixels_visible_instances_and_training_only(
    tmp_path,
):
    dataset, capability = _population(tmp_path)
    assert _count(dataset, capability) == (2988, 80, 4)
    epoch = _epoch(dataset, capability)
    expected = np.array([2 / 80, 1 / 80 + 1 / 4, 1 / 4, 1e-10])
    expected /= expected.sum()
    np.testing.assert_array_equal(epoch.probabilities, expected)
    assert len(epoch) == 4
    # Overlapping fields expose the boundary objects twice, but denominators
    # remain 80 and 4 original pixels, not overlap-expanded counts.
    ordered = tuple((0, (x, 0, 32, 32)) for x in (0, 24, 48, 64))
    observed = []
    for index in range(5):
        encoded = json.dumps(
            {
                "domain": "sha256-canonical-json-pcg64-instance-field-epoch-v1",
                "training_seed": 1937,
                "epoch_index": index,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        rng = np.random.Generator(
            np.random.PCG64(int(hashlib.sha256(encoded).hexdigest(), 16))
        )
        draws = rng.choice(4, size=4, replace=True, p=expected)
        selected = epoch.for_epoch(index)
        assert selected.inputs.field_locations == tuple(ordered[i] for i in draws)
        assert selected.probabilities is epoch.probabilities
        observed.append(selected.inputs.field_locations)
    assert len(set(observed)) > 1
    assert any(len(set(draws)) < 4 for draws in observed)
    assert epoch.inputs.epoch_index == 0
    assert epoch.for_epoch(4).for_epoch(0).inputs == epoch.inputs


@pytest.mark.parametrize("negative", [False, True])
def test_sampling_preparation_is_bounded_metadata_only_and_cached(
    tmp_path, monkeypatch, negative
):
    dataset, capability = _population(tmp_path, negative=negative)
    counts = dict(enumerate(_count(dataset, capability)))
    reads, windows = [], []
    read = training_io.read_training_roi_geometry
    rasterize = sampling.rasterize_model_geometry

    def gated(*args):
        assert artifact_volume_gate.locked()
        reads.append(args[1])
        return read(*args)

    def bounded(geometry, rect):
        assert not artifact_volume_gate.locked()
        assert 0 < rect["width"] <= 32 and 0 < rect["height"] <= 32
        windows.append(rect)
        return rasterize(geometry, rect)

    def forbidden(*args, **kwargs):
        pytest.fail("sampling may not read RGB or generate scientific targets")

    monkeypatch.setattr(training_io, "read_training_roi_geometry", gated)
    monkeypatch.setattr(sampling, "rasterize_model_geometry", bounded)
    monkeypatch.setattr(SourceReader, "read_rgb_view", forbidden)
    monkeypatch.setattr(sampling, "read_training_field", forbidden)
    epoch = _epoch(dataset, capability, counts=counts)
    assert reads == [0, 0]  # Population geometry, then class visibility.
    assert windows
    if negative:
        np.testing.assert_array_equal(epoch.probabilities, np.full(4, 0.25))
    for index in (1, 2, 9, 1):
        assert len(epoch.for_epoch(index)) == 4
    assert reads == [0, 0]


def test_one_field_means_one_draw_and_zero_count_class_is_floored(tmp_path):
    capability = _capability()
    dataset = _retained(
        tmp_path,
        _rectangle(0, 0, 16, 16),
        [_object(1, _rectangle(1, 1, 3, 3), "Other")],
        capability,
    )
    epoch = _epoch(dataset, capability)
    assert _count(dataset, capability) == (252, 0, 4)
    assert len(epoch) == 1
    np.testing.assert_array_equal(epoch.probabilities, [1.0])


def test_disconnected_components_count_as_one_captured_instance(tmp_path):
    capability = _capability()
    capability["default_field_profile"].update(maxSideModelPx=32)
    disconnected = {
        "type": "MultiPolygon",
        "coordinates": [
            _rectangle(1, 1, 3, 3)["coordinates"],
            _rectangle(10, 1, 12, 3)["coordinates"],
        ],
    }
    with_hole = _rectangle(49, 1, 53, 5)
    with_hole["coordinates"].append(_rectangle(50, 2, 52, 4)["coordinates"][0])
    dataset = _retained(
        tmp_path,
        _rectangle(0, 0, 96, 32),
        [_object(1, disconnected), _object(2, with_hole, "Other")],
        capability,
        width=96,
        height=32,
    )
    assert _count(dataset, capability) == (3052, 8, 12)
    expected = np.array([1 / 8, 1 / 12, 1 / 12, 1e-10])
    expected /= expected.sum()
    np.testing.assert_array_equal(_epoch(dataset, capability).probabilities, expected)


def test_vector_intersection_without_a_real_supervised_pixel_is_not_an_instance(
    tmp_path,
):
    capability = _capability()
    capability["default_field_profile"].update(maxSideModelPx=32, smallRoiContext=True)
    # The first tiny polygon intersects the ROI but no pixel center; the second
    # extends outside the physical image. Neither can inflate field visibility.
    dataset = _retained(
        tmp_path,
        _rectangle(0.5, 0.5, 64, 32),
        [
            _object(1, _rectangle(18.1, 18.1, 18.4, 18.4)),
            _object(2, _rectangle(63.5, 1, 66, 3)),
            _object(3, _rectangle(28, 3, 30, 5), "Other"),
        ],
        capability,
    )
    epoch = _epoch(dataset, capability)
    assert _count(dataset, capability)[1] == 0
    expected = np.array([1 / 4, 1 / 4, 1e-10])
    expected /= expected.sum()
    np.testing.assert_array_equal(epoch.probabilities, expected)


def test_repeated_fields_have_ordinal_augmentation_and_worker_invariance(tmp_path):
    dataset, capability = _population(tmp_path)
    aug_capability, profile = _appearance_configured()
    augmentation = _component(["noise-v1"], capability=aug_capability, profile=profile)
    epoch = _epoch(dataset, capability, augmentation=augmentation)
    locations = epoch.inputs.field_locations
    repeated = next(
        (i, j)
        for i in range(len(epoch))
        for j in range(i)
        if locations[i] == locations[j]
    )
    first, second = (epoch[index] for index in repeated)
    assert not np.array_equal(first.pixels, second.pixels)
    np.testing.assert_array_equal(
        first.labels.supervision_mask, second.labels.supervision_mask
    )
    collate = partial(
        collate_training_fields,
        preprocess=normalize99,
        field_profile=capability["default_field_profile"],
        class_count=3,
    )
    results = [
        list(
            DataLoader(
                epoch,
                batch_size=3,
                collate_fn=collate,
                num_workers=workers,
                multiprocessing_context="spawn" if workers else None,
            )
        )
        for workers in (0, 2)
    ]
    assert [batch.images.shape[0] for batch in results[0]] == [3, 1]
    for first, second in zip(*results, strict=True):
        for field in fields(first):
            assert torch.equal(getattr(first, field.name), getattr(second, field.name))


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_oversample_is_a_strict_optional_boolean(value):
    settings = _capability()["default_training_settings"]
    with pytest.raises(ValueError, match="oversample"):
        validate_training_settings({**settings, "oversample": value})


def test_capability_advertises_by_presence_not_boolean_value():
    capability = _capability()
    defaults = capability["default_training_settings"]
    for enabled in (False, True):
        settings = {**defaults, "oversample": enabled}
        with pytest.raises(ValueError, match="presence"):
            validate_training_settings_for_capability(settings, capability, 4096)
    defaults["oversample"] = False
    assert validate_model_training_capability(capability, 4096) == capability
    assert (
        validate_training_settings_for_capability(
            {**defaults, "oversample": True}, capability, 4096
        )["oversample"]
        is True
    )
    omitted = {key: value for key, value in defaults.items() if key != "oversample"}
    with pytest.raises(ValueError, match="presence"):
        validate_training_settings_for_capability(omitted, capability, 4096)
    for output in ("instance_segmentation", "dense_classification"):
        capability["output_type"] = output
        capability.pop("flow_loss_weighting_options", None)
        defaults.pop("flow_loss_weighting", None)
        with pytest.raises(ValueError, match="instance classification"):
            validate_model_training_capability(capability, 4096)


@pytest.mark.parametrize("enabled", [False, True])
def test_report_binds_the_exact_sampler_without_a_live_capability(enabled):
    report = queued_report(classified_training_model_output())
    report["trainer_settings"]["oversample"] = enabled
    report["trainer_configuration"]["sampler"] = instance_sampler_profile(enabled)
    assert validate_training_report(report, training_api_limits()) == report
    report["trainer_settings"]["oversample"] = not enabled
    with pytest.raises(ValueError, match="sampler"):
        validate_training_report(report, training_api_limits())
    unclassified = queued_report()
    unclassified["trainer_settings"]["oversample"] = enabled
    with pytest.raises(ValueError, match="instance classification"):
        validate_training_report(unclassified, training_api_limits())


@pytest.mark.parametrize("head", ["flow", "hovernet"])
def test_connected_training_keeps_validation_class_evidence_and_lazy_cache(
    tmp_path, monkeypatch, head, single_thread_torch
):
    setup = _setup(tmp_path, head=head)
    _, report, _, capability = setup
    capability["default_training_settings"]["oversample"] = False
    report["trainer_settings"]["oversample"] = True
    report["trainer_configuration"]["sampler"] = instance_sampler_profile(True)
    weighting = report["trainer_configuration"]["objective"]["parameters"][
        "class_weighting"
    ]
    weighting["channels"].reverse()
    before = copy.deepcopy(report)
    validate_training_report(report, training_api_limits())
    preparations = []
    score = sampling._field_sampling_probabilities

    def count(*args):
        preparations.append(args[2])
        return score(*args)

    monkeypatch.setattr(sampling, "_field_sampling_probabilities", count)
    with pytest.raises(TrainingCancelled):
        _execute(
            setup,
            lambda _: pytest.fail("cancelled"),
            lambda _: None,
            cancellation_requested=lambda: True,
        )
    assert preparations == []
    predictions, epochs = [], []

    def predict(image):
        predictions.append(image.pixels.shape)
        return {
            "instance_mask": np.zeros(image.pixels.shape[:2], np.int32),
            "votes_per_instance": {},
        }

    _execute(setup, predict, epochs.append)
    assert preparations == [
        {row["channel"]: row["raw_count"] for row in weighting["channels"]}
    ]
    assert len(predictions) == 6 and len(epochs) == 3
    assert report == before
    validate_training_report(
        {**report, "state": "RUNNING", "epochs": epochs}, training_api_limits()
    )
