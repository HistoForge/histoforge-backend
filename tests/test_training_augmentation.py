"""Accepted augmentation selections become reproducible executable run evidence."""

import copy
import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from histoforge.canonical_json import canonical_digest
from histoforge.cloud.training.augmentation import (
    validate_augmentation_profiles,
)
from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
from histoforge.cloud.training.contract import validate_training_augmentation_selection
from histoforge.cloud.training.implementations import DINOV3_INSTANCE_TRAINER_ID
from histoforge.cloud.training.trainers.instance.augmentation import (
    apply_training_augmentation,
)
from histoforge.cloud.training.trainers.instance.configuration import (
    resolve_instance_configuration,
)
from histoforge.cloud.training.trainers.instance.spatial_augmentation import (
    transform_training_field,
)
from tests.test_training_batches import _field
from tests.test_training_capabilities import _capability, _profile, _registry
from tests.test_training_dataset import training_api_limits
from tests.test_training_spatial_augmentation import _arrays


def _configured():
    """Synthetic parameters exercise execution, not qualified stain defaults."""

    capability, profile = _capability(), _profile()
    capability["augmentation_profiles"] = [
        {
            "profile_id": "spatial-fixture-v1",
            "display_name": "Spatial fixture",
            "description": "Test-only geometric transformations.",
            "operators": [
                {
                    "operator_id": operator_id,
                    "display_name": name,
                    "description": name,
                    "default_enabled": True,
                }
                for operator_id, name in (
                    ("rotate-v1", "Rotations"),
                    ("flip-v1", "Flips"),
                )
            ],
        }
    ]
    capability["default_augmentation_profile_id"] = "spatial-fixture-v1"
    profile["augmentation_profiles"] = {
        "spatial-fixture-v1": {
            "rotate-v1": {
                "implementation": "quarter-turn-v1",
                "quarter_turn_probabilities": [0.1, 0.2, 0.3, 0.4],
            },
            "flip-v1": {
                "implementation": "axis-flips-v1",
                "horizontal_probability": 0.35,
                "vertical_probability": 0.65,
            },
        }
    }
    return capability, profile


def _appearance_configured():
    """Add test-only appearance values without claiming a qualified H&E preset."""

    capability, profile = _configured()
    advertised = capability["augmentation_profiles"][-1]
    advertised["description"] = "Test-only spatial and appearance transforms."
    advertised["operators"].extend(
        {
            "operator_id": operator_id,
            "display_name": name,
            "description": name,
            "default_enabled": True,
        }
        for operator_id, name in (
            ("hed-v1", "HED variation"),
            ("brightness-contrast-v1", "Brightness and contrast"),
            ("blur-v1", "Blur"),
            ("noise-v1", "Noise"),
        )
    )
    profile["augmentation_profiles"]["spatial-fixture-v1"].update(
        {
            "hed-v1": {
                "implementation": "masked-hed-v1",
                "probability": 1.0,
                "sigma_range": 0.2,
                "mean_intensity_range": [0.0, 1.0],
            },
            "brightness-contrast-v1": {
                "implementation": "masked-brightness-contrast-v1",
                "probability": 1.0,
                "brightness_range": [0.9, 1.1],
                "contrast_range": [0.8, 1.2],
            },
            "blur-v1": {
                "implementation": "masked-gaussian-blur-v1",
                "probability": 1.0,
                "sigma_range": [0.5, 1.0],
            },
            "noise-v1": {
                "implementation": "masked-gaussian-noise-v1",
                "probability": 1.0,
                "sigma": 0.02,
            },
        }
    )
    return capability, profile


def _component(enabled=("rotate-v1", "flip-v1"), *, capability=None, profile=None):
    if capability is None:
        capability, profile = _configured()
    entry = _registry(
        TrainingCapabilityEntry(
            "dinov3-adapter-cellpose", capability, profile, DINOV3_INSTANCE_TRAINER_ID
        )
    ).entries()[0]
    selection = validate_training_augmentation_selection(
        {"profile_id": "spatial-fixture-v1", "enabled_operator_ids": list(enabled)},
        entry.capability,
        training_api_limits()["report"]["max_text_bytes"],
    )
    return resolve_instance_configuration(
        entry.trainer_profile,
        entry.capability,
        selection,
        {"type": "instance_segmentation", "classes": ["Cell"]},
        entry.capability["default_training_settings"],
    )["augmentation"]


def _apply(field, component, index=0, *, seed=1937, epoch=2):
    return apply_training_augmentation(
        field, component, training_seed=seed, epoch_index=epoch, sample_index=index
    )


def _assert_equal(left, right):
    for name, value in _arrays(left).items():
        np.testing.assert_array_equal(value, _arrays(right)[name])


@pytest.mark.parametrize("rotate,flip", list(product((False, True), repeat=2)))
def test_every_custom_subset_records_only_enabled_parameters_and_captured_names(
    rotate, flip
):
    capability, profile = _configured()
    enabled = [
        name for name, active in (("rotate-v1", rotate), ("flip-v1", flip)) if active
    ]
    component = _component(enabled, capability=capability, profile=profile)
    parameters = component["parameters"]
    assert parameters["base_profile_id"] == "spatial-fixture-v1"
    assert parameters["base_profile_display_name"] == "Spatial fixture"
    assert parameters["input_numeric_domain"] == "raw-srgb-hwc-finite-0-255-v1"
    assert parameters["appearance_numeric_policy"] == {
        "artificial_pixels": "preserve-input-values-v1",
        "clipping": "after-each-appearance-operator-v1",
        "output_dtype": "float32",
        "real_pixel_range": [0.0, 255.0],
    }
    assert parameters["default_enabled_operator_ids"] == ["rotate-v1", "flip-v1"]
    assert [item["operator_id"] for item in parameters["enabled_operators"]] == enabled
    assert set(parameters["operator_parameters"]) == set(enabled)
    for operator_id in enabled:
        assert (
            parameters["operator_parameters"][operator_id]
            == profile["augmentation_profiles"]["spatial-fixture-v1"][operator_id]
        )
    frozen = copy.deepcopy(component)
    profile["augmentation_profiles"]["spatial-fixture-v1"].clear()
    capability["augmentation_profiles"][-1]["display_name"] = "Changed later"
    assert component == frozen


def test_recorded_component_replays_without_registry_and_without_global_rng_changes():
    component = _component()
    restored = json.loads(json.dumps(component))
    field = _field()
    before_rng = np.random.get_state()
    expected = {index: _apply(field, component, index) for index in range(12)}
    for index in reversed(range(12)):
        _assert_equal(expected[index], _apply(field, restored, index))
    after_rng = np.random.get_state()
    for before, after in zip(before_rng, after_rng):
        np.testing.assert_array_equal(before, after)
    assert any(
        not np.array_equal(
            expected[index].pixels,
            _apply(field, component, index, epoch=3).pixels,
            equal_nan=True,
        )
        for index in expected
    )
    assert any(
        not np.array_equal(
            expected[index].pixels,
            _apply(field, component, index, seed=1938).pixels,
            equal_nan=True,
        )
        for index in expected
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_numeric_domain", "different-domain-v1"),
        (
            "appearance_numeric_policy",
            {
                "artificial_pixels": "preserve-input-values-v1",
                "clipping": "different-clipping-v1",
                "output_dtype": "float32",
                "real_pixel_range": [0.0, 255.0],
            },
        ),
    ],
)
def test_runtime_rejects_a_changed_numeric_contract(field, value):
    component = _component([])
    component["parameters"][field] = value
    with pytest.raises(ValueError, match="implementation is not supported"):
        _apply(_field(), component)


def test_random_draws_match_the_frozen_transform_sequence():
    """Pin seeding, categorical draws, and flip order independently of replay."""

    expected = [
        (2, False, True),
        (3, True, True),
        (1, False, True),
        (2, True, True),
        (3, False, True),
        (3, True, False),
        (1, False, True),
        (3, False, True),
        (0, True, True),
        (0, False, False),
        (2, True, False),
        (2, False, True),
    ]
    field, component = _field(), _component()
    for index, (turns, horizontal, vertical) in enumerate(expected):
        transformed = transform_training_field(
            field,
            quarter_turns=turns,
            flip_horizontal=horizontal,
            flip_vertical=vertical,
        )
        _assert_equal(_apply(field, component, index), transformed)


@pytest.mark.parametrize(
    "limit",
    [
        "max_component_parameters_bytes",
        "max_component_parameter_depth",
        "max_component_collection_entries",
    ],
)
def test_every_advertised_preset_fits_the_existing_report_parameter_bounds(limit):
    capability, profile = _configured()
    limits = training_api_limits()["report"]
    limits[limit] = 1
    with pytest.raises(ValueError, match="exceed"):
        validate_augmentation_profiles(
            profile["augmentation_profiles"], capability, limits
        )


def test_capability_resolution_does_not_import_tensor_or_training_libraries():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import histoforge.cloud.training.capabilities; "
            "from histoforge.cloud.training.profiles import "
            "full_coverage_sampler_profile; "
            "full_coverage_sampler_profile(); "
            "assert not {'torch', 'fastremap', 'scipy'} & sys.modules.keys()",
            str(Path(__file__).resolve().parents[1]),
        ],
        check=True,
        timeout=30,
    )


def test_disabling_another_operator_does_not_shift_flips_random_draws():
    capability, profile = _configured()
    profile["augmentation_profiles"]["spatial-fixture-v1"]["rotate-v1"][
        "quarter_turn_probabilities"
    ] = [1.0, 0.0, 0.0, 0.0]
    full = _component(capability=capability, profile=profile)
    flips = _component(("flip-v1",), capability=capability, profile=profile)
    field = _field()
    for index in range(16):
        _assert_equal(_apply(field, full, index), _apply(field, flips, index))


def test_execution_follows_the_captured_operator_order():
    capability, profile = _configured()
    capability["augmentation_profiles"][-1]["operators"].reverse()
    parameters = profile["augmentation_profiles"]["spatial-fixture-v1"]
    parameters["rotate-v1"]["quarter_turn_probabilities"] = [0, 1, 0, 0]
    parameters["flip-v1"].update(horizontal_probability=1, vertical_probability=0)
    component = _component(
        ("flip-v1", "rotate-v1"), capability=capability, profile=profile
    )
    field = _field()
    expected = transform_training_field(
        transform_training_field(field, flip_horizontal=True), quarter_turns=1
    )
    _assert_equal(_apply(field, component), expected)


@pytest.mark.parametrize("enabled", [[], ["flip-v1", "rotate-v1"]])
def test_selection_still_uses_the_existing_capability_order_contract(enabled):
    if not enabled:
        field = _field()
        actual = _apply(field, _component(enabled))
        _assert_equal(actual, field)
        assert all(
            not np.shares_memory(value, _arrays(field)[name])
            for name, value in _arrays(actual).items()
        )
    else:
        with pytest.raises(ValueError, match="follow profile order"):
            _component(enabled)


@pytest.mark.parametrize(
    "change",
    [
        lambda values: values.pop("spatial-fixture-v1"),
        lambda values: values["spatial-fixture-v1"].pop("flip-v1"),
        lambda values: values["spatial-fixture-v1"]["flip-v1"].update(
            implementation="missing-library-v1"
        ),
        lambda values: values["spatial-fixture-v1"]["flip-v1"].update(
            horizontal_probability=-0.1
        ),
        lambda values: values["spatial-fixture-v1"]["flip-v1"].update(
            horizontal_probability=True
        ),
        lambda values: values["spatial-fixture-v1"]["flip-v1"].update(
            vertical_probability=float("nan")
        ),
        lambda values: values["spatial-fixture-v1"]["flip-v1"].update(unknown=1),
        lambda values: values["spatial-fixture-v1"]["rotate-v1"].update(
            quarter_turn_probabilities=[0.1, 0.1, 0.1, 0.1]
        ),
        lambda values: values["spatial-fixture-v1"]["rotate-v1"].update(
            quarter_turn_probabilities=[0, 0, 1]
        ),
    ],
)
def test_bad_or_unimplemented_curated_operator_is_rejected_before_advertisement(change):
    capability, profile = _configured()
    change(profile["augmentation_profiles"])
    with pytest.raises(ValueError):
        _registry(
            TrainingCapabilityEntry(
                "dinov3-adapter-cellpose",
                capability,
                profile,
                DINOV3_INSTANCE_TRAINER_ID,
            )
        )


class _AugmentedFields:
    """A picklable ordinary DataLoader dataset, with no worker-owned RNG state."""

    def __init__(self, component):
        self.component = component
        self.fields = [_field(16, 32), _field(32, 16)]

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return _apply(self.fields[index % 2], self.component, index)


def test_spawned_data_loading_workers_produce_the_same_ordered_fields():
    from torch.utils.data import DataLoader

    dataset = _AugmentedFields(_component())
    expected = list(DataLoader(dataset, batch_size=None, num_workers=0))
    actual = list(
        DataLoader(
            dataset,
            batch_size=None,
            num_workers=2,
            multiprocessing_context="spawn",
            timeout=30,
        )
    )
    assert len(actual) == len(expected) == 8
    for left, right in zip(actual, expected):
        _assert_equal(left, right)


def test_appearance_operators_replay_exactly_and_change_only_real_rgb():
    capability, profile = _appearance_configured()
    enabled = [
        "hed-v1",
        "brightness-contrast-v1",
        "blur-v1",
        "noise-v1",
    ]
    component = _component(enabled, capability=capability, profile=profile)
    field = _field(19, 27)
    before = copy.deepcopy(field)

    actual = _apply(field, component)
    repeated = _apply(field, json.loads(json.dumps(component)))

    np.testing.assert_array_equal(actual.pixels, repeated.pixels)
    real = field.labels.real_pixel_mask
    assert actual.pixels.dtype == np.float32
    assert np.isfinite(actual.pixels[real]).all()
    assert actual.pixels[real].min() >= 0 and actual.pixels[real].max() <= 255
    assert np.any(actual.pixels[real] != field.pixels[real])
    np.testing.assert_array_equal(actual.pixels[~real], field.pixels[~real])
    for name, value in _arrays(actual).items():
        if name != "pixels":
            np.testing.assert_array_equal(value, _arrays(before)[name])
        assert not np.shares_memory(value, _arrays(field)[name])
    _assert_equal(field, before)


@pytest.mark.parametrize(
    "operator_id,function_name",
    [
        ("hed-v1", "augment_hed"),
        ("brightness-contrast-v1", "augment_brightness_contrast"),
        ("blur-v1", "augment_blur"),
        ("noise-v1", "augment_noise"),
    ],
)
def test_appearance_dispatch_matches_the_vendored_primitive(operator_id, function_name):
    from histoforge._vendor.histoforge_models.pathology import managed_training

    capability, profile = _appearance_configured()
    component = _component([operator_id], capability=capability, profile=profile)
    field = _field(19, 27)
    settings = component["parameters"]["operator_parameters"][operator_id]
    seed = int(
        canonical_digest(
            {
                "domain": component["parameters"]["seed_derivation"],
                "training_seed": 1937,
                "epoch_index": 2,
                "sample_index": 0,
                "operator_id": operator_id,
            }
        ),
        16,
    )
    kwargs = {key: value for key, value in settings.items() if key != "implementation"}
    expected = getattr(managed_training, function_name)(
        field.pixels,
        real_pixel_mask=field.labels.real_pixel_mask,
        rng=np.random.Generator(np.random.PCG64(seed)),
        **kwargs,
    )
    np.testing.assert_array_equal(_apply(field, component).pixels, expected)


@pytest.mark.parametrize(
    "operator_id,field,value",
    [
        ("hed-v1", "probability", 1.1),
        ("hed-v1", "sigma_range", -0.1),
        ("hed-v1", "mean_intensity_range", [0.5, 0.5]),
        ("brightness-contrast-v1", "brightness_range", [-0.1, 1.0]),
        ("brightness-contrast-v1", "contrast_range", [1.0, float("inf")]),
        ("blur-v1", "sigma_range", [2.0, 1.0]),
        ("noise-v1", "sigma", 1.1),
        ("noise-v1", "unknown", 1.0),
    ],
)
def test_bad_appearance_parameters_fail_before_advertisement(operator_id, field, value):
    capability, profile = _appearance_configured()
    profile["augmentation_profiles"]["spatial-fixture-v1"][operator_id][field] = value
    with pytest.raises(ValueError):
        _registry(
            TrainingCapabilityEntry(
                "dinov3-adapter-cellpose",
                capability,
                profile,
                DINOV3_INSTANCE_TRAINER_ID,
            )
        )
