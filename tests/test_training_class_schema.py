"""Source/target contracts and real adapter readout adaptation, without FM downloads."""

import copy
from types import SimpleNamespace

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology import common
from histoforge.cloud.architectures.dinov3_training import (
    adapt_dinov3_training_network,
    project_dinov3_training_model,
)
from histoforge.cloud.training.class_schema import validate_training_class_transition
from tests.test_training_capabilities import _capability, _model
from tests.test_training_dataset_content import _dataset
from tests.test_training_run_preparation import (
    RUN_ID,
    STARTING_WEIGHT_DIGEST,
    _preparation,
    _recipe,
    _request,
)
from tests.training_class_fixtures import TinyClassAdapter, initialized_adapter


def _classified():
    return {
        "type": "instance_segmentation_and_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "emitted", "pathClass": "Cell"},
            {"channel": 2, "role": "training_only", "pathClass": "Other"},
        ],
    }


@pytest.mark.parametrize("mode", ["use", "add", "define"])
def test_projection_is_detached_and_preserves_every_non_target_setting(mode):
    source = _model()
    source["output"] = _classified()
    source["params"]["num_classes"] = 3
    original = copy.deepcopy(source)
    output = _classified()
    output["classChannels"][1]["pathClass"] = "Renamed cell"
    if mode == "add":
        output["classChannels"].append(
            {"channel": 3, "role": "emitted", "pathClass": "New cell"}
        )
    task = {"model_output": output, "model_microns_per_pixel": 0.37}
    target = project_dinov3_training_model(source, task, mode)
    expected = copy.deepcopy(source)
    expected["output"] = output
    expected["params"]["num_classes"] = len(output["classChannels"])
    expected["input"]["micronsPerPixel"] = 0.37
    assert target == expected
    target["params"]["num_classes"] = 99
    assert source == original


@pytest.mark.parametrize("mode", ["use", "add"])
def test_preserving_modes_cannot_drop_training_only_or_change_channel_roles(mode):
    target = _classified()
    if mode == "add":
        target["classChannels"].append(
            {"channel": 3, "role": "emitted", "pathClass": "New cell"}
        )
    target["classChannels"][2]["role"] = "emitted"
    with pytest.raises(ValueError, match="identities and roles"):
        validate_training_class_transition(_classified(), target, mode)
    with pytest.raises(ValueError, match="count|prefix"):
        validate_training_class_transition(_classified(), _classified(), "add")
    assert validate_training_class_transition(_classified(), target, "define") == target


@pytest.mark.parametrize("mode", ["add", "define"])
def test_segmentation_base_with_one_foreground_class_uses_classified_policy(
    tmp_path, mode
):
    capability = _capability()
    capability["output_type"] = "instance_segmentation_and_classification"
    output = _classified()
    output["classChannels"] = output["classChannels"][:2]
    dataset = _dataset(tmp_path, objects=[], capability=capability, model_output=output)
    record = {
        "type": "retained",
        "dataset_digest": dataset.manifest["dataset_digest"],
        "retained_bytes": dataset.retained_bytes,
    }
    files = SimpleNamespace(read_dataset=lambda *_: dataset)
    request = _request(record["dataset_digest"], capability)
    request["class_initialization"] = mode
    preparation = _preparation(files, record, capability, [])
    report = preparation(request, _recipe(), STARTING_WEIGHT_DIGEST).initial_report(
        RUN_ID
    )
    assert report["source_model_output"]["type"] == "instance_segmentation"
    assert report["task"]["model_output"] == output
    objective = report["trainer_configuration"]["objective"]["parameters"]
    assert objective["class_count"] == 2
    assert objective["use_uncertainty_weighting"] is True
    assert objective["classification_objective"] == "cross_entropy+tversky"
    assert (
        report["metric_policy"]["primary_validation_series_id"]
        == "validation-object-macro-f1.v1"
    )
    with pytest.raises(ValueError, match="target task"):
        preparation(
            {**request, "class_initialization": "use"},
            _recipe(),
            STARTING_WEIGHT_DIGEST,
        )


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
@pytest.mark.parametrize("mode", ["use", "add", "define"])
def test_adaptation_preserves_encoder_segmentation_and_reconstructs_exactly(
    monkeypatch, head_type, mode
):
    monkeypatch.setattr(common, "DinoV3AdapterBackbone", TinyClassAdapter)
    torch.manual_seed(13)
    original = initialized_adapter(3, head_type=head_type)
    with torch.no_grad():
        original.out_class.weight.add_(0.3)
        original.W3.add_(0.01)  # Valid nonidentity checkpoint readout must survive Add.
    state = copy.deepcopy(original.state_dict())
    loaded = common.build_and_load({"head_type": head_type, "num_classes": 3}, state)
    count = 4 if mode == "add" else 3
    adapt_dinov3_training_network(loaded, {"params": {"num_classes": count}}, mode, 19)
    for name, tensor in state.items():
        if not name.startswith("out_class.") and name != "W3":
            torch.testing.assert_close(
                loaded.state_dict()[name], tensor, rtol=0, atol=0
            )
    if mode in {"use", "add"}:
        torch.testing.assert_close(
            loaded.out_class.weight[:48], original.out_class.weight, rtol=0, atol=0
        )
        torch.testing.assert_close(loaded.W3[:48, :3], original.W3, rtol=0, atol=0)
    else:
        assert not torch.equal(loaded.out_class.weight, original.out_class.weight)
    fresh = common.build_and_load(
        {"head_type": head_type, "num_classes": count}, loaded.state_dict()
    )
    image = torch.randn(1, 3, 16, 32)
    torch.testing.assert_close(loaded(image)[0], fresh(image)[0], rtol=0, atol=0)
    with pytest.raises(common.StateDictIdentityError):
        common.build_and_load(
            {"head_type": head_type, "num_classes": count + 1}, loaded.state_dict()
        )
