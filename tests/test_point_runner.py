"""Finite detector contract through the shared decoder, wire result and limits.

All pixels and tensors are synthetic. No pretrained model, source slide, Modal
allocation or training run is used. Native point persistence has separate tests.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology import common, dinov3
from histoforge._vendor.histoforge_models.pathology.point_inference import (
    PointDetectionLimitExceeded,
    run_point_inference,
)
from histoforge.cloud.architectures.point.net import build
from histoforge.cloud.architectures.point.runner import PointRunner
from histoforge.cloud.architectures.point.runtime import (
    PointModelRuntime,
    create_point_runtime,
)
from histoforge.cloud.architectures.registry import (
    model_runner_availability,
    runner_network_precision,
    runner_registry,
)
from histoforge.cloud.inference import execution
from histoforge.cloud.inference.contract import validate_inference_features
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_point_feature_records,
)
from histoforge.cloud.inference.finite.contract import validate_inference_result
from histoforge.cloud.inference.finite.runtime import (
    encode_finite_result,
    execute_finite_request,
)
from histoforge.cloud.modules.schema import parse_module_recipe
from histoforge.cloud.rasterization.inputs import build_model_raster_input
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.training.capabilities import model_training_availability
from safetensors.torch import load_file, save_file
from tests.test_finite_runtime import _admission_values, _SourceStore
from tests.test_variable_field_inference import _profile
from torch import nn


def _recipe():
    return {
        "schemaVersion": 4,
        "name": "Synthetic center detector",
        "models": {
            "centers": {
                "arch": "dinov3-adapter-point",
                "gpu": "L40S",
                "weights": {"type": "vendored", "file": "centers.safetensors"},
                "input": {
                    "type": "field",
                    "micronsPerPixel": 0.25,
                    "fieldProfile": _profile(),
                },
                "output": {
                    "type": "point_detection",
                    "classChannels": [
                        {"channel": 0, "role": "emitted", "pathClass": "Object"}
                    ],
                },
                "params": {
                    "num_classes": 1,
                    "batch_size": 2,
                    "score_threshold": 1.0,
                    "peak_radius_model_px": 1,
                },
            }
        },
        "steps": [{"model": "centers"}],
    }


class _CenterNet(torch.nn.Module):
    """A known RGB-to-logit mapping through real normalization and field blending."""

    head_type = "point_detection"
    nclasses = 1

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.batches = []

    @property
    def dtype(self):
        return self.anchor.dtype

    @property
    def device(self):
        return self.anchor.device

    def to(self, device):
        assert device == "cuda"
        return self

    def forward(self, batch):
        self.batches.append(tuple(batch.shape))
        return 20 * batch[:, :1] - 10, batch.new_zeros((len(batch), 256))


def _image():
    pixels = np.zeros((17, 131, 3), np.uint8)
    pixels[..., 0] = np.linspace(0, 100, 131, dtype=np.uint8)
    pixels[7, 30, 0] = pixels[10, 100, 0] = pixels[0, 0, 0] = 255
    real = np.ones(pixels.shape[:2], bool)
    real[0, 0] = False
    return RasterizedImage(pixels, real)


def _runner():
    runner = PointRunner._get_user_cls()()
    runner.runtime = create_point_runtime(_recipe()["models"]["centers"], _CenterNet())
    return runner


def _worker(monkeypatch, image):
    request, backend, _, source, files = _admission_values()
    capabilities = runner_registry()["definitions"]["dinov3-adapter-point"][
        "capabilities"
    ]
    backend["architecture"] = "dinov3-adapter-point"
    for key in ["runner_build", "rasterizer", "preprocessing_contract_digest"]:
        backend[key] = capabilities[key]
    request["expected_backend"] = backend
    old = request["raster_input"]
    request["raster_input"] = build_model_raster_input(
        old["source_view"], old["frame"], capabilities["rasterizer"]
    )
    activation = {
        "model": _recipe()["models"]["centers"],
        "backend": backend,
        "capabilities": capabilities,
    }
    monkeypatch.setattr(execution, "rasterize_model_input", lambda *_a, **_k: image)
    return request, activation, _SourceStore(source), files


def test_finite_point_runner_matches_the_shared_decoder_on_identical_grid():
    runner, image = _runner(), _image()
    model = _recipe()["models"]["centers"]
    before = copy.deepcopy(model)
    expected = run_point_inference(
        runner.runtime.network,
        image.pixels,
        field_profile=model["input"]["fieldProfile"],
        score_threshold=1.0,
        peak_radius_model_px=1,
        max_detections=100,
        batch_size=2,
        real_pixel_mask=image.real_pixel_mask,
    )
    np.testing.assert_array_equal(expected["xy"], [[30.5, 7.5], [100.5, 10.5]])
    budget = dict.fromkeys(
        [
            "max_features",
            "max_vertices",
            "max_record_bytes",
            "max_transferred_bytes",
            "max_decoded_bytes",
        ],
        1_000_000,
    )
    actual = runner._infer_features(image, budget)
    assert (
        actual
        == build_point_feature_records(
            [
                ("Object", xy, score)
                for xy, score in zip(expected["xy"], expected["scores"], strict=True)
            ],
            ["Object"],
            budget,
        ).as_collection()
    )
    assert model == before
    assert len(runner.runtime.network.batches) > 1


@pytest.mark.parametrize("empty", [False, True])
def test_worker_result_and_addressed_wire_validation_preserve_points(
    monkeypatch, empty
):
    image = _image()
    if empty:
        image.pixels.fill(0)
    request, activation, source, files = _worker(monkeypatch, image)
    outcome = execute_finite_request(
        request, activation, source, files, _runner()._infer_features
    )
    assert outcome["type"] == "result"
    wire = json.loads(encode_finite_result(outcome, "point-call"))
    actual = validate_inference_result(
        wire, request, "point-call", ["Object"], geometry_kind="point"
    )
    assert len(actual["geojson"]["features"]) == (0 if empty else 2)
    if not empty:
        with pytest.raises(ValueError):
            validate_inference_result(wire, request, "point-call", ["Object"])


@pytest.mark.parametrize("field", ["max_features", "max_vertices", "max_record_bytes"])
def test_complete_point_population_limit_failure_is_typed(monkeypatch, field):
    request, activation, source, files = _worker(monkeypatch, _image())
    request["result_budget"][field] = 1
    outcome = execute_finite_request(
        request, activation, source, files, _runner()._infer_features
    )
    assert outcome["type"] == "failure"
    assert outcome["error"]["code"] == "RESULT_LIMIT_EXCEEDED"
    assert "geojson" not in outcome


def test_invalid_point_output_is_not_misreported_as_a_result_limit(monkeypatch):
    request, activation, source, files = _worker(monkeypatch, _image())

    def malformed(*_args):
        raise ValueError("nonfinite model output")

    outcome = execute_finite_request(request, activation, source, files, malformed)
    assert outcome["error"]["code"] == "RUNNER_FAILED"


@pytest.mark.parametrize("warmup_overflow", [True, False])
def test_warmup_density_does_not_own_actual_result_admission(
    monkeypatch, warmup_overflow
):
    model = _recipe()["models"]["centers"]
    capabilities = model_runner_availability(model)["capabilities"]
    activation = {"network": _CenterNet(), "model": model, "capabilities": capabilities}
    monkeypatch.setattr(
        "histoforge.cloud.inference.activation.activate_model", lambda *_a: activation
    )
    predict = PointModelRuntime.predict
    calls = []

    def warmup(runtime, image, *, max_detections):
        calls.append(max_detections)
        if len(calls) == 1:
            if warmup_overflow:
                raise PointDetectionLimitExceeded("point result exceeds max_detections")
            raise ValueError("nonfinite logits")
        return predict(runtime, image, max_detections=max_detections)

    monkeypatch.setattr(PointModelRuntime, "predict", warmup)
    runner_type = PointRunner._get_user_cls()
    runner = runner_type()
    runner.ref, runner.model_id = "owner/module", "centers"
    runner.content_digest, runner.resolved_commit_sha = "a" * 64, "b" * 40
    runner.gpu, runner.inference_precision = "L40S", "default"
    start = runner_type.__dict__["load_model"]._get_raw_f()
    if not warmup_overflow:
        from histoforge.cloud._errors import GatewayFailure

        with pytest.raises(GatewayFailure) as failure:
            start(runner)
        assert failure.value.error["code"] == "RUNNER_FAILED"
        return
    start(runner)
    assert runner.activation is activation
    budget = dict.fromkeys(
        [
            "max_features",
            "max_vertices",
            "max_record_bytes",
            "max_transferred_bytes",
            "max_decoded_bytes",
        ],
        1,
    )
    with pytest.raises(ResultLimitExceeded):
        runner._infer_features(_image(), budget)


def test_schema_4_detector_cannot_be_mislabeled_as_a_schema_3_module():
    recipe = _recipe()
    assert parse_module_recipe(json.dumps(recipe)) == recipe
    recipe["schemaVersion"] = 3
    with pytest.raises(ValueError, match="schemaVersion 4"):
        parse_module_recipe(json.dumps(recipe))
    recipe["schemaVersion"] = 4
    for role in ["background", "training_only"]:
        channel = recipe["models"]["centers"]["output"]["classChannels"][0]
        channel["role"] = role
        if role == "background":
            channel.pop("pathClass")
        else:
            channel["pathClass"] = "Object"
        with pytest.raises(ValueError):
            parse_module_recipe(json.dumps(recipe))


@pytest.mark.parametrize(
    "damage", ["profile", "crop", "count", "score", "radius", "extra", "large_region"]
)
def test_detector_requires_every_consumed_model_setting(damage):
    model = _recipe()["models"]["centers"]
    if damage == "profile":
        model["input"].pop("fieldProfile")
    elif damage == "crop":
        model["input"].update(type="crop", framing={"paddingModelPx": 0})
    elif damage == "count":
        model["params"]["num_classes"] = 2
    elif damage == "score":
        model["params"].pop("score_threshold")
    elif damage == "radius":
        model["params"]["peak_radius_model_px"] = 0
    elif damage == "extra":
        model["params"]["min_instance_pixels"] = 0
    else:
        model["input"]["largeRegion"] = {
            "coreWidthModelPx": 100,
            "coreHeightModelPx": 100,
            "haloPerSideModelPx": 0,
        }
    assert model_runner_availability(model)["code"] == "UNSUPPORTED_MODEL_CONFIGURATION"


def test_point_capabilities_leave_unqualified_paths_unavailable():
    recipe = _recipe()
    availability = model_runner_availability(recipe["models"]["centers"])
    assert availability["type"] == "available"
    assert availability["capabilities"]["modes"] == ["FINITE_CALL"]
    assert availability["capabilities"]["output_types"] == ["POINT_DETECTION"]
    with pytest.raises(ValueError, match="unsupported inference precision"):
        runner_network_precision("dinov3-adapter-point", "bf16")
    assert (
        model_training_availability(recipe, "centers", availability)["type"]
        == "unavailable"
    )


def test_previous_finite_envelopes_require_a_matching_client_update():
    fixture = Path(__file__).parent / "fixtures/contracts/finite-inference-v1.json"
    case = json.loads(fixture.read_text())["cases"][0]
    with pytest.raises(ValueError, match="schema_version"):
        validate_inference_result(
            json.loads(case["resultCanonical"]),
            json.loads(case["requestCanonical"]),
            json.loads(case["acceptedCanonical"])["backend_call_id"],
            case["declaredClasses"],
        )


def test_shared_finite_point_fixture_has_exact_recipe_results_and_scores():
    from histoforge.canonical_json import canonical_bytes, sha256_bytes

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/contracts/point-finite-inference-v1.json"
        ).read_text()
    )
    assert sha256_bytes(fixture["recipeCanonical"].encode()) == fixture["recipeSha256"]
    parse_module_recipe(fixture["recipeCanonical"])
    for case in fixture["cases"]:
        result = validate_inference_result(
            json.loads(case["resultCanonical"]),
            json.loads(case["requestCanonical"]),
            json.loads(case["acceptedCanonical"])["backend_call_id"],
            case["declaredClasses"],
            geometry_kind="point",
        )
        assert canonical_bytes(result).decode() == case["resultCanonical"]
        assert sha256_bytes(canonical_bytes(result)) == case["resultSha256"]


@pytest.mark.parametrize("xy", [(-0.01, 0), (512, 0), (0, 384)])
def test_point_raster_bounds_are_half_open_before_filtering(xy):
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/contracts/point-finite-inference-v1.json"
        ).read_text()
    )
    request = json.loads(fixture["cases"][0]["requestCanonical"])
    budget = request["result_budget"]
    points = build_point_feature_records(
        [("Object", xy, 0.5)], ["Object"], budget
    ).as_collection()
    with pytest.raises(ValueError, match="outside"):
        validate_inference_features(
            points,
            request["raster_input"]["frame"]["transform"]["raster"],
            budget,
            ["Object"],
            geometry_kind="point",
        )


def test_recipe_factory_strictly_reloads_real_point_adapter_tensors(
    monkeypatch, tmp_path
):
    class TokenBlock(nn.Module):
        def forward(self, value, rope=None):
            return value

    class PatchEmbedding(nn.Conv2d):
        def __init__(self):
            super().__init__(3, 1024, kernel_size=16, stride=16)

        def forward(self, value):
            return super().forward(value).permute(0, 2, 3, 1)

    class LocalTrunk(nn.Module):
        num_prefix_tokens = 5

        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbedding()
            self.blocks = nn.ModuleList([TokenBlock() for _ in range(24)])
            self.norm_pre = nn.Identity()

        def _pos_embed(self, value):
            tokens = value.reshape(value.shape[0], -1, 1024)
            return torch.cat(
                (value.new_zeros((len(value), 5, 1024)), tokens), dim=1
            ), None

    def local_trunk(name, *, pretrained, num_classes):
        assert name == dinov3.DINOV3_MODEL and pretrained is False and num_classes == 0
        return LocalTrunk()

    monkeypatch.setattr(dinov3.timm, "create_model", local_trunk)
    network = common.build({"head_type": "point_detection", "num_classes": 1})
    with torch.no_grad():
        network.out_class.weight.zero_()
        network.out_class.bias.zero_()
    checkpoint = tmp_path / "centers.safetensors"
    save_file(network.state_dict(), checkpoint)
    state = load_file(checkpoint)
    params = {**_recipe()["models"]["centers"]["params"], "score_threshold": 0.5}
    restored = build(params, state).eval()
    for name, tensor in state.items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    model = _recipe()["models"]["centers"]
    model["params"] = params
    result = create_point_runtime(model, restored).predict(_image(), max_detections=100)
    # (0, 0) is unavailable in this source; its first supported pixel is (1, 0).
    np.testing.assert_array_equal(result["xy"], [[1.5, 0.5]])
    np.testing.assert_array_equal(result["scores"], [0.5])
    for damage in ["flow", "classes", "dtype"]:
        changed = dict(state)
        config = dict(params)
        if damage == "flow":
            changed["head_type_code"] = torch.tensor(
                0, dtype=state["head_type_code"].dtype
            )
        elif damage == "classes":
            config["num_classes"] = 2
        else:
            changed["out_class.weight"] = changed["out_class.weight"].bfloat16()
        with pytest.raises(common.StateDictIdentityError):
            build(config, changed)
