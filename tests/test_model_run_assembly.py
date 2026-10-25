"""CPU assembly preserves exact activation, output, and lifecycle boundaries."""

import copy
import hashlib
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from histoforge.cloud._app import app
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures import registry
from histoforge.cloud.artifacts import store as artifact_store
from histoforge.cloud.inference import activation as model_activation
from histoforge.cloud.inference.runs import assembly, executor
from histoforge.cloud.inference.runs.core_halo_runtime import finalize_core_halo_input
from histoforge.cloud.modules.current import CurrentModules
from tests.test_core_halo_runtime import _publish_units, _values
from tests.test_model_run_executor import _context, _core_halo_plan, _Lifecycle


@pytest.fixture
def installed_case(tmp_path, monkeypatch):
    request, activation, limits, _plan = _values()
    backend = activation["backend"]
    module = backend["module"]
    model = copy.deepcopy(activation["model"])
    model["weights"] = {"type": "vendored", "file": "model.safetensors"}
    activation = {**activation, "model": model}
    directory = tmp_path / "modules" / module["module_ref"]
    directory.mkdir(parents=True)
    weights = directory / "model.safetensors"
    # Assembly verifies these bytes but must not parse them as tensors.
    content = b"exact checkpoint bytes; deliberately not a tensor file"
    weights.write_bytes(content)
    record = {
        **module,
        "resolved_commit_sha": module.get("resolved_commit_sha"),
        "resolved_artifacts": [
            {
                "model_id": backend["model_id"],
                "role": "weights",
                "artifact_source_type": "vendored",
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }

    class Current(CurrentModules):
        def read_record(self, ref):
            assert ref == module["module_ref"]
            return record

        def load_recipe(self, ref, *, record=None):
            assert ref == module["module_ref"]
            return {"models": {backend["model_id"]: model}}

    current = Current(tmp_path / "modules")
    monkeypatch.setattr(
        "histoforge.cloud.modules.runtime.current_module_snapshot",
        lambda: nullcontext(current),
    )
    monkeypatch.setattr(
        model_activation,
        "installed_model_descriptor",
        lambda *_args: {
            "runner_availability": {
                "type": "available",
                "capabilities": activation["capabilities"],
            }
        },
    )
    monkeypatch.setattr(
        model_activation, "installed_backend_identity", lambda *_args: backend
    )
    files = artifact_store.ArtifactFileStore(tmp_path / "artifacts")
    monkeypatch.setattr(artifact_store, "ArtifactFileStore", lambda *_a, **_k: files)
    return request, activation, limits, current, weights, files


def _resolve(case, **changes):
    request, _activation, _limits, current, _weights, _files = case
    backend = request["expected_backend"]
    module = backend["module"]
    arguments = {
        "module_ref": module["module_ref"],
        "model_id": backend["model_id"],
        "content_digest": module["content_digest"],
        "resolved_commit_sha": module.get("resolved_commit_sha"),
        "architecture": backend["architecture"],
        "inference_gpu": backend["gpu"],
        **changes,
    }
    return model_activation.resolve_installed_model_activation(current, **arguments)


def test_metadata_activation_verifies_exact_bytes_without_tensor_loading(
    installed_case, monkeypatch
):
    import safetensors.torch

    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *_a: pytest.fail("loaded tensors")
    )
    activation, weights = _resolve(installed_case)
    assert activation == installed_case[1]
    assert weights == installed_case[4]
    assert set(activation) == {"model", "backend", "capabilities"}


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"content_digest": "0" * 64}, "MODULE_VERSION_UNAVAILABLE"),
        ({"resolved_commit_sha": "0" * 40}, "MODULE_VERSION_UNAVAILABLE"),
        ({"model_id": "missing"}, "MODULE_IDENTITY_MISMATCH"),
        ({"architecture": "different"}, "MODULE_IDENTITY_MISMATCH"),
    ],
)
def test_metadata_activation_rejects_stale_or_wrong_identity(
    installed_case, changes, code
):
    with pytest.raises(GatewayFailure) as failure:
        _resolve(installed_case, **changes)
    assert failure.value.error["code"] == code


@pytest.mark.parametrize("same_length", [False, True])
def test_metadata_activation_rejects_actual_corrupt_checkpoint_bytes(
    installed_case, same_length
):
    weights = installed_case[4]
    size = weights.stat().st_size
    weights.write_bytes(b"x" * (size if same_length else size + 1))
    with pytest.raises(GatewayFailure) as failure:
        _resolve(installed_case)
    assert failure.value.error["code"] == "MODULE_IDENTITY_MISMATCH"


def test_cpu_entry_assembles_identical_bytes_without_loading_tensors(
    installed_case, monkeypatch
):
    import safetensors.torch

    request, activation, limits, _current, _weights, files = installed_case
    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda *_a: pytest.fail("loaded tensors")
    )
    publications = _publish_units(request, limits, files)
    baseline = finalize_core_halo_input(
        request, 0, limits, publications, activation, files
    )
    actual = assembly.assemble_model_run_input.local(request, 0, limits, publications)
    before = baseline["input"]["feature_stream_artifact"]
    after = actual["input"]["feature_stream_artifact"]
    before_id, after_id = before.pop("artifact_id"), after.pop("artifact_id")
    assert actual == baseline
    assert files.read_artifact_block(before_id, 0, before["size_bytes"]) == (
        files.read_artifact_block(after_id, 0, after["size_bytes"])
    )
    for item in publications:
        reference = item["reusable_state"]
        assert files.load_artifact_ref(reference["artifact_id"]) == reference


def test_cpu_failure_preserves_every_completed_core(installed_case):
    request, _activation, limits, _current, _weights, files = installed_case
    publications = _publish_units(request, limits, files)
    request["result_budget"]["max_features"] = 1
    with pytest.raises(GatewayFailure) as failure:
        assembly.assemble_model_run_input.local(request, 0, limits, publications)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    for item in publications:
        reference = item["reusable_state"]
        assert files.load_artifact_ref(reference["artifact_id"]) == reference


def test_assembly_finishes_module_reads_before_result_assembly(
    installed_case, monkeypatch
):
    request, activation, limits, current, _weights, _files = installed_case
    active = False

    @contextmanager
    def snapshot():
        nonlocal active
        active = True
        try:
            yield current
        finally:
            active = False

    def resolve(view, *_args):
        assert active and view is current
        return activation, _weights

    def finalize(*args, **kwargs):
        assert not active
        return {"result": "detached"}

    monkeypatch.setattr(
        "histoforge.cloud.modules.runtime.current_module_snapshot", snapshot
    )
    monkeypatch.setattr(model_activation, "resolve_installed_model_activation", resolve)
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime.finalize_core_halo_input",
        finalize,
    )
    assert assembly.assemble_model_run_input.local(request, 0, limits, []) == {
        "result": "detached"
    }


def test_cpu_entry_rejects_wrong_backend_before_reading_core_bytes(
    installed_case, monkeypatch
):
    request, _activation, limits, _current, _weights, files = installed_case
    request = copy.deepcopy(request)
    request["expected_backend"]["model_config_digest"] = "0" * 64
    monkeypatch.setattr(
        files, "read_artifact_block", lambda *_a: pytest.fail("read core bytes")
    )
    with pytest.raises(GatewayFailure) as failure:
        assembly.assemble_model_run_input.local(request, 0, limits, [])
    assert failure.value.error["code"] == "MODULE_IDENTITY_MISMATCH"


@pytest.mark.parametrize(
    "error,code",
    [
        (FileNotFoundError("missing module"), "MODULE_VERSION_UNAVAILABLE"),
        (OSError("storage unavailable"), "SERVICE_UNAVAILABLE"),
        (RuntimeError("private detail"), "RUNNER_FAILED"),
        (
            GatewayFailure("UNSUPPORTED_CAPABILITY", "unavailable"),
            "UNSUPPORTED_CAPABILITY",
        ),
    ],
)
def test_cpu_entry_maps_activation_failures(installed_case, monkeypatch, error, code):
    def fail(*_args):
        raise error

    monkeypatch.setattr(model_activation, "resolve_installed_model_activation", fail)
    request, _activation, limits, *_rest = installed_case
    with pytest.raises(GatewayFailure) as failure:
        assembly.assemble_model_run_input.local(request, 0, limits, [])
    assert failure.value.error["code"] == code
    assert "private detail" not in failure.value.error["message"]


def test_assembly_resources_do_not_extend_prediction_timeouts_or_capabilities():
    from histoforge.cloud.architectures.cellpose.runner import CellposeRunner
    from histoforge.cloud.architectures.hovernet.runner import HovernetRunner
    from histoforge.cloud.architectures.semantic.runner import SemanticRunner

    policy = registry.model_run_assembly_resources()
    assert policy == {
        "cpu": 4,
        "memory": 32768,
        "timeout": 7200,
        "max_containers": 4,
        "retries": 0,
    }
    spec = app.registered_functions["assemble_model_run_input"].spec
    assert spec.gpus is None
    assert spec.cpu == policy["cpu"]
    assert spec.memory == policy["memory"]
    for cls in (CellposeRunner, HovernetRunner, SemanticRunner):
        assert "finalize_durable_core_halo_input" not in cls._get_user_cls().__dict__
    definitions = registry.runner_registry()["definitions"]
    for architecture, definition in definitions.items():
        assert registry.runner_resources(architecture).attempt_timeout_seconds == 600
        assert (
            definition["capabilities"]["finite"]["max_attempt_execution_seconds"] == 600
        )
    assert registry.MODEL_RUN_MAX_EXECUTION_SECONDS == 43200
    assert definitions["dinov3-adapter-semantic"]["capabilities"]["modes"] == [
        "FINITE_CALL"
    ]
    assert "large_region_support" not in definitions["dinov3-adapter-hovernet"]


@pytest.mark.parametrize("accept_publication", [False, True])
def test_completed_cores_dispatch_to_cpu_and_still_obey_publication_fence(
    monkeypatch, accept_publication
):
    context = _context()
    context["request"]["inputs"] = context["request"]["inputs"][:1]
    context["execution_plans"] = [{"input_id": 3, "plan": _core_halo_plan()}]
    retained = [{"input_id": 3, "unit_id": unit} for unit in range(2)]
    context["completed_units"] = retained
    lifecycle = _Lifecycle(context)
    calls = []

    def assemble(*args):
        calls.append(args)
        return {"input": {"input_id": 3}, "decoded_bytes": 7}

    monkeypatch.setattr(
        executor, "assemble_model_run_input", SimpleNamespace(remote=assemble)
    )
    # No GPU method exists: resuming fully completed cores must need only CPU.
    runner = executor._RemoteRunner(object(), 4)
    if not accept_publication:
        lifecycle.publish_input = lambda *_args: False
    assert (
        executor.execute_reserved_model_run("run-id", lifecycle, lambda _r: runner)
        is accept_publication
    )
    assert calls == [(context["request"], 3, context["limits"], retained, [])]
    assert lifecycle.completed == (["run-id"] if accept_publication else [])
    assert lifecycle.published_unit_failures == []
    assert lifecycle.failed == []
    assert context["completed_units"] == retained
