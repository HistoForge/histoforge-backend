"""Catalog choices resolve once into the existing exact execution identity."""

from copy import deepcopy
from dataclasses import replace

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures import registry
from histoforge.cloud.inference.activation import (
    installed_backend_identity,
    installed_model_descriptor,
    resolve_installed_model_activation,
)
from histoforge.cloud.inference.execution import require_activation_identity
from histoforge.cloud.inference.route_activation import (
    current_model_activation,
    resolve_requested_inference_activation,
)
from histoforge.cloud.inference.runs import assembly
from histoforge.cloud.modules.catalog import validate_catalog_page
from tests.test_module_catalog import _install, _page, _store


@pytest.fixture
def installed(tmp_path):
    store = _store(tmp_path)
    record = _install(store, tmp_path, "owner/module")
    recipe = store.current.load_recipe("owner/module")
    activation = current_model_activation(
        store.current, "owner/module", "model", lambda _run: True
    )
    return store, record, recipe, activation


@pytest.mark.parametrize("precision", ["default", "bf16"])
def test_selection_reaches_admission_and_exact_worker_metadata(installed, precision):
    store, record, recipe, native = installed
    before = deepcopy(native)
    build = registry.runner_build_for_precision("dinov3-adapter-cellpose", precision)
    expected = installed_backend_identity(record, recipe, "model", "A100", build)
    selected = resolve_requested_inference_activation(expected, native)
    require_activation_identity(expected, selected)
    assert selected["backend"] == expected
    assert selected["capabilities"] == {**native["capabilities"], "runner_build": build}
    metadata, path = resolve_installed_model_activation(
        store.current,
        "owner/module",
        "model",
        record["content_digest"],
        record["resolved_commit_sha"],
        "dinov3-adapter-cellpose",
        "A100",
        build,
    )
    assert metadata["backend"] == expected
    assert metadata["capabilities"] == selected["capabilities"]
    assert path.read_bytes() == b"weights"  # Metadata validation did not load tensors.
    assert native == before
    assert store.current.load_recipe("owner/module") == recipe


def test_unadvertised_or_wrong_gpu_build_is_rejected_not_downgraded(installed):
    _store_value, _record, _recipe_value, native = installed
    bf16 = registry.runner_build_for_precision("dinov3-adapter-cellpose", "bf16")
    for build, gpu in ((bf16, "H200"), ({**bf16, "manifest_digest": "0" * 64}, "L40S")):
        # H200 is a valid recipe GPU but is not advertised for this BF16 runner.
        native_with_gpu = {**native, "backend": {**native["backend"], "gpu": gpu}}
        expected = {**native_with_gpu["backend"], "runner_build": build}
        with pytest.raises(GatewayFailure) as failure:
            resolve_requested_inference_activation(expected, native_with_gpu)
        assert failure.value.error["code"] == "MODULE_IDENTITY_MISMATCH"


def test_precision_does_not_allow_other_identity_changes_or_cross_precision_reuse(
    installed,
):
    _store_value, _record, _recipe_value, native = installed
    build = registry.runner_build_for_precision("dinov3-adapter-cellpose", "bf16")
    expected = {**native["backend"], "runner_build": build}
    selected = resolve_requested_inference_activation(expected, native)
    for wrong in (
        native["backend"],
        {**expected, "model_config_digest": "0" * 64},
        {**expected, "artifact_digests": {"weights": "0" * 64}},
    ):
        with pytest.raises(GatewayFailure):
            require_activation_identity(wrong, selected)


@pytest.mark.parametrize("precision", ["default", "bf16"])
def test_resolution_does_not_repair_inconsistent_current_capabilities(
    installed, precision
):
    *_rest, native = installed
    native = deepcopy(native)
    native["capabilities"]["runner_build"]["manifest_digest"] = "0" * 64
    expected = {
        **native["backend"],
        "runner_build": registry.runner_build_for_precision(
            "dinov3-adapter-cellpose", precision
        ),
    }
    with pytest.raises(GatewayFailure) as failure:
        resolve_requested_inference_activation(expected, native)
    assert failure.value.error["code"] == "MODULE_IDENTITY_MISMATCH"


def test_unavailable_model_and_runner_without_bf16_advertise_no_override(
    installed, monkeypatch
):
    _store_value, record, recipe, _native = installed
    unsupported = deepcopy(recipe)
    unsupported["models"]["model"]["arch"] = "future-runner"
    descriptor = installed_model_descriptor(record, unsupported, "model")
    assert descriptor["runner_availability"]["type"] == "unavailable"
    assert descriptor["precision_overrides"] == {}
    architecture = recipe["models"]["model"]["arch"]
    monkeypatch.setitem(
        registry._RUNNER_DEFINITIONS,
        architecture,
        replace(registry._RUNNER_DEFINITIONS[architecture], supports_bf16=False),
    )
    registry.runner_registry.cache_clear()
    try:
        assert (
            installed_model_descriptor(record, recipe, "model")["precision_overrides"]
            == {}
        )
        with pytest.raises(ValueError):
            registry.runner_build_for_precision(architecture, "bf16")
    finally:
        registry.runner_registry.cache_clear()


def test_catalog_requires_explicit_precision_contract_even_when_empty(installed):
    store, *_ = installed
    page = _page(store)
    model = page["modules"][0]["models"]["model"]
    model.pop("precision_overrides")
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_catalog_page(page)
    model["precision_overrides"] = {}
    assert validate_catalog_page(page) == page


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {"fp16": {}},
        {"bf16": {}},
        {"bf16": {"runner_build": {}, "gpus": ["A100"]}},
    ],
)
def test_catalog_rejects_malformed_precision_options(installed, invalid):
    store, *_ = installed
    page = _page(store)
    page["modules"][0]["models"]["model"]["precision_overrides"] = invalid
    with pytest.raises(ValueError):
        validate_catalog_page(page)


@pytest.mark.parametrize("gpus", [[], [""], ["  "], ["A100", "A100"], [None], [{}]])
def test_catalog_rejects_unusable_bf16_gpu_lists(installed, gpus):
    store, *_ = installed
    page = _page(store)
    override = page["modules"][0]["models"]["model"]["precision_overrides"]["bf16"]
    override["gpus"] = gpus
    with pytest.raises(ValueError, match="BF16 GPUs"):
        validate_catalog_page(page)


def test_catalog_gpu_names_are_not_a_second_provider_allowlist(installed):
    store, *_ = installed
    page = _page(store)
    override = page["modules"][0]["models"]["model"]["precision_overrides"]["bf16"]
    override["gpus"] = ["FutureGPU"]
    assert validate_catalog_page(page) == page
    with pytest.raises(ValueError, match="not supported"):
        registry.runner_precision_for_build(
            "dinov3-adapter-cellpose", override["runner_build"], "FutureGPU"
        )


def test_catalog_rejects_shared_native_bf16_build_and_unavailable_override(installed):
    store, *_ = installed
    page = _page(store)
    model = page["modules"][0]["models"]["model"]
    override = model["precision_overrides"]["bf16"]
    override["runner_build"] = model["runner_availability"]["capabilities"][
        "runner_build"
    ]
    with pytest.raises(ValueError, match="must differ"):
        validate_catalog_page(page)
    model["runner_availability"] = {
        "type": "unavailable",
        "code": "RUNNER_NOT_DEPLOYED",
        "message": "Unavailable",
    }
    with pytest.raises(ValueError, match="unavailable runner"):
        validate_catalog_page(page)


def test_cpu_assembly_preserves_the_captured_precision_build(installed, monkeypatch):
    from contextlib import nullcontext

    store, record, recipe, _native = installed
    build = registry.runner_build_for_precision("dinov3-adapter-cellpose", "bf16")
    expected = installed_backend_identity(record, recipe, "model", "A100", build)
    monkeypatch.setattr(
        "histoforge.cloud.modules.runtime.current_module_snapshot",
        lambda: nullcontext(store.current),
    )
    monkeypatch.setattr(
        "histoforge.cloud.artifacts.store.ArtifactFileStore", lambda *_a, **_k: object()
    )
    captured = []

    def finalize(request, input_id, limits, units, activation, files, **kwargs):
        require_activation_identity(request["expected_backend"], activation)
        captured.append(activation)
        return {"input": {"input_id": input_id}}

    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime.finalize_core_halo_input",
        finalize,
    )
    result = assembly.assemble_model_run_input.local(
        {"expected_backend": expected}, 7, {}, []
    )
    assert result == {"input": {"input_id": 7}}
    assert captured[0]["backend"]["runner_build"] == build
    assert captured[0]["capabilities"]["runner_build"] == build
    assert "network" not in captured[0]
