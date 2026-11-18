from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from histoforge.canonical_json import parse_json
from histoforge.cloud import _app
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.registry import resolved_fixed_field_profile
from histoforge.cloud.modules import runtime as module_runtime
from histoforge.cloud.modules.identity import module_content_identity
from histoforge.cloud.modules.store import ModuleStore
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.source_preparation import TrainingSourcePreparation
from tests.test_training_capabilities import _variable_capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_run_request import run_request

RUN_ID = "44444444-4444-4444-8444-444444444444"


def model(name: str, *, weights: dict | None = None, input_type: str = "field") -> dict:
    model_input = {"type": input_type, "micronsPerPixel": 0.22}
    if input_type == "crop":
        model_input["framing"] = {"paddingModelPx": 32}
    return {
        "arch": "dinov3-adapter-cellpose",
        "gpu": "L4",
        "weights": weights or {"type": "vendored", "file": f"{name}.safetensors"},
        "input": model_input,
        "output": {"type": "instance_segmentation", "classes": [name.title()]},
        "params": {"num_classes": 0, "min_instance_pixels": 15},
    }


def install_module(
    tmp_path: Path,
    recipe: dict,
    *,
    external: dict[tuple[str, str], Path] | None = None,
    vendored_weight_bytes: dict[str, bytes] | None = None,
    retained_files: dict[str, bytes] | None = None,
) -> ModuleStore:
    source = tmp_path / "source-module"
    source.mkdir(parents=True)
    (source / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    for model_id, value in recipe["models"].items():
        if value["weights"]["type"] == "vendored":
            payload = (
                vendored_weight_bytes[model_id]
                if vendored_weight_bytes is not None
                and model_id in vendored_weight_bytes
                else f"{model_id} weights".encode()
            )
            (source / value["weights"]["file"]).write_bytes(payload)
    for step in recipe["steps"]:
        if "script" in step:
            (source / step["script"]).write_text(
                '/* @histoforge-classes ["Derived"] */\nreturn []\n',
                encoding="utf-8",
            )
    for name, content in (retained_files or {}).items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    store = ModuleStore(
        tmp_path / "modules",
        max_module_bytes=1_000_000,
        max_installed_bytes=10_000_000,
        max_catalog_encoded_bytes=8 * 1024**2,
    )
    store.install_from_directory(
        source,
        module_ref="owner/workflow",
        resolved_commit_sha="a" * 40,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
        external_artifacts=external,
        installed_at="2026-09-04T12:00:00.000Z",
    )
    return store


def workflow_recipe() -> dict:
    return {
        "schemaVersion": 3,
        "name": "Source workflow",
        "description": "Description for the complete workflow",
        "models": {"nuclei": model("nuclei"), "cells": model("cells")},
        "steps": [
            {"model": "nuclei"},
            {"model": "cells"},
            {"script": "classify.groovy"},
        ],
    }


def request_for(
    store: ModuleStore, *, result_form: str = "selected_model_only"
) -> dict:
    request = run_request()
    request["source_module"] = module_content_identity(
        store.current.read_record("owner/workflow")
    )
    request["selected_model_id"] = "nuclei"
    request["result_form"] = result_form
    model = store.current.load_recipe("owner/workflow")["models"]["nuclei"]
    request["field_profile"] = model["input"].get(
        "fieldProfile"
    ) or resolved_fixed_field_profile(model)
    return request


def prepare(tmp_path: Path, store: ModuleStore) -> TrainingSourcePreparation:
    return TrainingSourcePreparation(
        lambda: nullcontext(store.current),
        TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2),
        training_api_limits(),
    )


def pending_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts" / "training" / "runs" / "pending" / RUN_ID


def candidate_root(tmp_path: Path, report_digest: str) -> Path:
    return (
        tmp_path
        / "artifacts"
        / "training"
        / "runs"
        / "source-candidates"
        / RUN_ID
        / report_digest
    )


def test_selected_model_snapshot_omits_other_weights_but_retains_scripts(
    tmp_path: Path,
) -> None:
    store = install_module(tmp_path, workflow_recipe())

    with prepare(tmp_path, store).prepare(request_for(store)) as prepared:
        marker = prepared.publish(RUN_ID, "b" * 64)

    assert [item["relative_path"] for item in marker["source_files"]] == [
        "source/module/classify.groovy",
        "source/module/nuclei.safetensors",
        "source/module/recipe.json",
    ]
    recipe = parse_json(
        (
            candidate_root(tmp_path, "b" * 64) / "source" / "module" / "recipe.json"
        ).read_bytes()
    )
    assert list(recipe["models"]) == ["nuclei"]
    assert recipe["steps"] == [{"model": "nuclei"}]
    assert "description" not in recipe
    assert marker["starting_weight_path"] == "source/module/nuclei.safetensors"


def test_complete_workflow_snapshot_preserves_every_executable(tmp_path: Path) -> None:
    store = install_module(tmp_path, workflow_recipe())

    with prepare(tmp_path, store).prepare(
        request_for(store, result_form="complete_source_workflow")
    ) as prepared:
        marker = prepared.publish(RUN_ID, "b" * 64)

    assert [item["relative_path"] for item in marker["source_files"]] == [
        "source/module/cells.safetensors",
        "source/module/classify.groovy",
        "source/module/nuclei.safetensors",
        "source/module/recipe.json",
    ]
    copied = parse_json(
        (
            candidate_root(tmp_path, "b" * 64) / "source" / "module" / "recipe.json"
        ).read_bytes()
    )
    assert copied == store.current.load_recipe("owner/workflow")


def test_selected_referenced_weights_are_copied_into_the_snapshot(
    tmp_path: Path,
) -> None:
    referenced = {
        "type": "referenced",
        "ref": "foundation/weights",
        "revision": "c" * 40,
    }
    recipe = workflow_recipe()
    recipe["models"]["nuclei"]["weights"] = referenced
    weights = tmp_path / "referenced.safetensors"
    weights.write_bytes(b"referenced weights")
    store = install_module(tmp_path, recipe, external={("nuclei", "weights"): weights})

    with prepare(tmp_path, store).prepare(request_for(store)) as prepared:
        marker = prepared.publish(RUN_ID, "b" * 64)

    assert marker["starting_weight_path"] == (
        "source/artifacts/nuclei/weights.safetensors"
    )
    assert (
        candidate_root(tmp_path, "b" * 64)
        / "source"
        / "artifacts"
        / "nuclei"
        / "weights.safetensors"
    ).read_bytes() == b"referenced weights"


def test_complete_workflow_requires_runnable_steps_before_snapshot(
    tmp_path: Path,
) -> None:
    recipe = workflow_recipe()
    recipe["models"]["cells"]["params"]["pretrained_encoder"] = True
    store = install_module(tmp_path, recipe)

    with pytest.raises(GatewayFailure, match="copied workflow model 'cells'"):
        with prepare(tmp_path, store).prepare(
            request_for(store, result_form="complete_source_workflow")
        ):
            pytest.fail("an unavailable copied step must fail before preparation")
    assert not (tmp_path / "artifacts").exists()

    # Standalone fine-tuning does not retain the unavailable model.
    with prepare(tmp_path, store).prepare(request_for(store)) as prepared:
        assert prepared.request["result_form"] == "selected_model_only"


def test_unused_model_does_not_block_a_complete_executable_workflow(
    tmp_path: Path,
) -> None:
    recipe = workflow_recipe()
    recipe["models"]["unused"] = model("unused")
    recipe["models"]["unused"]["params"]["pretrained_encoder"] = True
    store = install_module(tmp_path, recipe)
    with prepare(tmp_path, store).prepare(
        request_for(store, result_form="complete_source_workflow")
    ) as prepared:
        assert prepared.request["result_form"] == "complete_source_workflow"


@pytest.mark.parametrize(
    "result_form", ["selected_model_only", "complete_source_workflow"]
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("aggregation", "unknown-v1"),
        ("preprocessing", "unknown-v1"),
        ("alignmentModelPx", 32),
    ],
)
def test_requested_result_profile_is_checked_before_source_file_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result_form: str, field: str, value
) -> None:
    store = install_module(tmp_path, workflow_recipe())
    request = request_for(store, result_form=result_form)
    request["field_profile"] = _variable_capability()["default_field_profile"]
    request["field_profile"][field] = value
    preparation = prepare(tmp_path, store)

    def unexpected_source_files(*args):
        pytest.fail("unsupported result profile reached source-file preparation")

    monkeypatch.setattr(preparation, "_source_files", unexpected_source_files)
    with pytest.raises(GatewayFailure, match="requested result settings") as rejected:
        with preparation.prepare(request):
            pytest.fail("unsupported result profile was admitted")
    assert rejected.value.error["code"] == "VALIDATION_ERROR"
    assert not (tmp_path / "artifacts").exists()


def test_result_inference_gpu_accepts_source_default_or_curated_override(
    tmp_path: Path,
) -> None:
    store = install_module(tmp_path, workflow_recipe())
    source_default = request_for(store)
    source_default["result_inference_gpu"] = "L4"
    with prepare(tmp_path, store).prepare(source_default) as prepared:
        assert prepared.request["result_inference_gpu"] == "L4"

    unsupported = request_for(store)
    unsupported["result_inference_gpu"] = "T4"
    with pytest.raises(GatewayFailure, match="requested result settings") as rejected:
        with prepare(tmp_path / "unsupported", store).prepare(unsupported):
            pytest.fail("unsupported result inference GPU was admitted")
    assert rejected.value.error["code"] == "VALIDATION_ERROR"


def test_source_identity_and_result_form_are_checked_before_publication(
    tmp_path: Path,
) -> None:
    store = install_module(tmp_path, workflow_recipe())
    request = request_for(store)
    request["source_module"]["content_digest"] = "f" * 64
    with pytest.raises(GatewayFailure) as mismatch:
        with prepare(tmp_path, store).prepare(request):
            pass
    assert mismatch.value.error["code"] == "MODULE_IDENTITY_MISMATCH"

    one_model = workflow_recipe()
    one_model["models"] = {"nuclei": one_model["models"]["nuclei"]}
    one_model["steps"] = [{"model": "nuclei"}]
    isolated = install_module(tmp_path / "isolated", one_model)
    with pytest.raises(GatewayFailure) as incompatible:
        with prepare(tmp_path / "isolated", isolated).prepare(
            request_for(isolated, result_form="complete_source_workflow")
        ):
            pass
    assert incompatible.value.error["code"] == "VALIDATION_ERROR"


def test_region_gated_model_cannot_be_snapshotted_as_standalone(
    tmp_path: Path,
) -> None:
    recipe = workflow_recipe()
    recipe["models"]["cells"] = model("cells", input_type="crop")
    recipe["steps"] = [
        {"model": "nuclei"},
        {"model": "cells", "parentModel": "nuclei", "parentClass": ["Nuclei"]},
    ]
    store = install_module(tmp_path, recipe)
    request = request_for(store)
    request["selected_model_id"] = "cells"

    with pytest.raises(GatewayFailure) as incompatible:
        with prepare(tmp_path, store).prepare(request):
            pass
    assert incompatible.value.error["code"] == "VALIDATION_ERROR"


def test_source_change_after_install_fails_closed(tmp_path: Path) -> None:
    store = install_module(tmp_path, workflow_recipe())
    request = request_for(store)
    path = store.current.module_dir("owner/workflow") / "nuclei.safetensors"
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="does not match installed bytes"):
        with prepare(tmp_path, store).prepare(request):
            pass

    assert not pending_root(tmp_path).exists()


def test_recipe_read_is_bound_to_the_revalidated_module_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = install_module(tmp_path, workflow_recipe())
    request = request_for(store)
    replacement_recipe = workflow_recipe()
    replacement_recipe["name"] = "Replacement workflow"
    replacement = install_module(tmp_path / "replacement", replacement_recipe)
    original_revalidate = store.current.revalidate

    def revalidate_then_replace(module_ref: str) -> dict:
        frozen = original_revalidate(module_ref)
        visible = store.current.module_dir(module_ref)
        shutil.rmtree(visible)
        shutil.copytree(replacement.current.module_dir(module_ref), visible)
        return frozen

    monkeypatch.setattr(store.current, "revalidate", revalidate_then_replace)

    with pytest.raises(ValueError, match="does not match installed identity"):
        with prepare(tmp_path, store).prepare(request):
            pass

    assert not pending_root(tmp_path).exists()


@pytest.mark.parametrize("publication_fails", [False, True])
def test_module_mount_stays_stable_through_deferred_source_publication(
    tmp_path, monkeypatch, publication_fails
):
    store = install_module(tmp_path, workflow_recipe())
    request = request_for(store)
    attempted = Event()
    reloaded = Event()
    reload_count = 0

    def reload():
        nonlocal reload_count
        reload_count += 1
        if reload_count == 2:
            reloaded.set()

    monkeypatch.setattr(_app, "model_volume", SimpleNamespace(reload=reload))
    monkeypatch.setattr(module_runtime, "mounted_module_store", lambda: store)
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    publish = files.publish_candidate

    def checked_publish(marker, paths):
        assert not reloaded.is_set()
        assert (
            paths["source/module/nuclei.safetensors"].read_bytes() == b"nuclei weights"
        )
        if publication_fails:
            raise OSError("source publication failed")
        return publish(marker, paths)

    monkeypatch.setattr(files, "publish_candidate", checked_publish)
    preparation = TrainingSourcePreparation(
        module_runtime.current_module_snapshot, files, training_api_limits()
    )

    def competing_reader():
        attempted.set()
        with module_runtime.current_module_snapshot() as current:
            return current.read_record("owner/workflow")

    with ThreadPoolExecutor(max_workers=1) as executor:
        following = []

        def prepare_and_publish():
            with preparation.prepare(request) as prepared:
                following.append(executor.submit(competing_reader))
                assert attempted.wait(5)
                assert not reloaded.wait(0.05)
                return prepared.publish(RUN_ID, "b" * 64)

        if publication_fails:
            with pytest.raises(OSError, match="source publication failed"):
                prepare_and_publish()
        else:
            marker = prepare_and_publish()
            assert marker["source_module"] == request["source_module"]
            copied = (
                candidate_root(tmp_path, "b" * 64) / "source/module/nuclei.safetensors"
            )
            assert copied.read_bytes() == b"nuclei weights"
        assert following[0].result(timeout=5) == store.current.read_record(
            "owner/workflow"
        )
    assert reloaded.is_set() and reload_count == 2
