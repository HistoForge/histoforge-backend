"""Target catalog projection, availability, pagination, and admission bounds."""

import json
import uuid
from copy import deepcopy
from pathlib import Path

import pytest
from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    sha256_bytes,
)
from histoforge.cloud.inference.activation import installed_backend_identity
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.modules.catalog import catalog_page, validate_catalog_page
from histoforge.cloud.modules.store import ModuleStore
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import succeeded_report


def _recipe(*, name="Example", script=False):
    steps = [{"model": "model"}]
    if script:
        steps.append({"script": "classify.groovy"})
    return {
        "schemaVersion": 3,
        "name": name,
        "models": {
            "model": {
                "arch": "dinov3-adapter-cellpose",
                "gpu": "L40S",
                "weights": {"type": "vendored", "file": "model.safetensors"},
                "input": {"type": "field", "micronsPerPixel": 0.22},
                "output": {
                    "type": "instance_segmentation",
                    "classes": ["Cell"],
                },
                "params": {"num_classes": 0, "min_instance_pixels": 15},
            }
        },
        "steps": steps,
    }


def _install(
    store,
    tmp_path,
    module_ref,
    *,
    name="Example",
    script=False,
    weight_bytes=b"weights",
    large_region=None,
):
    source = tmp_path / module_ref.replace("/", "-")
    source.mkdir()
    recipe = _recipe(name=name, script=script)
    if large_region is not None:
        recipe["models"]["model"]["input"]["largeRegion"] = large_region
    (source / "recipe.json").write_text(
        json.dumps(recipe),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(weight_bytes)
    if script:
        (source / "classify.groovy").write_text(
            '/* @histoforge-classes ["Derived"] */\nreturn []\n',
            encoding="utf-8",
        )
    return store.install_from_directory(
        source,
        module_ref=module_ref,
        resolved_commit_sha="a" * 40,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
        installed_at="2026-08-15T12:34:56.789Z",
    )


def _store(tmp_path, *, catalog_cap=8 * 1024**2):
    return ModuleStore(
        tmp_path / "modules",
        max_module_bytes=1_000_000,
        max_installed_bytes=10_000_000,
        max_catalog_encoded_bytes=catalog_cap,
        training_api_limits=training_api_limits(),
    )


def _page(store, *, limit=100, cursor=None, byte_cap=8 * 1024**2):
    return catalog_page(
        store.current,
        training_result_visible=lambda _run_id: True,
        limit=limit,
        cursor=cursor,
        max_page_size=100,
        max_encoded_bytes=byte_cap,
    )


def test_catalog_projects_exact_identity_content_and_runner_availability(tmp_path):
    store = _store(tmp_path)
    _install(store, tmp_path, "owner/zeta", name="Zeta")
    alpha_record = _install(store, tmp_path, "owner/alpha", name="Alpha", script=True)

    page = _page(store)
    catalog = page["modules"]

    assert "next_cursor" not in page
    assert [entry["identity"]["module_ref"] for entry in catalog] == [
        "owner/alpha",
        "owner/zeta",
    ]
    alpha = catalog[0]
    assert alpha["installed_at"] == "2026-08-15T12:34:56.789Z"
    assert alpha["installed_module_file_bytes"] == sum(
        item["size_bytes"] for item in alpha_record["executable_files"]
    )
    assert alpha["identity"] == {
        "module_ref": "owner/alpha",
        "resolved_commit_sha": "a" * 40,
        "recipe_digest": alpha_record["recipe_digest"],
        "content_digest": alpha_record["content_digest"],
    }
    assert alpha["recipe"]["name"] == "Alpha"
    assert alpha["scripts"] == {
        "classify.groovy": '/* @histoforge-classes ["Derived"] */\nreturn []\n'
    }
    assert len(alpha["script_set_digest"]) == 64
    model = alpha["models"]["model"]
    assert model["model_config_digest"]
    assert model["artifact_digests"] == {
        "weights": alpha_record["resolved_artifacts"][0]["sha256"]
    }
    assert model["runner_availability"]["type"] == "available"
    capabilities = model["runner_availability"]["capabilities"]
    assert capabilities["modes"] == ["DURABLE_MODEL_RUN", "FINITE_CALL"]
    assert capabilities["durable_run"]["run"]["max_inputs"] == 1
    backend = installed_backend_identity(alpha_record, alpha["recipe"], "model")
    assert backend["module"] == alpha["identity"]
    assert backend["model_id"] == "model"
    assert backend["model_config_digest"] == model["model_config_digest"]
    assert backend["artifact_digests"] == model["artifact_digests"]
    overridden = installed_backend_identity(
        alpha_record, alpha["recipe"], "model", "A100"
    )
    assert overridden == {**backend, "gpu": "A100"}
    with pytest.raises(ValueError, match="not supported"):
        installed_backend_identity(alpha_record, alpha["recipe"], "model", "T4")
    training = model["training_availability"]
    assert training["type"] == "available"
    assert {
        option["capability"]["output_type"]: option["class_initializations"]
        for option in training["options"]
    } == {
        "instance_segmentation": ["use"],
        "instance_segmentation_and_classification": ["add", "define"],
    }
    assert len(training["options"]) == 2


def test_catalog_size_includes_training_report_bytes(tmp_path):
    store = _store(tmp_path)
    source = tmp_path / "trained-source"
    source.mkdir()
    weights = b"trained weights"
    report = succeeded_report()
    report["selected_model_id"] = "model"
    report["model_selection"]["selected_weight_digest"] = sha256_bytes(weights)
    report["task"]["model_microns_per_pixel"] = 0.22
    report["task"]["model_output"] = {
        "type": "instance_segmentation",
        "classes": ["Cell"],
    }
    report["task"]["annotation_class_map"] = {"Cell": "Cell"}
    report["source_model_output"] = deepcopy(report["task"]["model_output"])
    report["source_model_microns_per_pixel"] = 0.22
    for partition in ("training_counts", "validation_counts"):
        report[partition]["target_object_counts"] = {
            "Cell": report[partition]["target_object_counts"]["Leydig cell"]
        }
    report["field_profile"] = {
        "planner": "fixed-field-v1",
        "side_model_px": 256,
        "alignment_model_px": 16,
        "small_roi_context": False,
        "tile_overlap": 0.1,
        "aggregation": "cellpose-tapered-mean-v1",
        "preprocessing": "normalize99-real-image-v1",
    }
    report["trainer_configuration"]["preprocessing"]["profile_id"] = (
        "normalize99-real-image-v1"
    )
    recipe = _recipe(name="Trained")
    recipe["models"]["model"]["gpu"] = report["result_inference_gpu"]
    recipe["trainingReportDigest"] = canonical_digest(report)
    (source / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    (source / "model.safetensors").write_bytes(weights)
    report_bytes = canonical_bytes(report)
    (source / "training-report.json").write_bytes(report_bytes)
    record = store.install_from_directory(
        source,
        module_ref="owner/trained",
        resolved_commit_sha="a" * 40,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
        installed_at="2026-08-15T12:34:56.789Z",
    )

    descriptor = _page(store)["modules"][0]
    executable_bytes = sum(item["size_bytes"] for item in record["executable_files"])
    assert descriptor["installed_module_file_bytes"] == (
        executable_bytes + len(report_bytes)
    )
    assert store.current.load_training_report("owner/trained") == report


def test_catalog_projects_recipe_settings_independently_of_weight_identity(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    profile = build_core_halo_profile(3_520, 3_520, 1_024, "1" * 64)
    from histoforge.cloud.architectures import registry

    original = registry._RUNNER_DEFINITIONS["dinov3-adapter-cellpose"]
    monkeypatch.setitem(
        registry._RUNNER_DEFINITIONS,
        "dinov3-adapter-cellpose",
        registry.RunnerDefinition(
            resources=original.resources,
            large_region_support={
                "type": "core-halo",
                "validation_evidence_digest": "1" * 64,
            },
        ),
    )
    registry.runner_registry.cache_clear()
    try:
        settings = {
            "coreWidthModelPx": 3520,
            "coreHeightModelPx": 3520,
            "haloPerSideModelPx": 1024,
        }
        _install(
            store,
            tmp_path,
            "owner/alpha",
            weight_bytes=b"first weights",
            large_region=settings,
        )
        _install(
            store,
            tmp_path,
            "owner/beta",
            weight_bytes=b"different weights",
            large_region=settings,
        )

        models = [module["models"]["model"] for module in _page(store)["modules"]]

        assert [model["large_region_profile"] for model in models] == [
            profile,
            profile,
        ]
        assert (
            models[0]["artifact_digests"]["weights"]
            != models[1]["artifact_digests"]["weights"]
        )
    finally:
        registry.runner_registry.cache_clear()


def test_catalog_cursor_is_exclusive_and_pages_are_canonical_byte_bounded(tmp_path):
    store = _store(tmp_path)
    for name in ("alpha", "middle", "zeta"):
        _install(store, tmp_path, f"owner/{name}", name=name)

    first = _page(store, limit=2)
    assert [item["identity"]["module_ref"] for item in first["modules"]] == [
        "owner/alpha",
        "owner/middle",
    ]
    assert first["next_cursor"]
    assert len(canonical_bytes(first)) <= 8 * 1024**2

    second = _page(store, limit=2, cursor=first["next_cursor"])
    assert [item["identity"]["module_ref"] for item in second["modules"]] == [
        "owner/zeta"
    ]
    assert "next_cursor" not in second

    with pytest.raises(ValueError, match="cursor"):
        _page(store, cursor="not+canonical")


def test_catalog_byte_bound_stops_before_next_complete_descriptor(tmp_path):
    store = _store(tmp_path)
    _install(store, tmp_path, "owner/alpha")
    _install(store, tmp_path, "owner/zeta")
    first_only = _page(store, limit=1)
    exact_cap = len(canonical_bytes(first_only))

    bounded = _page(store, limit=100, byte_cap=exact_cap)

    assert len(bounded["modules"]) == 1
    assert bounded["next_cursor"] == first_only["next_cursor"]
    assert len(canonical_bytes(bounded)) == exact_cap


def test_catalog_fails_if_current_executable_bytes_changed(tmp_path):
    store = _store(tmp_path)
    _install(store, tmp_path, "owner/module", script=True)
    script = store.current.module_dir("owner/module") / "classify.groovy"
    script.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="installed identity"):
        _page(store)


def test_install_rejects_module_that_cannot_fit_one_catalog_page(tmp_path):
    store = _store(tmp_path, catalog_cap=1_000)

    with pytest.raises(ValueError, match="catalog byte limit"):
        _install(store, tmp_path, "owner/module", script=True)

    assert store.current.list_module_refs() == []


def test_shared_catalog_fixture_closes_content_and_availability_contract():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures" / "contracts" / "module-catalog-v1.json"
        ).read_text(encoding="utf-8")
    )
    page = validate_catalog_page(fixture["page"])
    encoded = canonical_bytes(page)

    assert page == fixture["page"]
    assert encoded.decode("utf-8") == fixture["canonical"]
    assert sha256_bytes(encoded) == fixture["sha256"]

    wrong_model_digest = deepcopy(fixture["page"])
    wrong_model_digest["modules"][0]["models"]["cells"]["model_config_digest"] = (
        "f" * 64
    )
    with pytest.raises(ValueError, match="model_config_digest"):
        validate_catalog_page(wrong_model_digest)

    missing_commit = deepcopy(fixture["page"])
    missing_commit["modules"][0]["identity"].pop("resolved_commit_sha")
    with pytest.raises(ValueError, match="commit or reserved training-result ref"):
        validate_catalog_page(missing_commit)

    workspace_result = deepcopy(fixture["page"])
    workspace_result.pop("next_cursor", None)
    workspace_result["modules"][0]["identity"].pop("resolved_commit_sha")
    workspace_result["modules"][0]["identity"]["module_ref"] = (
        "workspace/run-44444444-4444-4444-8444-444444444444"
    )
    validate_catalog_page(workspace_result)

    wrong_artifact_role = deepcopy(fixture["page"])
    artifacts = wrong_artifact_role["modules"][0]["models"]["cells"]["artifact_digests"]
    artifacts["not-a-recipe-role"] = artifacts.pop("weights")
    with pytest.raises(ValueError, match="artifact roles"):
        validate_catalog_page(wrong_artifact_role)

    missing_durable_limits = deepcopy(fixture["page"])
    capabilities = missing_durable_limits["modules"][0]["models"]["cells"][
        "runner_availability"
    ]["capabilities"]
    capabilities.pop("durable_run")
    with pytest.raises(ValueError, match="durable_run"):
        validate_catalog_page(missing_durable_limits)

    wrong_cursor = deepcopy(fixture["page"])
    wrong_cursor["next_cursor"] = "b3duZXIvb3RoZXI"
    with pytest.raises(ValueError, match="exclusive last ref"):
        validate_catalog_page(wrong_cursor)


def test_catalog_hides_attested_result_until_the_state_owner_allows_it(
    monkeypatch,
):
    run_id = "44444444-4444-4444-8444-444444444444"
    workspace_ref = f"workspace/run-{run_id}"

    class Current:
        @staticmethod
        def list_module_refs():
            return ["owner/module", workspace_ref]

        @staticmethod
        def read_record(module_ref):
            if module_ref == workspace_ref:
                return {"training_result_attestation": {"run_id": run_id}}
            return {}

    monkeypatch.setattr(
        "histoforge.cloud.modules.catalog.installed_module_descriptor",
        lambda _current, module_ref: {
            "identity": {"module_ref": module_ref},
        },
    )

    hidden = catalog_page(
        Current(),
        training_result_visible=lambda _run_id: False,
        limit=100,
        cursor=None,
        max_page_size=100,
        max_encoded_bytes=1_000_000,
    )
    visible = catalog_page(
        Current(),
        training_result_visible=lambda _run_id: True,
        limit=100,
        cursor=None,
        max_page_size=100,
        max_encoded_bytes=1_000_000,
    )

    assert [item["identity"]["module_ref"] for item in hidden["modules"]] == [
        "owner/module"
    ]
    assert [item["identity"]["module_ref"] for item in visible["modules"]] == [
        "owner/module",
        workspace_ref,
    ]
