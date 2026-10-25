"""Current-ref store mutation, capacity, and crash-cut behavior."""

import json
import uuid
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, canonical_digest, sha256_bytes
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.modules.identity import (
    STORED_MODULE_RECORD_PATH,
    build_stored_module_record,
)
from histoforge.cloud.modules.runtime import mounted_module_store
from histoforge.cloud.modules.store import ModuleStore, TrainingResultConflict
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import succeeded_report

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
INSTALLED_AT = "2026-08-15T12:34:56.789Z"


def _recipe(*, weights=None, name="Example"):
    return {
        "schemaVersion": 3,
        "name": name,
        "models": {
            "model": {
                "arch": "dinov3-adapter-cellpose",
                "gpu": "A10G",
                "weights": weights or {"type": "vendored", "file": "model.safetensors"},
                "input": {"type": "field", "micronsPerPixel": 0.22},
                "output": {
                    "type": "instance_segmentation",
                    "classes": ["Cell"],
                },
                "params": {"num_classes": 0, "min_instance_pixels": 15},
            }
        },
        "steps": [{"model": "model"}],
    }


def _source(root: Path, *, recipe=None, weights=b"weights") -> Path:
    root.mkdir(parents=True)
    (root / "recipe.json").write_text(json.dumps(recipe or _recipe()), encoding="utf-8")
    if weights is not None:
        (root / "model.safetensors").write_bytes(weights)
    return root


def _store(root: Path, *, module_cap=1_000_000, installed_cap=10_000_000):
    return ModuleStore(
        root,
        max_module_bytes=module_cap,
        max_installed_bytes=installed_cap,
        max_catalog_encoded_bytes=8 * 1024**2,
        training_api_limits=training_api_limits(),
    )


def _install(
    store: ModuleStore,
    source: Path,
    *,
    ref="owner/module",
    commit=COMMIT_A,
    publish=lambda: None,
    external=None,
):
    return store.install_from_directory(
        source,
        module_ref=ref,
        resolved_commit_sha=commit,
        job_id=str(uuid.uuid4()),
        publish=publish,
        external_artifacts=external,
        installed_at=INSTALLED_AT,
    )


def _visible_state(store: ModuleStore, ref="owner/module") -> str:
    directory = store.current.module_dir(ref)
    if not directory.exists():
        return "absent"
    try:
        record = store.current.read_record(ref)
    except Exception:
        return "invalid"
    return record["resolved_commit_sha"]


def test_mounted_store_uses_validated_wire_training_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("MODULES_ROOT", str(tmp_path / "modules"))

    store = mounted_module_store()

    assert (
        store.training_api_limits == DEPLOYMENT_CONFIG.training_api_limits.wire_value()
    )


def _training_result_source(root: Path, *, weights=b"trained"):
    report = succeeded_report()
    report["selected_model_id"] = "model"
    report["model_selection"]["selected_weight_digest"] = sha256_bytes(weights)
    report["task"]["model_microns_per_pixel"] = 0.22
    report["task"]["model_output"] = {
        "type": "instance_segmentation",
        "classes": ["Cell"],
    }
    report["task"]["annotation_class_map"] = {"Cell": "Cell"}
    report["source_model_output"] = report["task"]["model_output"]
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
    recipe = _recipe()
    recipe["models"]["model"]["gpu"] = report["result_inference_gpu"]
    recipe["trainingReportDigest"] = canonical_digest(report)
    source = _source(root, recipe=recipe, weights=weights)
    (source / "training-report.json").write_bytes(canonical_bytes(report))
    ref = f"workspace/run-{report['run_id']}"
    preview = build_stored_module_record(
        source,
        module_ref=ref,
        resolved_commit_sha=None,
        installed_at=INSTALLED_AT,
        training_api_limits=training_api_limits(),
    )
    attestation = {
        "run_id": report["run_id"],
        "module_content_digest": preview["content_digest"],
        "report_digest": recipe["trainingReportDigest"],
    }
    return source, ref, attestation


def test_fresh_install_publishes_private_stage_then_complete_current(tmp_path):
    store = _store(tmp_path / "modules")
    source = _source(tmp_path / "source")
    states = []

    record = _install(
        store, source, publish=lambda: states.append(_visible_state(store))
    )

    assert states == ["absent", COMMIT_A, COMMIT_A]
    assert store.current.list_module_refs() == ["owner/module"]
    assert store.current.load_recipe("owner/module")["name"] == "Example"
    assert (
        store.current.artifact_path("owner/module", "model").read_bytes() == b"weights"
    )
    assert record == store.current.read_record("owner/module")
    assert (
        store.current.module_dir("owner/module") / STORED_MODULE_RECORD_PATH
    ).is_file()
    assert not any((store.root / ".staging").iterdir())


def test_replace_visibility_is_only_old_absent_or_new(tmp_path):
    store = _store(tmp_path / "modules")
    _install(store, _source(tmp_path / "old"))
    new = _source(tmp_path / "new", recipe=_recipe(name="New"), weights=b"new")
    states = []

    _install(
        store,
        new,
        commit=COMMIT_B,
        publish=lambda: states.append(_visible_state(store)),
    )

    assert states == [COMMIT_A, "absent", COMMIT_B, COMMIT_B]
    assert "invalid" not in states
    assert store.current.load_recipe("owner/module")["name"] == "New"
    assert store.current.artifact_path("owner/module", "model").read_bytes() == b"new"


@pytest.mark.parametrize(
    ("failed_publish", "expected"),
    [(1, COMMIT_A), (2, "absent"), (3, COMMIT_B), (4, COMMIT_B)],
)
def test_replacement_failure_cuts_never_expose_mixed_bytes(
    tmp_path, failed_publish, expected
):
    store = _store(tmp_path / "modules")
    _install(store, _source(tmp_path / "old"))
    replacement = _source(
        tmp_path / "replacement", recipe=_recipe(name="New"), weights=b"new"
    )
    calls = 0

    def fail_at_cut():
        nonlocal calls
        calls += 1
        if calls == failed_publish:
            raise RuntimeError("injected publication failure")

    with pytest.raises(RuntimeError, match="injected"):
        _install(store, replacement, commit=COMMIT_B, publish=fail_at_cut)

    assert _visible_state(store) == expected
    assert _visible_state(store) != "invalid"

    # A later explicit attempt owns stale private cleanup and can reconstruct
    # the exact replacement even when the previous cut left visibility absent.
    _install(store, replacement, commit=COMMIT_B)
    assert _visible_state(store) == COMMIT_B
    assert store.current.load_recipe("owner/module")["name"] == "New"


def test_delete_commits_absence_before_private_cleanup(tmp_path):
    store = _store(tmp_path / "modules")
    _install(store, _source(tmp_path / "source"))
    states = []

    store.delete(
        "owner/module",
        job_id=str(uuid.uuid4()),
        publish=lambda: states.append(_visible_state(store)),
    )

    assert states == ["absent", "absent"]
    assert store.current.list_module_refs() == []
    with pytest.raises(FileNotFoundError):
        store.current.read_record("owner/module")


def test_current_snapshot_is_detached_and_revalidated_before_publication(tmp_path):
    store = _store(tmp_path / "modules")
    record = _install(store, _source(tmp_path / "source"))
    snapshot = tmp_path / "snapshot"

    assert store.current.snapshot("owner/module", snapshot) == record
    assert sorted(path.name for path in snapshot.iterdir()) == [
        "model.safetensors",
        "recipe.json",
    ]
    (snapshot / "model.safetensors").write_bytes(b"snapshot-only")
    assert (
        store.current.artifact_path("owner/module", "model").read_bytes() == b"weights"
    )

    store.current.artifact_path("owner/module", "model").write_bytes(b"corrupt-current")
    with pytest.raises(ValueError, match="does not match installed bytes"):
        store.current.snapshot("owner/module", tmp_path / "second-snapshot")


def test_current_recipe_reader_rejects_bytes_changed_after_install(tmp_path):
    store = _store(tmp_path / "modules")
    _install(store, _source(tmp_path / "source"))
    recipe_path = store.current.module_dir("owner/module") / "recipe.json"
    recipe_path.write_text(json.dumps(_recipe(name="Tampered")), encoding="utf-8")

    with pytest.raises(ValueError, match="installed identity"):
        store.current.load_recipe("owner/module")


def test_referenced_cache_is_reused_and_mark_swept_from_current_records(tmp_path):
    store = _store(tmp_path / "modules")
    referenced = {
        "type": "referenced",
        "ref": "foundation/encoder",
        "revision": "c" * 40,
    }
    source_one = _source(
        tmp_path / "one", recipe=_recipe(weights=referenced, name="One"), weights=None
    )
    source_two = _source(
        tmp_path / "two", recipe=_recipe(weights=referenced, name="Two"), weights=None
    )
    artifact = tmp_path / "encoder.safetensors"
    artifact.write_bytes(b"shared encoder")
    external = {("model", "weights"): artifact}

    first = _install(store, source_one, ref="owner/one", external=external)
    second = _install(store, source_two, ref="owner/two", external=external)

    assert (
        first["resolved_artifacts"][0]["sha256"]
        == second["resolved_artifacts"][0]["sha256"]
    )
    cache = store.root / ".artifacts" / "sha256"
    assert [path.name for path in cache.iterdir()] == [
        first["resolved_artifacts"][0]["sha256"]
    ]
    assert (
        store.current.artifact_path("owner/one", "model").read_bytes()
        == b"shared encoder"
    )

    store.delete("owner/one", job_id=str(uuid.uuid4()), publish=lambda: None)
    assert len(list(cache.iterdir())) == 1
    store.delete("owner/two", job_id=str(uuid.uuid4()), publish=lambda: None)
    assert list(cache.iterdir()) == []


def test_local_working_artifact_resolves_exact_source_module_bytes(tmp_path):
    store = _store(tmp_path / "modules")
    _install(
        store,
        _source(tmp_path / "trained-source", weights=b"trained"),
        ref="local/trained",
        commit=None,
    )
    working = _source(
        tmp_path / "working",
        recipe=_recipe(
            weights={"type": "local", "ref": "local/trained"},
            name="Working",
        ),
        weights=None,
    )
    source_artifact = store.current.artifact_path("local/trained", "model")

    record = _install(
        store,
        working,
        ref="owner/working",
        commit=None,
        external={("model", "weights"): source_artifact},
    )

    assert record["resolved_artifacts"][0]["artifact_source_type"] == "local"
    assert (
        store.current.artifact_path("owner/working", "model").read_bytes() == b"trained"
    )


def test_corrupt_shared_cache_is_never_reused_or_swept_silently(tmp_path):
    store = _store(tmp_path / "modules")
    referenced = {
        "type": "referenced",
        "ref": "foundation/encoder",
        "revision": "c" * 40,
    }
    source = _source(
        tmp_path / "source", recipe=_recipe(weights=referenced), weights=None
    )
    artifact = tmp_path / "encoder.safetensors"
    artifact.write_bytes(b"shared encoder")
    external = {("model", "weights"): artifact}
    record = _install(store, source, external=external)
    cache_path = (
        store.root / ".artifacts" / "sha256" / record["resolved_artifacts"][0]["sha256"]
    )
    cache_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="cache size|cache digest"):
        _install(store, source, ref="owner/second", external=external)


def test_later_attempt_sweeps_orphan_cache_from_failed_private_stage(tmp_path):
    store = _store(tmp_path / "modules")
    referenced = {
        "type": "referenced",
        "ref": "foundation/encoder",
        "revision": "c" * 40,
    }
    source = _source(
        tmp_path / "source", recipe=_recipe(weights=referenced), weights=None
    )
    artifact = tmp_path / "encoder.safetensors"
    artifact.write_bytes(b"shared encoder")
    external = {("model", "weights"): artifact}

    def fail_stage_publication():
        raise RuntimeError("stage publication failed")

    with pytest.raises(RuntimeError, match="stage publication failed"):
        _install(
            store,
            source,
            external=external,
            publish=fail_stage_publication,
        )

    cache = store.root / ".artifacts" / "sha256"
    assert len(list(cache.iterdir())) == 1

    _install(store, _source(tmp_path / "vendored"), ref="owner/vendored")
    assert list(cache.iterdir()) == []


def test_capacity_checks_include_candidate_record_and_shared_cache(tmp_path):
    source = _source(tmp_path / "source", weights=b"1234567890")
    roomy = _store(tmp_path / "roomy")
    record = _install(roomy, source)
    exact_used = roomy.stored_bytes()
    assert exact_used > sum(item["size_bytes"] for item in record["executable_files"])

    too_small_module = _store(
        tmp_path / "small-module", module_cap=9, installed_cap=1_000_000
    )
    with pytest.raises(ValueError, match="max_module_bytes"):
        _install(too_small_module, source)

    too_small_store = _store(
        tmp_path / "small-store",
        module_cap=exact_used,
        installed_cap=exact_used + 1,
    )
    _install(too_small_store, source, ref="owner/first")
    with pytest.raises(ValueError, match="max_installed_bytes"):
        _install(too_small_store, source, ref="owner/second")

    referenced = {
        "type": "referenced",
        "ref": "foundation/encoder",
        "revision": "c" * 40,
    }
    referenced_source = _source(
        tmp_path / "referenced-source",
        recipe=_recipe(weights=referenced),
        weights=None,
    )
    artifact = tmp_path / "large-encoder.safetensors"
    artifact.write_bytes(b"x" * 10_000)
    shared_artifact_store = _store(
        tmp_path / "shared-artifact-store",
        module_cap=2_000,
        installed_cap=1_000_000,
    )
    referenced_record = _install(
        shared_artifact_store,
        referenced_source,
        external={("model", "weights"): artifact},
    )
    assert referenced_record["resolved_artifacts"][0]["size_bytes"] == 10_000


def test_source_inside_store_and_symlinked_namespace_are_rejected(tmp_path):
    root = tmp_path / "modules"
    store = _store(root)
    source = _source(root / "source")
    with pytest.raises(ValueError, match="outside"):
        _install(store, source)

    shutil_source = _source(tmp_path / "outside-source")
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir(exist_ok=True)
    try:
        (root / "owner").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="link|traverses"):
        _install(store, shutil_source)


def test_configured_store_root_may_be_a_platform_mount_link(tmp_path):
    mounted = tmp_path / "mounted-volume"
    mounted.mkdir()
    root = tmp_path / "modules"
    try:
        root.symlink_to(mounted, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    store = _store(root)
    record = _install(store, _source(tmp_path / "source"))

    assert store.current.read_record("owner/module") == record
    assert store.stored_bytes() > 0


def test_workspace_training_result_is_create_only_and_exactly_replayable(tmp_path):
    store = _store(tmp_path / "modules")
    source, ref, attestation = _training_result_source(tmp_path / "result")
    publications = []

    first = store.install_from_directory(
        source,
        module_ref=ref,
        resolved_commit_sha=None,
        job_id=str(uuid.uuid4()),
        publish=lambda: publications.append("published"),
        installed_at=INSTALLED_AT,
        training_result_attestation=attestation,
    )
    replay = store.install_from_directory(
        source,
        module_ref=ref,
        resolved_commit_sha=None,
        job_id=str(uuid.uuid4()),
        publish=lambda: publications.append("unexpected"),
        installed_at="2026-08-15T12:35:00.000Z",
        training_result_attestation=attestation,
    )

    assert replay == first
    assert publications == ["published", "published", "published"]
    assert (store.current.module_dir(ref) / "training-report.json").is_file()

    different, _, different_attestation = _training_result_source(
        tmp_path / "different", weights=b"different"
    )
    with pytest.raises(TrainingResultConflict, match="different training result"):
        store.install_from_directory(
            different,
            module_ref=ref,
            resolved_commit_sha=None,
            job_id=str(uuid.uuid4()),
            publish=lambda: None,
            installed_at=INSTALLED_AT,
            training_result_attestation=different_attestation,
        )
    assert store.current.read_record(ref) == first


def test_workspace_owner_cannot_use_replacing_install_path(tmp_path):
    store = _store(tmp_path / "modules")

    with pytest.raises(ValueError, match="reserved for training results"):
        _install(
            store,
            _source(tmp_path / "source"),
            ref="workspace/arbitrary",
            commit=None,
        )
