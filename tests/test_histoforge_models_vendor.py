"""The shared HistoForge Models runtime is one exact source snapshot."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys

import numpy as np
import pytest
from histoforge._vendor.histoforge_models.pathology import managed_training
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedSegmentationBatch,
    build_managed_targets,
    managed_segmentation_loss,
)
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud import runtime_identity
from scripts.maintenance.sync_histoforge_models_vendor import (
    DESTINATION_ROOT,
    LICENSE_DESTINATION_PATH,
    MANIFEST_PATH,
    NOTICE_DESTINATION_PATH,
    REPOSITORY_ROOT,
    _verify_import_closure,
    verify_vendored_snapshot,
)


def test_provenance_manifest_and_vendored_bytes_are_exact():
    manifest = verify_vendored_snapshot()

    assert manifest["source_repository"] == (
        "https://github.com/andreibeliaev/histoforge-models.git"
    )
    assert canonical_bytes(manifest) == (REPOSITORY_ROOT / MANIFEST_PATH).read_bytes()
    assert len(manifest["files"]) == 35
    paths = {entry["path"] for entry in manifest["files"]}
    assert "metrics.py" in paths
    assert "pathology/managed_training/targets.py" in paths
    assert "pathology/managed_training/dense_losses.py" in paths
    assert "pathology/managed_training/classification_losses.py" in paths
    assert "pathology/managed_training/optimization.py" not in paths
    assert "pathology/managed_training/trainer.py" not in paths
    assert "pathology/field_geometry.py" in paths
    assert "pathology/window_maps.py" in paths
    assert "pathology/network_execution.py" in paths
    assert "pathology/cellpose_window_decode.py" in paths
    assert "pathology/metric_aggregation.py" in paths
    assert "pathology/optimization.py" in paths
    assert "pathology/class_head.py" in paths
    assert "__init__.py" not in paths  # backend-owned lightweight namespace
    assert not any(path.startswith("pathology/training/") for path in paths)
    assert manifest["license"]["declared_expression"] == ("Apache-2.0 AND BSD-3-Clause")
    assert manifest["notice"]["destination_path"] == (
        NOTICE_DESTINATION_PATH.as_posix()
    )


def test_vendored_drift_is_rejected(tmp_path):
    destination = tmp_path / DESTINATION_ROOT
    destination.parent.mkdir(parents=True)
    shutil.copytree(REPOSITORY_ROOT / DESTINATION_ROOT, destination)
    for copied_path in (LICENSE_DESTINATION_PATH, NOTICE_DESTINATION_PATH):
        target = tmp_path / copied_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / copied_path, target)
    verify_vendored_snapshot(repository_root=tmp_path)

    with (destination / "pathology/managed_training/targets.py").open("ab") as stream:
        stream.write(b"\n# drift\n")

    with pytest.raises(ValueError, match="vendored managed training file differs"):
        verify_vendored_snapshot(repository_root=tmp_path)


@pytest.mark.parametrize(
    ("legal_path", "record_name"),
    [
        (LICENSE_DESTINATION_PATH, "license"),
        (NOTICE_DESTINATION_PATH, "notice"),
    ],
)
def test_vendored_legal_evidence_drift_is_rejected(tmp_path, legal_path, record_name):
    destination = tmp_path / DESTINATION_ROOT
    destination.parent.mkdir(parents=True)
    shutil.copytree(REPOSITORY_ROOT / DESTINATION_ROOT, destination)
    for copied_path in (LICENSE_DESTINATION_PATH, NOTICE_DESTINATION_PATH):
        target = tmp_path / copied_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / copied_path, target)

    with (tmp_path / legal_path).open("ab") as stream:
        stream.write(b"\ndrift\n")

    with pytest.raises(
        ValueError, match=f"vendored managed training {record_name} differs"
    ):
        verify_vendored_snapshot(repository_root=tmp_path)


def test_vendored_package_has_only_declared_external_dependencies():
    package_root = REPOSITORY_ROOT / DESTINATION_ROOT / "pathology/managed_training"
    allowed_external = {"fastremap", "numpy", "scipy", "torch"}
    violations = []
    for source_path in sorted(package_root.glob("*.py")):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").partition(".")[0]
                if node.level > 1 or (
                    node.level == 0
                    and root not in sys.stdlib_module_names
                    and root not in allowed_external
                ):
                    violations.append((source_path.name, node.lineno))
            elif isinstance(node, ast.Import):
                roots = {alias.name.partition(".")[0] for alias in node.names}
                if any(
                    root not in sys.stdlib_module_names and root not in allowed_external
                    for root in roots
                ):
                    violations.append((source_path.name, node.lineno))

    assert violations == []


def test_shared_closure_rejects_an_omitted_standalone_dependency():
    with pytest.raises(ValueError, match="import leaves shared runtime"):
        _verify_import_closure(
            {
                "pathology/flow_inference.py": b"from ..models import CellposeModel\n",
            }
        )


@pytest.mark.parametrize("namespace", ["histoforge_models", "cellpose"])
@pytest.mark.parametrize(
    "statement", ["from {namespace} import core", "import {namespace}.core"]
)
def test_shared_closure_rejects_an_absolute_source_namespace(namespace, statement):
    with pytest.raises(ValueError, match="absolute scientific package import"):
        _verify_import_closure(
            {
                "pathology/flow_inference.py": statement.format(
                    namespace=namespace
                ).encode(),
            }
        )


def test_vendored_package_imports_without_standalone_scientific_packages(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(REPOSITORY_ROOT)!r})
from histoforge._vendor.histoforge_models.pathology import managed_training
assert managed_training.__name__.startswith('histoforge._vendor.histoforge_models.')
assert not any(
    name.partition('.')[0] in ('cellpose', 'histoforge_models')
    for name in sys.modules
)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        check=True,
    )


def test_vendored_package_exposes_api_and_builds_both_target_families():
    assert set(managed_training.__all__) == {
        "ClassificationObjective",
        "FlowLossWeighting",
        "HeadType",
        "ManagedDenseBatch",
        "ManagedLossResult",
        "ManagedObjective",
        "ManagedSegmentationBatch",
        "ManagedSegmentationTargets",
        "augment_blur",
        "augment_brightness_contrast",
        "augment_hed",
        "augment_noise",
        "build_managed_targets",
        "managed_dense_loss",
        "managed_segmentation_loss",
        "validate_managed_loss_configuration",
    }
    labels = np.zeros((9, 11), dtype=np.int32)
    labels[2:7, 3:9] = 7
    for head_type in ("flow", "hovernet"):
        targets = build_managed_targets(labels, head_type=head_type)
        assert targets.directional.shape == (2, 9, 11)
        assert np.array_equal(targets.foreground, labels > 0)


def test_vendored_scientific_loss_backpropagates_one_field():
    import torch
    from torch import nn

    model = nn.Conv2d(3, 3, kernel_size=1)
    support = torch.ones((1, 4, 4), dtype=torch.bool)
    batch = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 4, 4)),
        directional_targets=torch.zeros((1, 2, 4, 4)),
        foreground_targets=torch.zeros((1, 4, 4)),
        supervision_mask=support,
        real_pixel_mask=support,
    )

    batch.validate(0)
    result = managed_segmentation_loss(
        model(batch.images),
        batch,
        head_type="flow",
        class_count=0,
    )

    assert result.field_count == 1
    assert torch.isfinite(result.total_sum)
    result.total_sum.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_runtime_source_identity_includes_vendor_evidence():
    source_paths = {path for path, _content in runtime_identity._source_files()}

    assert "histoforge/_vendor/histoforge_models/LICENSE" in source_paths
    assert "histoforge/_vendor/histoforge_models/NOTICE" in source_paths
    assert "histoforge/_vendor/histoforge_models/provenance.json" in source_paths
