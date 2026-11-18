"""The local deployment helper must not require the cloud scientific stack."""

import subprocess
import sys


def test_deployment_graph_import_stays_within_bundled_cli_dependencies():
    script = r"""
import os
import sys
from importlib.abc import MetaPathFinder
from types import SimpleNamespace

unbundled_roots = {
    "cv2",
    "fastapi",
    "fastremap",
    "huggingface_hub",
    "httpx",
    "imagecodecs",
    "numcodecs",
    "numpy",
    "openslide",
    "rasterio",
    "safetensors",
    "scipy",
    "shapely",
    "skimage",
    "tifffile",
    "timm",
    "torch",
    "zarr",
}


class RejectUnbundledDependency(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in unbundled_roots:
            raise ModuleNotFoundError(
                f"deployment graph imported unbundled dependency {fullname}"
            )
        return None


sys.meta_path.insert(0, RejectUnbundledDependency())
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

from histoforge import cli

cli.cmd_self_check(SimpleNamespace(json=True))
assert "truststore" in sys.modules
assert unbundled_roots.isdisjoint(
    name.partition(".")[0] for name in sys.modules
)
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_claimed_worker_imports_trainer_before_loading_model():
    script = r"""
import sys
from types import SimpleNamespace

from histoforge.cloud.training.runs import executor as training_executor
from histoforge.cloud.training.runs import source_model
from histoforge.cloud.training.implementations import DINOV3_INSTANCE_TRAINER_ID

trainer_module = "histoforge.cloud.training.trainers.instance.loop"
assert trainer_module not in sys.modules


class ExpectedStop(Exception):
    pass


class DatasetFiles:
    def read_dataset(self, dataset_id, limits):
        return SimpleNamespace(manifest={})


class SourceFiles:
    def read_source(self, run_id):
        return object()


def load_model(*args):
    assert trainer_module in sys.modules
    raise ExpectedStop


training_executor.read_running_training_report = lambda *args, **kwargs: {}
training_executor.validate_training_run_request_against_manifest_capability = (
    lambda *args: None
)
source_model.load_training_source_model = load_model

try:
    training_executor.train_and_publish_claimed_run(
        {"run_id": "run-id", "request": {"dataset_id": "dataset-id"}},
        SimpleNamespace(
            architecture="dinov3-adapter-cellpose",
            capability={}, trainer_implementation_id=DINOV3_INSTANCE_TRAINER_ID
        ),
        dataset_files=DatasetFiles(),
        source_files=SourceFiles(),
        run_files=object(),
        result_files=object(),
        control=object(),
        device="cpu",
        api_limits={},
        source_limits={},
    )
except ExpectedStop:
    pass
else:
    raise AssertionError("model loading did not reach the ordering witness")
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_dense_cold_entrypoint_initializes_targets_before_torch():
    script = r"""
import sys
from importlib.abc import MetaPathFinder
from histoforge.cloud.training.bindings.dinov3_dense import execute_dense_training
from histoforge.cloud.training.runs import source_model

assert "torch" not in sys.modules

class RequireTargetOrder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch":
            assert "fastremap" in sys.modules
            package = "histoforge._vendor.histoforge_models.pathology.managed_training"
            assert package + ".targets" in sys.modules
        return None

class ExpectedStop(Exception):
    pass

def stop_before_io(*args):
    assert "histoforge.cloud.training.trainers.dense.loop" in sys.modules
    assert "torch" in sys.modules
    raise ExpectedStop

sys.meta_path.insert(0, RequireTargetOrder())
source_model.load_training_source_model = stop_before_io
try:
    execute_dense_training(None, run={}, report={}, dataset=None, source=None,
                           device="cpu", api_limits={}, source_limits={},
                           epoch_completed=None)
except ExpectedStop:
    pass
else:
    raise AssertionError("cold entrypoint did not reach the source loader")
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
