"""The Modal app, image, and volume — the deployable cloud unit's foundation."""

import logging
from pathlib import Path

import modal

from histoforge.cloud._config import DEPLOYMENT_CONFIG

# Configure coarse inference stages once, without changing host/root or training
# logging. Each record reaches one handler, including after module re-import.
for _logger_name in (
    "histoforge.cloud.architectures",
    "histoforge.cloud.inference",
    "histoforge._vendor.histoforge_models.pathology",
):
    _stage_logger = logging.getLogger(_logger_name)
    _stage_logger.setLevel(logging.INFO)
    _stage_logger.propagate = False
    if not _stage_logger.handlers:
        _stage_logger.addHandler(logging.StreamHandler())

app = modal.App(DEPLOYMENT_CONFIG.gateway_app_name)

# The deployment helper scopes this override to HistoForge's image build. This
# builder mounts Modal's supporting dependencies outside our locked environment.
IMAGE_BUILDER_VERSION = "2025.06"

_RUNTIME_DIR = Path(__file__).with_name("runtime")
_CONDA_LOCK = _RUNTIME_DIR / "conda-linux-64.lock"
_PIP_LOCK = _RUNTIME_DIR / "requirements-modal.lock"
_VENDORED_MODELS_ROOT = Path(__file__).parents[1] / "_vendor" / "histoforge_models"
_VENDORED_MODELS_PROVENANCE = _VENDORED_MODELS_ROOT / "provenance.json"
_VENDORED_MODELS_LICENSE = _VENDORED_MODELS_ROOT / "LICENSE"
_VENDORED_MODELS_NOTICE = _VENDORED_MODELS_ROOT / "NOTICE"

# Create the locked environment in a fresh prefix. Installing an explicit lock
# over a prepopulated base can leave conflicting package metadata behind. The
# registry image contains micromamba but no Python environment to overlay.
image = (
    modal.Image.from_registry(
        "mambaorg/micromamba:2.1.1-debian12-slim",
        setup_dockerfile_commands=["USER root"],
    )
    .add_local_file(
        _CONDA_LOCK,
        "/root/histoforge/cloud/runtime/conda-linux-64.lock",
        copy=True,
    )
    .run_commands(
        "micromamba create --yes --name histoforge "
        "--file /root/histoforge/cloud/runtime/conda-linux-64.lock",
    )
    # Activate only after creation, in both the inherited build shell and
    # runtime entrypoint. PATH also selects this Python for container exec.
    .env({"ENV_NAME": "histoforge", "MAMBA_DOCKERFILE_ACTIVATE": "1"})
    .dockerfile_commands("ENV PATH=/opt/conda/envs/histoforge/bin:$PATH")
    # Modal sets SSL_CERT_DIR to a base-image directory that does not contain
    # the issuer chain used by Hugging Face. Prefer the hash-pinned conda CA
    # bundle explicitly; HTTPX otherwise ignores certifi when that directory
    # environment variable is present.
    .env({"SSL_CERT_FILE": "/opt/conda/envs/histoforge/ssl/cert.pem"})
    .pip_install_from_requirements(
        str(_PIP_LOCK),
        extra_options="--require-hashes --no-deps",
    )
    .run_commands(
        'python -c "import sys; '
        "assert sys.prefix == '/opt/conda/envs/histoforge', sys.prefix; "
        "import torch; print('CUDA:', torch.cuda.is_available());"
        " print('torch:', torch.__version__)\"",
    )
    .add_local_file(
        _PIP_LOCK,
        "/root/histoforge/cloud/runtime/requirements-modal.lock",
        copy=True,
    )
    .add_local_file(
        _RUNTIME_DIR / "reader-runtime.json",
        "/root/histoforge/cloud/runtime/reader-runtime.json",
        copy=True,
    )
    .add_local_file(
        _RUNTIME_DIR / "install_readers.py",
        "/root/histoforge/cloud/runtime/install_readers.py",
        copy=True,
    )
    .run_commands("python /root/histoforge/cloud/runtime/install_readers.py")
    .add_local_file(
        _VENDORED_MODELS_PROVENANCE,
        "/root/histoforge/_vendor/histoforge_models/provenance.json",
        copy=True,
    )
    .add_local_file(
        _VENDORED_MODELS_LICENSE,
        "/root/histoforge/_vendor/histoforge_models/LICENSE",
        copy=True,
    )
    .add_local_file(
        _VENDORED_MODELS_NOTICE,
        "/root/histoforge/_vendor/histoforge_models/NOTICE",
        copy=True,
    )
    # Runtime-mounted local source must be the final layer: Modal rejects any
    # later build step after add_local_python_source(copy=False). Ship the
    # complete package by import name without CWD-relative path handling.
    .add_local_python_source("histoforge")
)

# Module weights + recipes on a Modal Volume (mounted at /modules; too large
# to bake into the image). Resource name matches the mount for consistency.
model_volume = modal.Volume.from_name(
    DEPLOYMENT_CONFIG.modules_volume_name,
    create_if_missing=True,
    version=1,
)

# Source files and immutable computation artifacts use Volume v2 because their
# distinct immutable paths are published by independent gateway/runner
# containers. Every consequential publisher still commits before exposing a
# Dict reference.
artifact_volume = modal.Volume.from_name(
    DEPLOYMENT_CONFIG.artifact_volume_name,
    create_if_missing=True,
    version=2,
)

# Registry credentials stay in Modal. Only the module-administration class
# receives this Secret; gateway and model-runner containers do not.
huggingface_secret = modal.Secret.from_name(
    DEPLOYMENT_CONFIG.huggingface_secret_name,
    required_keys=["HF_TOKEN"],
)
