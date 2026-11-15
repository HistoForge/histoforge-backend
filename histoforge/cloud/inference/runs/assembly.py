"""Prepare target statistics and assemble results without loading a network."""

from time import monotonic

import modal

from histoforge.cloud._app import app, artifact_volume, image, model_volume
from histoforge.cloud.architectures.registry import model_run_assembly_resources


@app.function(
    image=image,
    volumes={"/modules": model_volume, "/artifacts": artifact_volume},
    cpu=1,
    memory=4096,
    timeout=30 * 60,
    max_containers=4,
    retries=0,
)
@modal.concurrent(max_inputs=1)
def calculate_model_run_percentiles(
    model_run_id: str,
    request: dict,
    input_id: int,
    model_run_limits: dict,
    source_bindings=(),
) -> list[list[float]] | None:
    """Read accepted sources on CPU, checking activity at most once a second.

    One reader decodes at most the existing 32-million-pixel source limit;
    histogram blocks are at most 512 squared target pixels. Four GiB is an
    initial private qualification allocation, not an accepted product capacity.
    The lifecycle still owns cancellation and the absolute run deadline.
    """

    from histoforge.cloud.inference.execution import mounted_inference_stores
    from histoforge.cloud.inference.runs.core_halo_runtime import (
        collect_core_halo_percentiles,
    )
    from histoforge.cloud.inference.runs.executor import _RemoteLifecycle

    lifecycle = _RemoteLifecycle()
    next_check = 0.0

    def is_active():
        nonlocal next_check
        now = monotonic()
        if now < next_check:
            return True
        active = lifecycle.source_delivery(model_run_id)["active"]
        next_check = now + 1.0
        return active

    activation = _installed_activation(request)
    sources, source_files, _files = mounted_inference_stores()
    return collect_core_halo_percentiles(
        request,
        input_id,
        model_run_limits,
        activation,
        sources,
        source_files,
        source_bindings,
        is_active=is_active,
    )


@app.function(
    image=image,
    volumes={"/modules": model_volume, "/artifacts": artifact_volume},
    **model_run_assembly_resources(),
)
@modal.concurrent(max_inputs=1)
def assemble_model_run_input(
    request: dict,
    input_id: int,
    model_run_limits: dict,
    unit_publications: list[dict],
    source_bindings=(),
) -> dict:
    """Publish one complete result; the existing lifecycle accepts or rejects it.

    The expected GPU and runner build identify the predictions, not this CPU worker.
    Installed model and artifact identities are rechecked without loading tensors.
    The assembler still validates all saved cores and their exact coverage.
    """

    from histoforge.cloud.artifacts.store import ArtifactFileStore
    from histoforge.cloud.inference.runs.core_halo_runtime import (
        finalize_core_halo_input,
    )

    activation = _installed_activation(request)
    files = ArtifactFileStore(
        "/artifacts",
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )
    return finalize_core_halo_input(
        request,
        input_id,
        model_run_limits,
        unit_publications,
        activation,
        files,
        source_bindings=source_bindings,
    )


def _installed_activation(request):
    """Resolve exact model metadata on mounted storage without loading tensors."""

    from histoforge.cloud._errors import GatewayFailure
    from histoforge.cloud.inference.activation import (
        activation_failure,
        resolve_installed_model_activation,
    )
    from histoforge.cloud.modules.runtime import current_module_snapshot

    backend = request["expected_backend"]
    module = backend["module"]
    try:
        with current_module_snapshot() as current:
            activation, _weights_path = resolve_installed_model_activation(
                current,
                module["module_ref"],
                backend["model_id"],
                module["content_digest"],
                module.get("resolved_commit_sha"),
                backend["architecture"],
                backend["gpu"],
                backend["runner_build"],
            )
    except GatewayFailure:
        raise
    except Exception as error:
        raise activation_failure(
            error, f"{module['module_ref']}/{backend['model_id']}"
        ) from error
    return activation
