"""Assemble saved processing-area results without loading a prediction network."""

import modal

from histoforge.cloud._app import app, artifact_volume, image, model_volume
from histoforge.cloud.architectures.registry import model_run_assembly_resources


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

    from histoforge.cloud._errors import GatewayFailure
    from histoforge.cloud.artifacts.store import ArtifactFileStore
    from histoforge.cloud.inference.activation import (
        activation_failure,
        resolve_installed_model_activation,
    )
    from histoforge.cloud.inference.runs.core_halo_runtime import (
        finalize_core_halo_input,
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
