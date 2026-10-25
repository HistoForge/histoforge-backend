"""The only Modal execution lane allowed to mutate the Module Volume.

The class topology serializes install, publish, optional local sync, delete, and
training-result reconciliation across one admitted input and at most one
container. Pure adapters and :class:`ModuleStore` own their respective
algorithms; this file only sequences them through the sole Module Volume writer.
"""

from __future__ import annotations

import os
from collections.abc import Callable, MutableMapping
from typing import Any

import modal

from histoforge.cloud._app import (
    app,
    artifact_volume,
    huggingface_secret,
    image,
    model_volume,
)
from histoforge.cloud._config import DEPLOYMENT_CONFIG


@app.cls(
    image=image,
    secrets=[huggingface_secret],
    timeout=1800,
    volumes={"/modules": model_volume, "/artifacts": artifact_volume},
    max_containers=1,
    single_use_containers=True,
    retries=0,
)
@modal.concurrent(max_inputs=1)
class ModuleMutationLane:
    """Serialize every operation that may change current module state."""

    @modal.method()
    def deployment_barrier(self) -> None:
        """Return only after mutations admitted before this call have finished."""

    @modal.method()
    def run(
        self,
        job_id: str,
        job_type: str,
        payload: dict[str, Any],
        deployment_id: str,
        auth_revision: int,
    ) -> None:
        from histoforge.cloud.modules.runtime import mounted_module_store

        jobs = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.admin_jobs_dict_name, create_if_missing=True
        )
        auth = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.gateway_state_dict_name, create_if_missing=True
        )

        def authorize() -> None:
            from histoforge.cloud._auth import require_current_auth_generation

            require_current_auth_generation(auth, deployment_id, auth_revision)

        model_volume.reload()
        execute_admin_job(
            mounted_module_store(),
            jobs,
            job_id,
            job_type,
            payload,
            model_volume.commit,
            hf_token=os.environ.get("HF_TOKEN"),
            authorize=authorize,
        )

    @modal.method()
    def delete(
        self,
        job_id: str,
        module_ref: str,
        deployment_id: str,
        auth_revision: int,
    ) -> None:
        from histoforge.cloud._auth import require_current_auth_generation
        from histoforge.cloud.modules.runtime import mounted_module_store

        auth = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.gateway_state_dict_name, create_if_missing=True
        )
        require_current_auth_generation(auth, deployment_id, auth_revision)
        model_volume.reload()
        mounted_module_store().delete(
            module_ref,
            job_id=job_id,
            publish=model_volume.commit,
        )

    @modal.method()
    def finalize_training_result(
        self,
        run_id: str,
        deployment_id: str,
        auth_revision: int,
    ) -> dict[str, Any]:
        """Reconcile one frozen result candidate through the sole module writer."""

        from histoforge.cloud._auth import require_current_auth_generation
        from histoforge.cloud.modules.runtime import mounted_module_store
        from histoforge.cloud.training.runs.files import TrainingRunFiles
        from histoforge.cloud.training.runs.finalizer import finalize_training_result
        from histoforge.cloud.training.runs.result_files import TrainingResultFiles
        from histoforge.cloud.training.state_client import TrainingStateClient

        auth = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.gateway_state_dict_name, create_if_missing=True
        )
        require_current_auth_generation(auth, deployment_id, auth_revision)
        model_volume.reload()
        api_limits = DEPLOYMENT_CONFIG.training_api_limits.wire_value()
        volume = {
            "reload_volume": artifact_volume.reload,
            "commit_volume": artifact_volume.commit,
        }
        return finalize_training_result(
            mounted_module_store(),
            TrainingRunFiles("/artifacts", api_limits, **volume),
            TrainingResultFiles(
                "/artifacts",
                api_limits,
                DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes,
                **volume,
            ),
            TrainingStateClient(
                {"deployment_id": deployment_id, "auth_revision": auth_revision}
            ),
            run_id,
            model_volume.commit,
            api_limits,
        )


def execute_admin_job(
    store,
    jobs: MutableMapping[str, Any],
    job_id: str,
    job_type: str,
    payload: dict[str, Any],
    publish_volume: Callable[[], None],
    *,
    hf_token: str | None,
    authorize: Callable[[], None] = lambda: None,
) -> None:
    """Execute one accepted job and publish exactly one typed terminal outcome."""

    from histoforge.cloud._errors import gateway_error_from_exception
    from histoforge.cloud.admin_jobs import store as admin_jobs
    from histoforge.cloud.admin_jobs.contract import installed_result, published_result

    try:
        admin_jobs.transition_running(jobs, job_id)
        authorize()
        if job_type not in {"INSTALL", "PUBLISH"}:
            raise RuntimeError(f"unsupported module mutation type: {job_type!r}")
        token = _require_huggingface_token(hf_token)
        if job_type == "INSTALL":
            record = _execute_install(
                store,
                job_id,
                payload,
                publish_volume,
                token,
            )
            admin_jobs.transition_succeeded(jobs, job_id, installed_result(record))
        elif job_type == "PUBLISH":
            publication = _execute_publish(
                store,
                job_id,
                payload,
                publish_volume,
                token,
            )
            result = published_result(
                payload["target_repository"],
                publication.commit_sha,
                sync_requested=payload["sync_locally"],
                local_record=publication.local_record,
                local_sync_error=publication.local_sync_error,
            )
            if publication.local_sync_error is None:
                admin_jobs.transition_succeeded(jobs, job_id, result)
            else:
                admin_jobs.transition_partial(jobs, job_id, result)
    except Exception as error:
        admin_jobs.transition_failed(
            jobs,
            job_id,
            gateway_error_from_exception(error),
        )


def _require_huggingface_token(value: str | None) -> str:
    """Return the server credential without ever accepting a client fallback."""

    if value is None or not value.strip():
        from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure

        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "server Hugging Face credential is not configured",
        )
    return value


def _execute_install(store, job_id, payload, publish_volume, hf_token):
    from histoforge.cloud.modules.install._installer import (
        download_module_snapshot,
        remove_download,
    )

    download = download_module_snapshot(
        payload["module_ref"],
        payload.get("hugging_face_revision"),
        hf_token,
        max_module_bytes=store.max_module_bytes,
    )
    try:
        record = store.install_from_directory(
            download.directory,
            module_ref=download.module_ref,
            resolved_commit_sha=download.resolved_commit_sha,
            job_id=job_id,
            publish=publish_volume,
        )
    finally:
        remove_download(download)
    return record


def _execute_publish(store, job_id, payload, publish_volume, hf_token):
    from histoforge.cloud.modules.publish._publisher import publish_module

    return publish_module(
        store,
        source_ref=payload["source_module"],
        target_repo=payload["target_repository"],
        visibility=payload["visibility"],
        card_markdown=payload["card_markdown"],
        sync_local=payload["sync_locally"],
        token=hf_token,
        job_id=job_id,
        publish_volume=publish_volume,
    )
