"""One serialized source/run reference boundary on the Artifact Volume.

Source and model-run stores retain their domain policies. This lane orders source
release, run admission/reuse, publications and expired-run collection together.
Upload reception, source assembly, model work and transfers remain outside it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

import modal

from histoforge.cloud._app import app, artifact_volume, image
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure

if TYPE_CHECKING:
    from histoforge.cloud.artifacts.store import ArtifactFileStore
    from histoforge.cloud.inference.runs.store import ModelRunStore
    from histoforge.cloud.sources.files import SourceFileStore
    from histoforge.cloud.sources.store import SourceArtifactStore

_ARTIFACT_ROOT = "/artifacts"


@app.cls(
    image=image,
    timeout=300,
    volumes={_ARTIFACT_ROOT: artifact_volume},
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
class ArtifactLifecycleLane:
    """Order mutations of source and inference-run artifact references."""

    @modal.method()
    def reserve(
        self,
        deployment_id: str,
        auth_revision: int,
        start_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Reserve one source ID after validating the current auth generation."""

        store, auth = _sources()
        _authorize(auth, deployment_id, auth_revision)
        result = store.reserve(start_request, files=_source_files())
        _authorize(auth, deployment_id, auth_revision)
        return result

    @modal.method()
    def release(
        self,
        deployment_id: str,
        auth_revision: int,
        artifact_id: str,
    ) -> None:
        """Release one source ID while excluding concurrent reserve/release."""

        store, auth = _sources()
        model_runs = _model_runs()
        _authorize(auth, deployment_id, auth_revision)
        store.release(
            artifact_id,
            _source_files(),
            has_durable_reference=model_runs.has_source_reference,
        )
        _authorize(auth, deployment_id, auth_revision)

    @modal.method()
    def reserve_model_run(
        self,
        deployment_id: str,
        auth_revision: int,
        request: dict[str, Any],
        model_run_limits: dict[str, Any],
        terminal_retention_seconds: int,
    ) -> dict[str, Any]:
        """Atomically fence source release against one durable-run start."""

        source_store, auth = _sources()
        model_runs = _model_runs()
        files = _artifact_files()
        _authorize(auth, deployment_id, auth_revision)
        accepted, should_dispatch = model_runs.reserve(
            request,
            model_run_limits,
            terminal_retention_seconds,
            files,
            require_sources_live_through=lambda submitted, deadline: (
                _require_sources_live_through(source_store, submitted, deadline)
            ),
            require_bindings_live_through=lambda submitted, bindings, deadline: (
                _require_bound_sources_live_through(
                    source_store, submitted, bindings, deadline
                )
            ),
        )
        _authorize(auth, deployment_id, auth_revision)
        return {"accepted": accepted, "should_dispatch": should_dispatch}

    @modal.method()
    def bind_model_run_source(
        self,
        deployment_id: str,
        auth_revision: int,
        model_run_id: str,
        binding: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach a sealed image without racing source release or run termination."""

        sources, auth = _sources()
        _authorize(auth, deployment_id, auth_revision)
        runs = _model_runs()
        files = _artifact_files()
        result = runs.bind_source(
            model_run_id,
            binding,
            files,
            require_source=lambda request, item, deadline: (
                _require_bound_sources_live_through(sources, request, [item], deadline)
            ),
        )
        _authorize(auth, deployment_id, auth_revision)
        return result

    @modal.method()
    def model_run_sources(
        self, deployment_id: str, auth_revision: int, model_run_id: str
    ) -> dict[str, Any]:
        """Return accepted image plans and bindings, including after failure."""

        auth = _auth()
        _authorize(auth, deployment_id, auth_revision)
        result = _model_runs().source_delivery(model_run_id, _artifact_files())
        _authorize(auth, deployment_id, auth_revision)
        return result

    @modal.method()
    def source_delivery(self, model_run_id: str) -> dict[str, Any]:
        """Read upload readiness for the CPU coordinator without acquiring a GPU."""

        return _model_runs().source_delivery(model_run_id, _artifact_files())

    @modal.method()
    def report_source_failure(
        self,
        deployment_id: str,
        auth_revision: int,
        model_run_id: str,
        failure: dict[str, Any],
    ) -> None:
        """Report failed image delivery through the existing failure transition."""

        auth = _auth()
        _authorize(auth, deployment_id, auth_revision)
        _model_runs().fail(model_run_id, failure, _artifact_files())
        _authorize(auth, deployment_id, auth_revision)

    @modal.method()
    def claim_chunk(
        self,
        deployment_id: str,
        auth_revision: int,
        artifact_id: str,
        chunk_id: int,
        body_size: int,
        digest: str,
    ) -> dict[str, Any]:
        """Claim one received range without holding the lane during byte IO."""

        store, auth = _sources()
        return store.claim_chunk(
            artifact_id,
            chunk_id,
            body_size,
            digest,
            authorize=lambda: _authorize(auth, deployment_id, auth_revision),
        )

    @modal.method()
    def commit_chunk(
        self,
        deployment_id: str,
        auth_revision: int,
        artifact_id: str,
        chunk_id: int,
        digest: str,
        start_request_digest: str,
    ) -> dict[str, Any]:
        """Publish one chunk acknowledgement after its Volume commit."""

        store, auth = _sources()
        return store.commit_chunk(
            artifact_id,
            chunk_id,
            digest,
            start_request_digest,
            authorize=lambda: _authorize(auth, deployment_id, auth_revision),
        )

    @modal.method()
    def claim_seal(
        self,
        deployment_id: str,
        auth_revision: int,
        artifact_id: str,
        seal_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Claim one seal without holding the lane during assembly or validation."""

        store, auth = _sources()
        return store.claim_seal(
            artifact_id,
            seal_request,
            authorize=lambda: _authorize(auth, deployment_id, auth_revision),
        )

    @modal.method()
    def commit_seal(
        self,
        deployment_id: str,
        auth_revision: int,
        artifact_id: str,
        seal_request: dict[str, Any],
        start_request_digest: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish one sealed reference after final bytes are committed."""

        store, auth = _sources()
        return store.commit_seal(
            artifact_id,
            seal_request,
            start_request_digest,
            manifest,
            authorize=lambda: _authorize(auth, deployment_id, auth_revision),
        )

    @modal.method()
    def begin_execution(self, model_run_id: str) -> dict[str, Any] | None:
        """Move a retained run to RUNNING and return its exact context."""

        return _model_runs().begin_execution(model_run_id, _artifact_files())

    @modal.method()
    def publish_unit(self, model_run_id: str, publication: dict[str, Any]) -> bool:
        """Publish one complete core/halo reusable unit."""

        store = _model_runs()
        files = _artifact_files()
        return _publish_or_discard(
            lambda: store.publish_unit(model_run_id, publication, files),
            publication.get("reusable_state"),
            store,
            files,
        )

    @modal.method()
    def publish_checkpoint(
        self, model_run_id: str, publication: dict[str, Any]
    ) -> bool:
        """Replace one global unit's checkpoint only with its highest sequence."""

        store = _model_runs()
        files = _artifact_files()
        return _publish_or_discard(
            lambda: store.publish_checkpoint(model_run_id, publication, files),
            publication.get("state"),
            store,
            files,
        )

    @modal.method()
    def publish_unit_failure(
        self, model_run_id: str, publication: dict[str, Any]
    ) -> bool:
        """Publish one coded unit failure while the run remains active."""

        return _model_runs().publish_unit_failure(
            model_run_id, publication, _artifact_files()
        )

    @modal.method()
    def publish_input(
        self,
        model_run_id: str,
        publication: dict[str, Any],
        decoded_bytes: int,
    ) -> bool:
        """Publish a complete biological input after its stream is committed."""

        store = _model_runs()
        files = _artifact_files()
        return _publish_or_discard(
            lambda: store.publish_input(
                model_run_id, publication, decoded_bytes, files
            ),
            publication.get("feature_stream_artifact"),
            store,
            files,
        )

    @modal.method()
    def complete(self, model_run_id: str) -> bool:
        """Publish COMPLETE when every requested input is already durable."""

        return _model_runs().complete(model_run_id, _artifact_files())

    @modal.method()
    def fail(self, model_run_id: str, failure: dict[str, Any]) -> bool:
        """Choose COMPLETE, PARTIAL or FAILED from already published results."""

        return _model_runs().fail(model_run_id, failure, _artifact_files())

    @modal.method()
    def cancel(
        self,
        deployment_id: str,
        auth_revision: int,
        model_run_id: str,
    ) -> None:
        """Apply authenticated idempotent best-effort cancellation."""

        auth = _auth()
        _authorize(auth, deployment_id, auth_revision)
        _model_runs().cancel(model_run_id, _artifact_files())
        _authorize(auth, deployment_id, auth_revision)

    @modal.method()
    def status(
        self,
        deployment_id: str,
        auth_revision: int,
        model_run_id: str,
    ) -> dict[str, Any]:
        """Return authenticated status after finalizing an observed deadline."""

        auth = _auth()
        _authorize(auth, deployment_id, auth_revision)
        result = _model_runs().status(model_run_id, _artifact_files())
        _authorize(auth, deployment_id, auth_revision)
        return result

    @modal.method()
    def deployment_barrier(self) -> None:
        """Return after every earlier source/run reference mutation has finished."""

    @modal.method()
    def cleanup_expired(self) -> None:
        """Remove expired run files, then uploads, using the existing deadlines."""

        # Serialize cleanup with admission and publication so a replacement run
        # cannot adopt files while they are being removed.
        _model_runs().cleanup_expired(_artifact_files())
        sources, _ = _sources()
        sources.cleanup_expired(_source_files())


@app.function(
    image=image,
    schedule=modal.Period(hours=1),
    timeout=300,
    max_containers=1,
    retries=0,
)
def cleanup_expired_inference_files() -> None:
    """Reclaim expired inference files even when no new work is submitted."""

    # Wait for the existing storage owner; this CPU-only trigger mounts no files.
    # Failures remain visible in Modal and are retried at the next scheduled run.
    ArtifactLifecycleLane().cleanup_expired.remote()


def _sources() -> tuple[SourceArtifactStore, Any]:
    from histoforge.cloud.sources.store import SourceArtifactStore

    records = modal.Dict.from_name(
        DEPLOYMENT_CONFIG.source_artifacts_dict_name,
        create_if_missing=True,
    )
    auth = modal.Dict.from_name(
        DEPLOYMENT_CONFIG.gateway_state_dict_name,
        create_if_missing=True,
    )
    return (
        SourceArtifactStore(
            records,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            DEPLOYMENT_CONFIG.limits.wire_value(),
            max_active_source_artifact_peak_bytes=(
                DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
            ),
        ),
        auth,
    )


def _source_files() -> SourceFileStore:
    from histoforge.cloud.sources.files import SourceFileStore

    return SourceFileStore(
        _ARTIFACT_ROOT,
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )


def _model_runs() -> ModelRunStore:
    from histoforge.cloud.inference.runs.store import ModelRunStore

    records = modal.Dict.from_name(
        DEPLOYMENT_CONFIG.model_runs_dict_name,
        create_if_missing=True,
    )
    return ModelRunStore(
        records,
        max_active_runs=DEPLOYMENT_CONFIG.limits.max_active_model_runs,
    )


def _require_sources_live_through(
    source_store: SourceArtifactStore,
    request: Mapping[str, Any],
    required_until: datetime,
) -> None:
    """Require each uploaded image to remain available through the run deadline."""

    for item in request["inputs"]:
        reference = item["raster_input"]["source_view"].get("artifact")
        if reference is None:
            continue
        resolved = source_store.resolve_sealed_reference(reference)
        _require_source_lifetime(resolved, required_until)


def _require_bound_sources_live_through(
    source_store, request, bindings, required_until
):
    inputs = {item["input_id"]: item for item in request["inputs"]}
    containers = _source_containers(request)
    for binding in bindings:
        _require_bound_source(
            source_store,
            inputs[binding["input_id"]],
            binding["source"],
            required_until,
            containers,
        )


def _source_containers(request):
    from histoforge.cloud.architectures.registry import runner_registry

    return runner_registry()["definitions"][
        request["expected_backend"]["architecture"]
    ]["capabilities"]["source_containers"]


def _require_bound_source(
    source_store, model_input, binding, required_until, containers
):
    from histoforge.cloud.rasterization.captured_source import (
        validate_captured_source_upload,
    )

    resolved = source_store.resolve_sealed_reference(binding["artifact"])
    try:
        validate_captured_source_upload(
            model_input["raster_input"]["source_view"]["capture"],
            binding,
            resolved,
            containers,
        )
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error
    _require_source_lifetime(resolved, required_until)


def _require_source_lifetime(resolved, required_until):
    expires_at = datetime.fromisoformat(
        resolved["absolute_expires_at"].removesuffix("Z") + "+00:00"
    )
    if required_until > expires_at:
        raise GatewayFailure(
            GatewayErrorCode.ARTIFACT_PRECONDITION_FAILED,
            "uploaded image expires before the model run's execution deadline",
            resource_id=resolved["artifact"]["artifact_id"],
        )


def _authorize(auth: Any, deployment_id: str, auth_revision: int) -> None:
    from histoforge.cloud._auth import require_current_auth_generation

    require_current_auth_generation(auth, deployment_id, auth_revision)


def _artifact_files() -> ArtifactFileStore:
    from histoforge.cloud.artifacts.store import ArtifactFileStore

    return ArtifactFileStore(
        _ARTIFACT_ROOT,
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )


def _auth() -> Any:
    return modal.Dict.from_name(
        DEPLOYMENT_CONFIG.gateway_state_dict_name,
        create_if_missing=True,
    )


def _publish_or_discard(
    operation: Callable[[], bool],
    reference: Any,
    store: ModelRunStore,
    files: ArtifactFileStore,
) -> bool:
    """Discard exactly one publication artifact when its control cut loses."""

    try:
        accepted = operation()
    except Exception:
        _discard_best_effort(reference, store, files)
        raise
    if not accepted:
        _discard_best_effort(reference, store, files)
    return accepted


def _discard_best_effort(
    reference: Any,
    store: ModelRunStore,
    files: ArtifactFileStore,
) -> None:
    """Keep cleanup failure from replacing the authoritative publish outcome."""

    try:
        store.discard_unreferenced_artifact(reference, files)
    except Exception:
        pass
