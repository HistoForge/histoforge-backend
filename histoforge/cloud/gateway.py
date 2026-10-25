"""FastAPI composition for the authenticated HistoForge gateway."""

import modal

from histoforge.cloud._app import app, artifact_volume, image, model_volume
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.sources.client import SourceArtifactLifecycleClient
from histoforge.cloud.training.state_client import TrainingStateClient

_AUTH_STATE_SCOPE_KEY = "histoforge.auth_state"


class _GatewayAdmission:
    """Authenticate every protected request before any body can be parsed."""

    def __init__(self, app, auth) -> None:
        self._app = app
        self._auth = auth

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if (
            scope.get("type") != "http"
            or (path != "/v1" and not path.startswith("/v1/"))
            or path == "/v1/health"
        ):
            await self._app(scope, receive, send)
            return

        from fastapi.responses import JSONResponse
        from starlette.concurrency import run_in_threadpool

        from histoforge.cloud._auth import require_authenticated_request
        from histoforge.cloud._errors import (
            GatewayFailure,
            gateway_error_http_status,
            unexpected_gateway_failure,
        )

        try:
            state = await run_in_threadpool(
                require_authenticated_request,
                scope.get("headers", []),
                self._auth,
            )
        except GatewayFailure as failure:
            response = JSONResponse(
                status_code=gateway_error_http_status(failure.error["code"]),
                content=failure.error,
            )
            await response(scope, receive, send)
            return
        except Exception as error:
            failure = unexpected_gateway_failure(error)
            response = JSONResponse(
                status_code=gateway_error_http_status(failure.error["code"]),
                content=failure.error,
            )
            await response(scope, receive, send)
            return

        scope[_AUTH_STATE_SCOPE_KEY] = state
        started = False

        async def send_tracked(message):
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self._app(scope, receive, send_tracked)
        except GatewayFailure as failure:
            if started:
                raise
            response = JSONResponse(
                status_code=gateway_error_http_status(failure.error["code"]),
                content=failure.error,
            )
            await response(scope, receive, send)


@app.function(
    image=image,
    memory=DEPLOYMENT_CONFIG.gateway_memory_mib,
    timeout=DEPLOYMENT_CONFIG.gateway_timeout_seconds,
    volumes={"/modules": model_volume, "/artifacts": artifact_volume},
)
@modal.concurrent(max_inputs=DEPLOYMENT_CONFIG.gateway_max_concurrent_inputs)
@modal.asgi_app()
def fastapi_app():
    from fastapi import (
        Depends,
        FastAPI,
        Request,
    )
    from fastapi.responses import JSONResponse

    from histoforge.cloud._errors import (
        GatewayFailure,
        gateway_error_http_status,
        unexpected_gateway_failure,
    )
    from histoforge.cloud.admin_jobs import routes as admin_job_routes
    from histoforge.cloud.artifacts import routes as artifact_routes
    from histoforge.cloud.artifacts.store import ArtifactFileStore
    from histoforge.cloud.inference.finite import routes as finite_inference_routes
    from histoforge.cloud.inference.runs import routes as model_run_routes
    from histoforge.cloud.modules import routes as module_routes
    from histoforge.cloud.modules.runtime import current_module_snapshot
    from histoforge.cloud.sources import routes as source_artifact_routes
    from histoforge.cloud.sources.files import SourceFileStore
    from histoforge.cloud.sources.store import SourceArtifactStore
    from histoforge.cloud.training.datasets import routes as training_dataset_routes
    from histoforge.cloud.training.gateway import TrainingGatewayServices
    from histoforge.cloud.training.runs import routes as training_run_routes

    # Health is the sole unauthenticated route. Disable FastAPI's generated
    # documentation routes rather than exposing a second unauthenticated API
    # surface beside the closed HistoForge contract.
    web = FastAPI(
        title="HistoForge gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )

    @web.exception_handler(GatewayFailure)
    async def gateway_failure_handler(_request: Request, failure: GatewayFailure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    # Resolved once at container start; VALUES are read per request below.
    auth = modal.Dict.from_name(
        DEPLOYMENT_CONFIG.gateway_state_dict_name, create_if_missing=True
    )
    web.add_middleware(_GatewayAdmission, auth=auth)
    jobs = modal.Dict.from_name(
        DEPLOYMENT_CONFIG.admin_jobs_dict_name, create_if_missing=True
    )
    source_records = modal.Dict.from_name(
        DEPLOYMENT_CONFIG.source_artifacts_dict_name,
        create_if_missing=True,
    )
    artifact_files = ArtifactFileStore(
        "/artifacts",
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )
    source_files = SourceFileStore(
        "/artifacts",
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )
    source_store = SourceArtifactStore(
        source_records,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        DEPLOYMENT_CONFIG.limits.wire_value(),
        max_active_source_artifact_peak_bytes=(
            DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
    )
    training_services = TrainingGatewayServices(
        "/artifacts",
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    )

    def require_token(request: Request):
        # The ASGI admission owner authenticates before FastAPI can parse a
        # request body. Dependencies only expose that exact captured state to
        # route handlers and preserve their explicit authentication contract.
        return request.scope[_AUTH_STATE_SCOPE_KEY]

    @web.get("/v1/health")
    async def health():
        from histoforge.cloud.capabilities import gateway_health

        return gateway_health()

    @web.get("/v1/capabilities")
    def capabilities(state=Depends(require_token)):
        from histoforge.cloud.capabilities import gateway_capabilities

        try:
            return gateway_capabilities(state)
        except GatewayFailure:
            raise
        except Exception as error:
            raise unexpected_gateway_failure(error) from error

    module_routes.register_routes(
        web,
        require_token=require_token,
        jobs=jobs,
        current_modules=current_module_snapshot,
        training_state_factory=TrainingStateClient,
    )
    admin_job_routes.register_routes(
        web,
        require_token=require_token,
        jobs=jobs,
        current_modules=current_module_snapshot,
    )
    source_artifact_routes.register_routes(
        web,
        require_token=require_token,
        store=source_store,
        files=source_files,
        max_json_request_bytes=DEPLOYMENT_CONFIG.limits.max_json_request_bytes,
        lifecycle=SourceArtifactLifecycleClient,
    )
    artifact_routes.register_routes(
        web,
        require_token=require_token,
        files=artifact_files,
    )
    # Literal run routes must precede the generic owner/name/model POST route.
    model_run_routes.register_routes(
        web,
        require_token=require_token,
        max_json_request_bytes=DEPLOYMENT_CONFIG.limits.max_json_request_bytes,
        training_state_factory=TrainingStateClient,
    )
    finite_inference_routes.register_routes(
        web,
        require_token=require_token,
        source_store=source_store,
        files=source_files,
        max_json_request_bytes=DEPLOYMENT_CONFIG.limits.max_json_request_bytes,
        training_state_factory=TrainingStateClient,
    )
    training_dataset_routes.register_routes(
        web,
        require_token=require_token,
        service_factory=training_services.dataset,
        max_json_request_bytes=DEPLOYMENT_CONFIG.limits.max_json_request_bytes,
        max_chunk_bytes=(
            DEPLOYMENT_CONFIG.training_api_limits.transfer.max_chunk_bytes
        ),
    )
    training_run_routes.register_routes(
        web,
        require_token=require_token,
        submission_service_factory=training_services.submission,
        access_service_factory=training_services.access,
        max_json_request_bytes=DEPLOYMENT_CONFIG.limits.max_json_request_bytes,
    )

    return web
