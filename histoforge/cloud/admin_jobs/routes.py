"""FastAPI adaptation for the typed administrative-job domain."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Request, Response
from starlette.concurrency import run_in_threadpool

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
)
from histoforge.cloud._config import (
    MAX_ADMIN_REQUEST_BYTES,
    MAX_CATALOG_PAGE_SIZE,
    MAX_JSON_REQUEST_BYTES,
)
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.admin_jobs import store as admin_jobs
from histoforge.cloud.admin_jobs.contract import (
    validate_install_request,
    validate_publish_request,
)
from histoforge.cloud.http_input import read_json_body


def register_routes(
    web, *, require_token, jobs, current_modules: Callable[[], Any]
) -> None:
    """Register install, publish, status-page, and dismissal routes."""

    @web.post("/v1/install", status_code=202)
    async def install(request: Request, state=Depends(require_token)):
        payload = _request_value(validate_install_request, await _read_request(request))
        requested = payload["module_ref"]
        return await _run_admin_operation(
            lambda: admin_jobs.submit_job(
                deployment_id=state["deployment_id"],
                auth_revision=state["auth_revision"],
                job_type_value="INSTALL",
                requested_module_ref=requested,
                conflict_refs=[requested],
                payload=payload,
            )
        )

    @web.post("/v1/publish", status_code=202)
    async def publish(request: Request, state=Depends(require_token)):
        payload = _request_value(validate_publish_request, await _read_request(request))
        source = payload["source_module"]

        def accept_publish():
            try:
                with current_modules() as current:
                    current.load_recipe(source)
            except FileNotFoundError as error:
                raise GatewayFailure(
                    GatewayErrorCode.MODULE_NOT_FOUND,
                    f"no current module exists for {source}",
                    resource_id=source,
                ) from error
            return admin_jobs.submit_job(
                deployment_id=state["deployment_id"],
                auth_revision=state["auth_revision"],
                job_type_value="PUBLISH",
                requested_module_ref=payload["target_repository"],
                conflict_refs=[source, payload["target_repository"]],
                payload=payload,
            )

        return await _run_admin_operation(accept_publish)

    @web.get("/v1/jobs", dependencies=[Depends(require_token)])
    async def list_jobs(limit: str | None = None, cursor: str | None = None):
        page_limit = _page_limit(limit)
        return await _run_admin_operation(
            lambda: admin_jobs.job_page(jobs, limit=page_limit, cursor=cursor)
        )

    @web.get("/v1/jobs/{job_id}", dependencies=[Depends(require_token)])
    async def get_job(job_id: str):
        canonical_id = _job_id(job_id)
        return await _run_admin_operation(
            lambda: admin_jobs.get_status(jobs, canonical_id)
        )

    @web.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_token)])
    async def delete_job(job_id: str):
        canonical_id = _job_id(job_id)
        await _run_admin_operation(lambda: admin_jobs.forget_job(jobs, canonical_id))
        return Response(status_code=204)


async def _read_request(request: Request) -> Any:
    value = await read_json_body(request, MAX_JSON_REQUEST_BYTES)
    canonical = canonical_bytes(value)
    if len(canonical) > MAX_ADMIN_REQUEST_BYTES:
        raise GatewayFailure(
            GatewayErrorCode.PAYLOAD_TOO_LARGE,
            "administrative request exceeds the configured canonical byte limit",
        )
    return parse_json(canonical)


def _request_value(validator, value: Any) -> dict[str, Any]:
    try:
        return validator(value)
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error


def _page_limit(value: str | None) -> int:
    if value is None:
        return MAX_CATALOG_PAGE_SIZE
    if (
        not value.isascii()
        or not value.isdecimal()
        or value.startswith("0")
        or len(value) > len(str(MAX_CATALOG_PAGE_SIZE))
    ):
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR, "invalid job page limit"
        )
    limit = int(value)
    if limit > MAX_CATALOG_PAGE_SIZE:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            f"job page limit must be between 1 and {MAX_CATALOG_PAGE_SIZE}",
        )
    return limit


async def _run_admin_operation(operation: Callable[[], Any]) -> Any:
    """Run blocking Modal/storage work off-loop and close unexpected failures."""

    try:
        return await run_in_threadpool(operation)
    except GatewayFailure:
        raise
    except Exception as error:
        raise unexpected_gateway_failure(error) from error


def _job_id(value: Any) -> str:
    try:
        return require_canonical_uuid(value, "job_id")
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR, "job_id must be a canonical UUID"
        ) from error
