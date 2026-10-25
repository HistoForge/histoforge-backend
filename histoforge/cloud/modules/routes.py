"""Authenticated catalog, deletion, and training-report routes for modules."""

import uuid
from collections.abc import Callable
from typing import Any

import modal
from fastapi import Depends, Request, Response

from histoforge.canonical_json import canonical_bytes, canonical_digest, require_sha256
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.admin_jobs import store as admin_jobs
from histoforge.cloud.http_input import canonical_path_segment
from histoforge.cloud.modules import catalog as module_catalog
from histoforge.cloud.modules.identity import module_content_identity
from histoforge.module_ref import ModuleRef, workspace_training_run_id


def _canonical_route_module_ref(
    request, owner: str, name: str, *, trailing_segments: int
) -> str | None:
    """Validate module identity from raw or platform-normalized path evidence."""

    try:
        encoded_owner = canonical_path_segment(
            request, owner, trailing_segments=trailing_segments + 1
        )
        encoded_name = canonical_path_segment(
            request, name, trailing_segments=trailing_segments
        )
        ref = ModuleRef.from_route_segments(encoded_owner, encoded_name)
    except (GatewayFailure, ValueError):
        return None
    if ref.owner != owner or ref.name != name:
        return None
    return str(ref)


def _delete_module_via_lane(
    module_ref: str, deployment_id: str, auth_revision: int
) -> None:
    """Synchronously serialize one module deletion through the mutation lane."""

    lane = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.module_mutation_class_name,
    )()
    lane.delete.remote(str(uuid.uuid4()), module_ref, deployment_id, auth_revision)


def register_routes(
    web,
    *,
    require_token,
    jobs,
    current_modules: Callable[[], Any],
    training_state_factory: Callable[[dict], Any],
) -> None:
    """Register module routes against current storage and authenticated state."""

    @web.get("/v1/modules")
    def list_modules(request: Request, state=Depends(require_token)):
        # An install committed by another container isn't visible here on a
        # warm gateway until we reload the Volume view.
        allowed = {"limit", "cursor"}
        unknown = set(request.query_params) - allowed
        duplicate = [
            name for name in allowed if len(request.query_params.getlist(name)) > 1
        ]
        if unknown or duplicate:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "catalog query has unknown or duplicate parameters",
            )
        limit_text = request.query_params.get("limit")
        if limit_text is not None and (
            not limit_text.isascii()
            or not limit_text.isdigit()
            or limit_text.startswith("0")
            or len(limit_text) > 10
        ):
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "catalog limit must use canonical positive decimal spelling",
            )
        try:
            limit = (
                DEPLOYMENT_CONFIG.limits.max_catalog_page_size
                if limit_text is None
                else int(limit_text)
            )
            with current_modules() as current:
                page = module_catalog.catalog_page(
                    current,
                    training_result_visible=(
                        training_state_factory(state).training_result_visible
                    ),
                    limit=limit,
                    cursor=request.query_params.get("cursor"),
                    max_page_size=DEPLOYMENT_CONFIG.limits.max_catalog_page_size,
                    max_encoded_bytes=(
                        DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes
                    ),
                )
            return Response(
                content=canonical_bytes(page),
                media_type="application/json",
                status_code=200,
            )
        except module_catalog.CatalogQueryError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                str(error),
            ) from error
        except GatewayFailure:
            raise
        except Exception as error:
            raise unexpected_gateway_failure(error) from error

    @web.delete("/v1/modules/{owner}/{name}")
    def delete_module(
        owner: str,
        name: str,
        request: Request,
        state=Depends(require_token),
    ):
        # Order matters:
        #   1. VALIDATE ref — path segments from the URL are attacker-
        #      controlled; ModuleRef rejects traversal and noncanonical refs.
        #      Reject before touching Modal or the filesystem.
        #   2. admin_jobs.active_job — reject when a live admin row shows
        #      that the sole lane can be busy. Besides protecting source/target
        #      bytes, this avoids knowingly queueing a synchronous delete past
        #      the client deadline. The transient row is an admission hint,
        #      never the storage lock.
        #   3. Invoke the serialized mutation lane synchronously. The lane
        #      reloads its mounted view and owns retirement + commit ordering.
        raw_ref = f"{owner}/{name}"
        ref = _canonical_route_module_ref(request, owner, name, trailing_segments=0)
        if ref is None:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                f"invalid module ref: {raw_ref!r}",
            )

        blocked = admin_jobs.active_job(jobs)
        if blocked:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_BUSY,
                "an observed administrative job occupies the module mutation lane",
                resource_id=blocked["job_id"],
            )

        try:
            _delete_module_via_lane(ref, state["deployment_id"], state["auth_revision"])
        except FileNotFoundError as error:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_NOT_FOUND,
                f"no current module exists for {ref}",
                resource_id=ref,
            ) from error
        except GatewayFailure:
            raise
        except Exception as error:
            raise unexpected_gateway_failure(error) from error
        return Response(status_code=204)

    @web.get("/v1/modules/{owner}/{name}/training-report")
    def module_training_report(
        owner: str,
        name: str,
        request: Request,
        _state=Depends(require_token),
    ):
        ref = _canonical_route_module_ref(request, owner, name, trailing_segments=1)
        if ref is None:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "invalid module ref in training-report route",
            )
        expected_fields = {"expected_content_digest"}
        if (
            set(request.query_params) != expected_fields
            or len(request.query_params.getlist("expected_content_digest")) != 1
        ):
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "training-report query requires exactly one expected_content_digest",
            )
        try:
            expected_digest = require_sha256(
                request.query_params["expected_content_digest"],
                "expected_content_digest",
            )
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                str(error),
            ) from error

        try:
            with current_modules() as current:
                return _training_report(current, ref, expected_digest)
        except GatewayFailure:
            raise
        except FileNotFoundError as error:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_NOT_FOUND,
                "the requested module does not exist",
                resource_id=ref,
            ) from error
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
                "installed module content is internally inconsistent",
                resource_id=ref,
            ) from error
        except Exception as error:
            raise unexpected_gateway_failure(error) from error


def _training_report(current: Any, ref: str, expected_digest: str) -> Response:
    """Read the complete report while its caller holds the mounted view stable."""

    record = current.read_record(ref)
    try:
        if record["content_digest"] != expected_digest:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
                "the module changed after its catalog descriptor was captured",
                resource_id=ref,
            )

        run_id = workspace_training_run_id(ref)
        if record["resolved_commit_sha"] is None:
            attestation = record.get("training_result_attestation")
            if run_id is None or attestation is None or attestation["run_id"] != run_id:
                raise GatewayFailure(
                    GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
                    (
                        "commitless module content lacks a valid "
                        "training-result attestation"
                    ),
                    resource_id=ref,
                )
        report = current.load_training_report(ref, record=record)
        if report is None:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_VERSION_UNAVAILABLE,
                "the module has no training report",
                resource_id=ref,
            )
        response = {
            "module_identity": module_content_identity(record),
            "report_digest": canonical_digest(report),
            "report": report,
        }
        return Response(
            content=canonical_bytes(response),
            media_type="application/json",
            status_code=200,
        )
    except GatewayFailure:
        raise
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "the module training report is internally inconsistent",
            resource_id=ref,
        ) from error
    except Exception as error:
        raise unexpected_gateway_failure(error) from error
