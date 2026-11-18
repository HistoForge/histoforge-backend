"""Closed target gateway errors and exception projection."""

from __future__ import annotations

from enum import Enum
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_int32,
    require_rfc3339_millis,
)


class GatewayErrorCode(str, Enum):
    """Closed machine-actionable failures shared by every target route."""

    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"
    MODULE_BUSY = "MODULE_BUSY"
    MODULE_VERSION_UNAVAILABLE = "MODULE_VERSION_UNAVAILABLE"
    MODULE_IDENTITY_MISMATCH = "MODULE_IDENTITY_MISMATCH"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    ARTIFACT_EXPIRED = "ARTIFACT_EXPIRED"
    ARTIFACT_PRECONDITION_FAILED = "ARTIFACT_PRECONDITION_FAILED"
    RANGE_NOT_SATISFIABLE = "RANGE_NOT_SATISFIABLE"
    CALL_NOT_FOUND = "CALL_NOT_FOUND"
    CALL_EXPIRED = "CALL_EXPIRED"
    CALL_CANCELLED = "CALL_CANCELLED"
    RUNNER_FAILED = "RUNNER_FAILED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    RESULT_LIMIT_EXCEEDED = "RESULT_LIMIT_EXCEEDED"
    REGISTRY_AUTHENTICATION_FAILED = "REGISTRY_AUTHENTICATION_FAILED"
    REGISTRY_REPOSITORY_NOT_FOUND = "REGISTRY_REPOSITORY_NOT_FOUND"
    REGISTRY_REVISION_NOT_FOUND = "REGISTRY_REVISION_NOT_FOUND"
    REGISTRY_PUBLICATION_CONFLICT = "REGISTRY_PUBLICATION_CONFLICT"
    REGISTRY_RATE_LIMITED = "REGISTRY_RATE_LIMITED"
    MODEL_RUN_NOT_FOUND = "MODEL_RUN_NOT_FOUND"
    MODEL_RUN_EXPIRED = "MODEL_RUN_EXPIRED"
    TRAINING_DATASET_NOT_FOUND = "TRAINING_DATASET_NOT_FOUND"
    TRAINING_UPLOAD_NOT_FOUND = "TRAINING_UPLOAD_NOT_FOUND"
    TRAINING_RUN_NOT_FOUND = "TRAINING_RUN_NOT_FOUND"
    TRAINING_STATE_CONFLICT = "TRAINING_STATE_CONFLICT"
    PROVIDER_CREDIT_EXHAUSTED = "PROVIDER_CREDIT_EXHAUSTED"
    PROVIDER_LIMIT_REACHED = "PROVIDER_LIMIT_REACHED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_HTTP_STATUS_BY_CODE = {
    GatewayErrorCode.AUTHENTICATION_REQUIRED: 401,
    GatewayErrorCode.PROVIDER_CREDIT_EXHAUSTED: 402,
    GatewayErrorCode.MODULE_NOT_FOUND: 404,
    GatewayErrorCode.JOB_NOT_FOUND: 404,
    GatewayErrorCode.ARTIFACT_NOT_FOUND: 404,
    GatewayErrorCode.CALL_NOT_FOUND: 404,
    GatewayErrorCode.MODEL_RUN_NOT_FOUND: 404,
    GatewayErrorCode.TRAINING_DATASET_NOT_FOUND: 404,
    GatewayErrorCode.TRAINING_UPLOAD_NOT_FOUND: 404,
    GatewayErrorCode.TRAINING_RUN_NOT_FOUND: 404,
    GatewayErrorCode.MODULE_BUSY: 409,
    GatewayErrorCode.MODULE_VERSION_UNAVAILABLE: 409,
    GatewayErrorCode.MODULE_IDENTITY_MISMATCH: 409,
    GatewayErrorCode.IDEMPOTENCY_CONFLICT: 409,
    GatewayErrorCode.CALL_CANCELLED: 409,
    GatewayErrorCode.TRAINING_STATE_CONFLICT: 409,
    GatewayErrorCode.ARTIFACT_EXPIRED: 410,
    GatewayErrorCode.CALL_EXPIRED: 410,
    GatewayErrorCode.MODEL_RUN_EXPIRED: 410,
    GatewayErrorCode.ARTIFACT_PRECONDITION_FAILED: 412,
    GatewayErrorCode.PAYLOAD_TOO_LARGE: 413,
    GatewayErrorCode.RESULT_LIMIT_EXCEEDED: 413,
    GatewayErrorCode.RANGE_NOT_SATISFIABLE: 416,
    GatewayErrorCode.VALIDATION_ERROR: 422,
    GatewayErrorCode.UNSUPPORTED_SCHEMA: 422,
    GatewayErrorCode.UNSUPPORTED_CAPABILITY: 422,
    GatewayErrorCode.INSUFFICIENT_CONTEXT: 422,
    GatewayErrorCode.RATE_LIMITED: 429,
    GatewayErrorCode.PROVIDER_LIMIT_REACHED: 429,
    GatewayErrorCode.RUNNER_FAILED: 502,
    GatewayErrorCode.REGISTRY_AUTHENTICATION_FAILED: 502,
    GatewayErrorCode.REGISTRY_REPOSITORY_NOT_FOUND: 502,
    GatewayErrorCode.REGISTRY_REVISION_NOT_FOUND: 502,
    GatewayErrorCode.REGISTRY_PUBLICATION_CONFLICT: 502,
    GatewayErrorCode.REGISTRY_RATE_LIMITED: 502,
    GatewayErrorCode.SERVICE_UNAVAILABLE: 503,
    GatewayErrorCode.INTERNAL_ERROR: 500,
}

_AUTOMATIC_RETRY_CODES = frozenset(
    {
        GatewayErrorCode.RATE_LIMITED,
        GatewayErrorCode.REGISTRY_RATE_LIMITED,
        GatewayErrorCode.SERVICE_UNAVAILABLE,
    }
)

if frozenset(_HTTP_STATUS_BY_CODE) != frozenset(GatewayErrorCode):
    raise RuntimeError("every GatewayErrorCode must have one HTTP status")

_REQUIRED_FIELDS = frozenset({"schema_version", "code", "message"})
_OPTIONAL_FIELDS = frozenset({"resource_id", "retry_not_before"})


class GatewayFailure(Exception):
    """Internal exception carrying one validated top-level gateway error."""

    def __init__(
        self,
        code: GatewayErrorCode | str,
        message: str,
        *,
        resource_id: str | None = None,
        retry_not_before: str | None = None,
    ) -> None:
        self.error = build_gateway_error(
            code,
            message,
            resource_id=resource_id,
            retry_not_before=retry_not_before,
        )
        super().__init__(message)

    def __reduce__(self):
        """Preserve the validated envelope across Modal's exception transport."""

        return (type(self), (self.error["code"], self.error["message"]), self.__dict__)


def unexpected_gateway_failure(error: BaseException) -> GatewayFailure:
    """Close an unexpected route-boundary fault without leaking diagnostics."""

    if isinstance(error, GatewayFailure):
        return error
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "a required gateway service is temporarily unavailable",
        )
    return GatewayFailure(
        GatewayErrorCode.INTERNAL_ERROR,
        "the gateway failed unexpectedly",
    )


def build_gateway_error(
    code: GatewayErrorCode | str,
    message: str,
    *,
    resource_id: str | None = None,
    retry_not_before: str | None = None,
) -> dict[str, Any]:
    """Construct one validated schema-1 target error envelope."""

    error_code = _error_code(code)
    value: dict[str, Any] = {
        "schema_version": 1,
        "code": error_code.value,
        "message": _string(message, "message"),
    }
    if resource_id is not None:
        value["resource_id"] = _string(resource_id, "resource_id")
    if retry_not_before is not None:
        if not is_automatically_retryable(error_code):
            raise ValueError("retry_not_before is valid only for retryable errors")
        value["retry_not_before"] = require_rfc3339_millis(
            retry_not_before, "retry_not_before"
        )
    return canonical_tree(value)


def validate_gateway_error(value: Any) -> dict[str, Any]:
    """Validate a closed schema-1 error envelope and return its canonical tree."""

    if not isinstance(value, dict):
        raise ValueError("gateway error must be an object")
    actual = frozenset(value)
    if not _REQUIRED_FIELDS <= actual or not actual <= (
        _REQUIRED_FIELDS | _OPTIONAL_FIELDS
    ):
        raise ValueError("gateway error has missing or unknown fields")

    schema_version = require_int32(value["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError("unsupported gateway error schema_version")
    code = _error_code(value["code"])
    message = _string(value["message"], "message")
    resource_id = (
        _string(value["resource_id"], "resource_id") if "resource_id" in value else None
    )
    retry_not_before = (
        require_rfc3339_millis(value["retry_not_before"], "retry_not_before")
        if "retry_not_before" in value
        else None
    )
    return build_gateway_error(
        code,
        message,
        resource_id=resource_id,
        retry_not_before=retry_not_before,
    )


def gateway_error_http_status(code: GatewayErrorCode | str) -> int:
    """Return the one settled HTTP status for a target error code."""

    return _HTTP_STATUS_BY_CODE[_error_code(code)]


def is_automatically_retryable(code: GatewayErrorCode | str) -> bool:
    """Return whether clients may automatically retry this coded failure."""

    return _error_code(code) in _AUTOMATIC_RETRY_CODES


def gateway_error_from_exception(
    error: BaseException, *, resource_id: str | None = None
) -> dict[str, Any]:
    """Project an expected worker exception into the closed target vocabulary.

    Diagnostic messages are intentionally generic for registry and unexpected faults
    so credentials, request bodies, and upstream response content never enter persisted
    UI state.
    """

    if isinstance(error, GatewayFailure):
        return error.error

    # Hugging Face is installed in the Modal administration image, but it is
    # intentionally absent from the lightweight laptop CLI. Keep these adapter
    # types behind the only function that needs them so importing auth/error
    # contracts cannot pull registry dependencies into every CLI command.
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(error, GatedRepoError):
        code = GatewayErrorCode.REGISTRY_AUTHENTICATION_FAILED
        message = "Hugging Face denied access to the requested repository"
    elif isinstance(error, RevisionNotFoundError):
        code = GatewayErrorCode.REGISTRY_REVISION_NOT_FOUND
        message = "the requested Hugging Face revision does not exist"
    elif isinstance(error, RepositoryNotFoundError):
        code = GatewayErrorCode.REGISTRY_REPOSITORY_NOT_FOUND
        message = "the requested Hugging Face repository does not exist"
    elif isinstance(error, HfHubHTTPError):
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status in {401, 403}:
            code = GatewayErrorCode.REGISTRY_AUTHENTICATION_FAILED
        elif status == 404:
            code = GatewayErrorCode.REGISTRY_REPOSITORY_NOT_FOUND
        elif status in {409, 412}:
            code = GatewayErrorCode.REGISTRY_PUBLICATION_CONFLICT
        elif status == 429:
            code = GatewayErrorCode.REGISTRY_RATE_LIMITED
        elif status is not None and status >= 500:
            code = GatewayErrorCode.SERVICE_UNAVAILABLE
        else:
            code = GatewayErrorCode.INTERNAL_ERROR
        message = "Hugging Face rejected the registry operation"
    elif isinstance(error, FileNotFoundError):
        code = GatewayErrorCode.MODULE_NOT_FOUND
        message = "the requested module does not exist"
    elif isinstance(error, ValueError):
        code = GatewayErrorCode.VALIDATION_ERROR
        message = str(error)
    elif isinstance(error, (ConnectionError, TimeoutError, OSError)):
        code = GatewayErrorCode.SERVICE_UNAVAILABLE
        message = "a required service is temporarily unavailable"
    else:
        code = GatewayErrorCode.INTERNAL_ERROR
        message = "the administrative operation failed unexpectedly"
    return build_gateway_error(code, message, resource_id=resource_id)


def _error_code(value: GatewayErrorCode | str) -> GatewayErrorCode:
    if isinstance(value, GatewayErrorCode):
        return value
    if not isinstance(value, str):
        raise ValueError("code must be a GatewayErrorCode string")
    try:
        return GatewayErrorCode(value)
    except ValueError as exc:
        raise ValueError(f"unknown GatewayErrorCode: {value}") from exc


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value
