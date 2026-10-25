"""Target gateway errors and closed worker-exception projection."""

import json
import pickle
from pathlib import Path

import httpx
import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    build_gateway_error,
    gateway_error_from_exception,
    gateway_error_http_status,
    is_automatically_retryable,
    unexpected_gateway_failure,
    validate_gateway_error,
)
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "gateway-errors-v1.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_shared_valid_gateway_errors_match_exact_bytes_and_digests():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["validCases"]:
        value = validate_gateway_error(case["value"])
        encoded = canonical_bytes(value)
        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]
        assert gateway_error_http_status(value["code"]) == case["httpStatus"]


def test_shared_gateway_error_http_mapping_is_exhaustive():
    expected_codes = set()
    for mapping in _fixture()["httpMappings"]:
        for code in mapping["codes"]:
            assert gateway_error_http_status(code) == mapping["httpStatus"]
            expected_codes.add(code)

    assert expected_codes == {code.value for code in GatewayErrorCode}


def test_shared_automatic_retry_policy_is_exact():
    expected = set(_fixture()["automaticRetryCodes"])
    actual = {
        code.value for code in GatewayErrorCode if is_automatically_retryable(code)
    }
    assert actual == expected


def test_shared_invalid_gateway_errors_fail_closed():
    for case in _fixture()["invalidCases"]:
        with pytest.raises(ValueError):
            validate_gateway_error(case["value"])


def test_gateway_error_builder_omits_absent_optionals():
    assert build_gateway_error(GatewayErrorCode.INTERNAL_ERROR, "fault") == {
        "schema_version": 1,
        "code": "INTERNAL_ERROR",
        "message": "fault",
    }


def _resp(status_code):
    return httpx.Response(
        status_code=status_code, request=httpx.Request("GET", "https://hf.co")
    )


def test_gated_repo_maps_before_repository_not_found():
    assert issubclass(GatedRepoError, RepositoryNotFoundError)
    error = gateway_error_from_exception(GatedRepoError("gated", response=_resp(403)))
    assert error["code"] == "REGISTRY_AUTHENTICATION_FAILED"
    assert "gated" not in error["message"]


def test_missing_repo_and_revision_have_distinct_codes():
    repository = gateway_error_from_exception(
        RepositoryNotFoundError("missing", response=_resp(404))
    )
    revision = gateway_error_from_exception(
        RevisionNotFoundError("missing", response=_resp(404))
    )
    assert repository["code"] == "REGISTRY_REPOSITORY_NOT_FOUND"
    assert revision["code"] == "REGISTRY_REVISION_NOT_FOUND"


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (403, "REGISTRY_AUTHENTICATION_FAILED"),
        (409, "REGISTRY_PUBLICATION_CONFLICT"),
        (412, "REGISTRY_PUBLICATION_CONFLICT"),
        (429, "REGISTRY_RATE_LIMITED"),
        (503, "SERVICE_UNAVAILABLE"),
    ],
)
def test_hf_http_status_maps_to_closed_code(status, code):
    error = gateway_error_from_exception(
        HfHubHTTPError("upstream secret", response=_resp(status))
    )
    assert error["code"] == code
    assert "upstream secret" not in error["message"]


def test_local_failures_map_without_leaking_unexpected_diagnostics():
    assert gateway_error_from_exception(FileNotFoundError("path"))["code"] == (
        "MODULE_NOT_FOUND"
    )
    assert gateway_error_from_exception(OSError("disk"))["code"] == (
        "SERVICE_UNAVAILABLE"
    )
    assert gateway_error_from_exception(ValueError("invalid field"))["code"] == (
        "VALIDATION_ERROR"
    )
    unexpected = gateway_error_from_exception(RuntimeError("credential=secret"))
    assert unexpected["code"] == "INTERNAL_ERROR"
    assert "secret" not in unexpected["message"]


def test_gateway_failure_preserves_its_already_validated_envelope():
    failure = GatewayFailure(
        GatewayErrorCode.MODULE_BUSY,
        "busy",
        resource_id="owner/module",
    )
    assert gateway_error_from_exception(failure) == failure.error


def test_gateway_failure_round_trips_across_remote_exception_transport():
    failure = GatewayFailure(
        GatewayErrorCode.RATE_LIMITED,
        "busy",
        resource_id="owner/module",
        retry_not_before="2026-08-15T01:02:03.004Z",
    )

    restored = pickle.loads(pickle.dumps(failure))

    assert type(restored) is GatewayFailure
    assert restored.error == failure.error


def test_unexpected_route_boundary_closes_service_and_internal_faults():
    service = unexpected_gateway_failure(OSError("private path"))
    internal = unexpected_gateway_failure(RuntimeError("credential=secret"))

    assert service.error["code"] == "SERVICE_UNAVAILABLE"
    assert internal.error["code"] == "INTERNAL_ERROR"
    assert "path" not in service.error["message"]
    assert "secret" not in internal.error["message"]
