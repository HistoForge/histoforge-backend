from types import SimpleNamespace

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.http_input import canonical_path_segment


def request(**scope):
    return SimpleNamespace(scope=scope)


def test_path_segment_rejects_alias_when_optional_raw_bytes_exist():
    value = "00000000-0000-4000-8000-000000000001"
    accepted = request(
        path=f"/v1/source-artifacts/{value}",
        raw_path=f"/v1/source-artifacts/{value}".encode("ascii"),
    )
    assert canonical_path_segment(accepted, value, trailing_segments=0) == value

    aliased = request(
        path=f"/v1/source-artifacts/{value}",
        raw_path=f"/v1/source-artifacts/%30{value[1:]}".encode("ascii"),
    )
    with pytest.raises(GatewayFailure, match="not canonically encoded"):
        canonical_path_segment(aliased, value, trailing_segments=0)


def test_path_segment_uses_exact_decoded_path_when_platform_omits_raw_bytes():
    value = "00000000-0000-4000-8000-000000000001"
    deployed_scope = request(path=f"/v1/source-artifacts/{value}")

    assert canonical_path_segment(deployed_scope, value, trailing_segments=0) == value

    with pytest.raises(GatewayFailure, match="not canonically encoded"):
        canonical_path_segment(deployed_scope, "different", trailing_segments=0)


@pytest.mark.parametrize(
    "scope, decoded, trailing, message",
    [
        ({}, "artifact", 0, "request path is unavailable"),
        ({"path": "/v1/source-artifacts/révision"}, "révision", 0, "canonical ASCII"),
        ({"path": "/v1"}, "artifact", 5, "invalid shape"),
    ],
)
def test_path_segment_fails_closed_when_path_evidence_is_invalid(
    scope, decoded, trailing, message
):
    with pytest.raises(GatewayFailure, match=message):
        canonical_path_segment(
            request(**scope),
            decoded,
            trailing_segments=trailing,
        )
