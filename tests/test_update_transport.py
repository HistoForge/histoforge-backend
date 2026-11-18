"""Only Modal's same-call handoff may carry the gateway credential forward."""

import io
import urllib.error
from email.message import Message

import pytest
from histoforge import _cloud_update as update

URL = "https://workspace.modal.run/v1/training/runs"


def handoff(location, extra_location=None):
    headers = Message()
    headers["Location"] = location
    if extra_location is not None:
        headers["Location"] = extra_location
    return urllib.error.HTTPError(URL, 303, "See Other", headers, io.BytesIO())


def response(body=b"{}", status=200):
    stream = io.BytesIO(body)
    stream.status = status
    return stream


def test_result_handoff_does_not_repeat_mutation(monkeypatch):
    calls = []

    def open(_self, request, **_kwargs):
        calls.append(
            (request.full_url, request.method, request.headers["Authorization"])
        )
        if len(calls) == 1:
            raise handoff("?__modal_attempt_token=opaque")
        return response(b'{"state":"CANCELLED"}')

    monkeypatch.setattr(update.urllib.request.OpenerDirector, "open", open)
    assert update.gateway_request(
        "https://workspace.modal.run", "secret", "/v1/training/runs", method="POST"
    ) == {"state": "CANCELLED"}
    assert calls == [
        (URL, "POST", "Bearer secret"),
        (URL + "?__modal_attempt_token=opaque", "GET", "Bearer secret"),
    ]


@pytest.mark.parametrize(
    "location",
    [
        "https://attacker.invalid/v1/training/runs?__modal_attempt_token=x",
        "http://workspace.modal.run/v1/training/runs?__modal_attempt_token=x",
        "/different?__modal_attempt_token=x",
        "?other=x",
        "?__modal_attempt_token=",
        "?__modal_attempt_token=x&other=y",
        "?__modal_attempt_token=x;y",
        "?__modal_attempt_token=x#fragment",
    ],
)
def test_invalid_handoff_is_rejected_before_sending_another_request(
    monkeypatch, location
):
    calls = []

    def open(_self, request, **_kwargs):
        calls.append(request)
        raise handoff(location)

    monkeypatch.setattr(update.urllib.request.OpenerDirector, "open", open)
    with pytest.raises(ValueError, match="continuation"):
        update.gateway_request(
            "https://workspace.modal.run", "secret", "/v1/training/runs"
        )
    assert len(calls) == 1


def test_ambiguous_location_and_changing_result_token_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="one Location"):
        update._modal_result_url(
            URL, handoff("?__modal_attempt_token=a", "?__modal_attempt_token=b").headers
        )
    targets = iter(["a", "b"])

    def open(*_args, **_kwargs):
        raise handoff("?__modal_attempt_token=" + next(targets))

    monkeypatch.setattr(update.urllib.request.OpenerDirector, "open", open)
    with pytest.raises(ValueError, match="changed"):
        update.gateway_request(
            "https://workspace.modal.run", "secret", "/v1/training/runs"
        )


@pytest.mark.parametrize(
    ("body", "status", "succeeds"),
    [(b"", 204, True), (b"{}", 200, False), (b"unexpected", 204, False)],
)
def test_inference_cancellation_requires_empty_204(monkeypatch, body, status, succeeds):
    monkeypatch.setattr(
        update.urllib.request.OpenerDirector,
        "open",
        lambda *_args, **_kwargs: response(body, status),
    )
    if succeeds:
        assert (
            update.gateway_request(
                "https://workspace.modal.run", "secret", "/run", method="DELETE"
            )
            is None
        )
    else:
        with pytest.raises(ValueError):
            update.gateway_request(
                "https://workspace.modal.run", "secret", "/run", method="DELETE"
            )


def test_control_body_and_continuation_are_bounded(monkeypatch):
    monkeypatch.setattr(update, "_MAX_CONTROL_BYTES", 8)
    monkeypatch.setattr(
        update.urllib.request.OpenerDirector,
        "open",
        lambda *_args, **_kwargs: response(b"123456789"),
    )
    with pytest.raises(ValueError, match="byte limit"):
        update.gateway_request("https://workspace.modal.run", "secret", "/run")

    def endless(*_args, **_kwargs):
        raise handoff("?__modal_attempt_token=a")

    monkeypatch.setattr(update.urllib.request.OpenerDirector, "open", endless)
    with pytest.raises(ValueError, match="continuation limit"):
        update.gateway_request(
            "https://workspace.modal.run", "secret", "/v1/training/runs"
        )
