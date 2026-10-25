"""Stop cloud work for an explicitly confirmed, exclusive-workspace update.

This is a laptop-side caller of existing cancellation and reconciliation APIs,
not another job store. Module publication and training finalization are allowed
to finish. Unaccepted requests/uploads are fenced by the subsequent credential
rotation and keep their existing expiry cleanup.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from histoforge.canonical_json import parse_json
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.inference.runs.records import ModelRunRecords
from histoforge.cloud.training.runs.contract import (
    validate_training_run_page,
    validate_training_run_record,
)

_MAX_CONTROL_BYTES = 8 * 1024**2
_WAIT_SECONDS = DEPLOYMENT_CONFIG.gateway_timeout_seconds


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_RejectRedirect())


def gateway_request(
    url: str, token: str, path: str, *, method: str = "GET", timeout: float = 30
) -> Any:
    """Read bounded control JSON on one origin, never forwarding the bearer."""

    original = url + path
    target = original
    deadline = time.monotonic() + timeout
    continuation = None
    for _ in range(21):
        request = urllib.request.Request(
            target,
            headers={"Authorization": f"Bearer {token}"},
            method=method if continuation is None else "GET",
        )
        try:
            response = _OPENER.open(request, timeout=_remaining(deadline))
        except urllib.error.HTTPError as error:
            if error.code != 303:
                raise
            # Modal hands a long-running web call back as an opaque result URL.
            # Continue that one call, never replay the mutation or follow a
            # generic redirect with the bearer credential.
            with error:
                target = _modal_result_url(original, error.headers)
            if continuation is not None and target != continuation:
                raise ValueError("Modal result continuation changed")
            continuation = target
            continue
        with response:
            body = response.read(_MAX_CONTROL_BYTES + 1)
            if len(body) > _MAX_CONTROL_BYTES:
                raise ValueError(
                    "gateway control response exceeds the helper's byte limit"
                )
            if response.status == 204:
                if body:
                    raise ValueError("gateway cancellation returned an unexpected body")
                return None
            if method == "DELETE":
                raise ValueError(
                    "gateway did not confirm inference cancellation with HTTP 204"
                )
            return parse_json(body)
    raise ValueError("Modal result continuation limit exceeded")


def _modal_result_url(original: str, headers: Any) -> str:
    locations = headers.get_all("Location", [])
    if len(locations) != 1:
        raise ValueError("Modal result continuation requires one Location header")
    target = urllib.parse.urljoin(original, locations[0])
    start, end = urllib.parse.urlsplit(original), urllib.parse.urlsplit(target)
    prefix = "__modal_attempt_token="
    if (
        (start.scheme, start.netloc, start.path) != (end.scheme, end.netloc, end.path)
        or end.username is not None
        or end.fragment
        or not end.query.startswith(prefix)
        or not end.query[len(prefix) :]
        or any(value in end.query[len(prefix) :] for value in ("&", ";"))
    ):
        raise ValueError(
            "Modal result continuation changed the request target or token query"
        )
    return target


def stop_cloud_work(
    url: str,
    token: str,
    *,
    wait_for_module_writer: Callable[[], None],
) -> None:
    """Cancel accepted runs and drain saves before replacement; never hide failure.

    The caller invokes this before rotation and after the existing mutation
    barriers. No concurrent submissions/reconnects are supported during maintenance.
    Finite calls are stopped by the local action owner and Modal's recreate cut.
    """

    import modal

    from histoforge.cloud.admin_jobs import store as admin_jobs

    # Modal lazily creates these optional job stores on first use. Only confirmed
    # absence means no jobs; authorization, transport and malformed-state errors fail.
    jobs = modal.Dict.from_name(DEPLOYMENT_CONFIG.admin_jobs_dict_name)
    runs = ModelRunRecords(modal.Dict.from_name(DEPLOYMENT_CONFIG.model_runs_dict_name))
    limits = DEPLOYMENT_CONFIG.training_api_limits.wire_value()
    deadline = time.monotonic() + _WAIT_SECONDS
    print("Stopping unfinished cloud analysis and training…", file=sys.stderr)

    try:
        inference_runs = runs.active_ids()
    except modal.exception.NotFoundError:
        inference_runs = []
    for run_id in inference_runs:
        gateway_request(url, token, f"/v1/inference/runs/{run_id}", method="DELETE")

    # A dismissed admin row is not evidence that its worker stopped. The existing
    # serialized module writer is authoritative, including final model saves.
    # Cancel expensive training first, before waiting behind a long publication.
    _stop_training(url, token, limits, deadline)
    print(
        "Waiting for module installation, publication and final saves…", file=sys.stderr
    )
    wait_for_module_writer()

    while True:
        _stop_training(url, token, limits, deadline)
        try:
            active = admin_jobs.active_job(jobs)
        except modal.exception.NotFoundError:
            active = None
        if active is None:
            return
        _wait(deadline, f"module job {active['job_id']} ({active['phase']})")


def _stop_training(url: str, token: str, limits: dict, deadline: float) -> None:
    while True:
        active = []
        for run in _training_runs(url, token, limits, deadline):
            run_id = run["run_id"]
            if run["state"] in {"QUEUED", "RUNNING"}:
                try:
                    result = gateway_request(
                        url,
                        token,
                        f"/v1/training/runs/{run_id}/cancel",
                        method="POST",
                        timeout=_remaining(deadline),
                    )
                except urllib.error.HTTPError as error:
                    if error.code != 409:
                        raise
                    # The run may have finished or started saving just before
                    # cancellation. Its current state, not the 409, decides.
                    result = gateway_request(
                        url,
                        token,
                        f"/v1/training/runs/{run_id}",
                        timeout=_remaining(deadline),
                    )
                current = validate_training_run_record(result, limits)
                if current["run_id"] != run_id:
                    raise ValueError("training cancellation returned a different run")
                if current["state"] in {"QUEUED", "RUNNING"}:
                    raise RuntimeError(
                        f"training run {run_id} did not confirm cancellation"
                    )
                if current["state"] == "FINALIZING":
                    active.append(run_id)
            elif run["state"] == "FINALIZING":
                # Listing invokes the existing reconciler: it can finish a saved
                # candidate but must not turn an indeterminate save into success.
                active.append(run_id)
        if not active:
            return
        _wait(deadline, f"final training save {active[0]}")


def _training_runs(url: str, token: str, limits: dict, deadline: float) -> list[dict]:
    items = []
    cursor = None
    seen = set()
    while True:
        query = {"limit": limits["max_page_size"]}
        if cursor is not None:
            query["cursor"] = cursor
        page = validate_training_run_page(
            gateway_request(
                url,
                token,
                "/v1/training/runs?" + urllib.parse.urlencode(query),
                timeout=_remaining(deadline),
            ),
            limits,
        )
        items.extend(page["items"])
        if len(items) > DEPLOYMENT_CONFIG.training_service_limits.max_control_records:
            raise ValueError("training run inventory exceeds its control-store bound")
        cursor = page.get("next_cursor")
        if cursor is None:
            return items
        if cursor in seen:
            raise ValueError("training run inventory repeated a page cursor")
        seen.add(cursor)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            "cloud work did not stop before the update wait limit; update not started"
        )
    return remaining


def _wait(deadline: float, description: str) -> None:
    remaining = _remaining(deadline)
    print(f"Waiting for {description}…", file=sys.stderr)
    time.sleep(min(2, remaining))
