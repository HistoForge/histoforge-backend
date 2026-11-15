"""CPU coordination of one model run, its uploaded images and prediction workers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Event
from typing import Any

import modal

from histoforge.canonical_json import require_nonnegative_safe_integer
from histoforge.cloud._app import app, image
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    build_gateway_error,
)
from histoforge.cloud.architectures.registry import (
    MODEL_RUN_MAX_EXECUTION_SECONDS,
    resolve_runner_cls_name,
    runner_precision_for_build,
    runner_resources,
    uses_target_normalization,
)
from histoforge.cloud.inference.runs.assembly import (
    assemble_model_run_input,
    calculate_model_run_percentiles,
)

# Allow one minute after the run deadline to publish its final status. The saved
# run deadline still rejects late prediction results, regardless of Modal timing.
_COORDINATOR_TIMEOUT_SECONDS = MODEL_RUN_MAX_EXECUTION_SECONDS + 60


@app.function(
    image=image,
    timeout=_COORDINATOR_TIMEOUT_SECONDS,
    retries=0,
)
def execute_model_run(model_run_id: str) -> None:
    """Execute one already-reserved run without owning any mutable ledger."""

    execute_reserved_model_run(
        model_run_id,
        _RemoteLifecycle(),
        _remote_runner,
    )


def execute_reserved_model_run(
    model_run_id: str,
    lifecycle: Any,
    runner_factory: Callable[[Mapping[str, Any]], Any],
) -> bool:
    """Walk incomplete inputs and publish only complete immutable boundaries."""

    context = lifecycle.begin_execution(model_run_id)
    if context is None:
        return False
    request = context["request"]
    limits = context["limits"]
    completed = set(context["completed_input_ids"])
    try:
        if context["failed_units"]:
            lifecycle.fail(model_run_id, context["failed_units"][0]["failure"])
            return False
        runner = runner_factory(request)
        plans = {item["input_id"]: item["plan"] for item in context["execution_plans"]}
        completed_units = {
            (item["input_id"], item["unit_id"]): item
            for item in context["completed_units"]
        }
        for model_input in request["inputs"]:
            input_id = model_input["input_id"]
            if input_id in completed:
                continue
            plan = plans[input_id]
            if plan["type"] == "core-halo":
                result = _execute_core_halo_input(
                    model_run_id,
                    request,
                    input_id,
                    plan,
                    limits,
                    completed_units,
                    lifecycle,
                    runner,
                )
                if result is None:
                    return False
            else:
                bindings = _wait_for_source_bindings(
                    model_run_id, model_input, 0, lifecycle
                )
                if bindings is None:
                    return False
                result = _worker_result(
                    runner.execute_input(
                        request, input_id, limits, source_bindings=bindings
                    )
                )
            if not lifecycle.publish_input(
                model_run_id,
                result["input"],
                result["decoded_bytes"],
            ):
                return False
        return bool(lifecycle.complete(model_run_id))
    except GatewayFailure as error:
        lifecycle.fail(model_run_id, error.error)
        return False
    except Exception:
        lifecycle.fail(
            model_run_id,
            build_gateway_error(
                GatewayErrorCode.RUNNER_FAILED,
                "durable model execution failed before terminal publication",
                resource_id=(
                    f"{request['producer']['module_ref']}/"
                    f"{request['producer']['step_id']}"
                ),
            ),
        )
        return False


class _RemoteLifecycle:
    """Modal transport adapter for the one serialized lifecycle owner."""

    def __init__(self) -> None:
        self._lane = modal.Cls.from_name(
            DEPLOYMENT_CONFIG.gateway_app_name,
            DEPLOYMENT_CONFIG.artifact_lifecycle_class_name,
        )()

    def begin_execution(self, model_run_id: str) -> dict[str, Any] | None:
        return self._lane.begin_execution.remote(model_run_id)

    def source_delivery(self, model_run_id: str) -> dict[str, Any]:
        return self._lane.source_delivery.remote(model_run_id)

    def publish_input(
        self, model_run_id: str, publication: dict[str, Any], decoded_bytes: int
    ) -> bool:
        return self._lane.publish_input.remote(model_run_id, publication, decoded_bytes)

    def publish_unit(self, model_run_id: str, publication: dict[str, Any]) -> bool:
        return self._lane.publish_unit.remote(model_run_id, publication)

    def publish_unit_failure(
        self, model_run_id: str, publication: dict[str, Any]
    ) -> bool:
        return self._lane.publish_unit_failure.remote(model_run_id, publication)

    def complete(self, model_run_id: str) -> bool:
        return self._lane.complete.remote(model_run_id)

    def fail(self, model_run_id: str, failure: dict[str, Any]) -> bool:
        return self._lane.fail.remote(model_run_id, failure)


class _RemoteRunner:
    """Dispatch prediction to its warm GPU runner and saved-result assembly to CPU."""

    def __init__(self, runner: Any, core_halo_parallelism: int) -> None:
        self._runner = runner
        self.core_halo_parallelism = core_halo_parallelism

    def execute_input(
        self,
        request: dict[str, Any],
        input_id: int,
        limits: dict[str, Any],
        source_bindings=(),
    ) -> dict[str, Any]:
        return self._runner.infer_durable_input.remote(
            request, input_id, limits, source_bindings
        )

    def execute_core_halo_unit(
        self,
        request: dict[str, Any],
        input_id: int,
        unit_id: int,
        limits: dict[str, Any],
        source_bindings=(),
        channel_percentiles=None,
    ) -> dict[str, Any]:
        return self._runner.infer_durable_core_halo_unit.remote(
            request,
            input_id,
            unit_id,
            limits,
            source_bindings,
            **(
                {"channel_percentiles": channel_percentiles}
                if channel_percentiles is not None
                else {}
            ),
        )

    def calculate_core_halo_percentiles(
        self, model_run_id, request, input_id, limits, source_bindings
    ):
        return calculate_model_run_percentiles.remote(
            model_run_id, request, input_id, limits, source_bindings
        )

    def finalize_core_halo_input(
        self,
        request: dict[str, Any],
        input_id: int,
        limits: dict[str, Any],
        units: list[dict[str, Any]],
        source_bindings=(),
    ) -> dict[str, Any]:
        return assemble_model_run_input.remote(
            request, input_id, limits, units, source_bindings
        )


def _remote_runner(request: Mapping[str, Any]) -> _RemoteRunner:
    backend = request["expected_backend"]
    resources = runner_resources(backend["architecture"])
    class_name = resolve_runner_cls_name(backend["architecture"])
    if class_name is None:
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            "the selected model has no deployed durable runner",
            resource_id=(f"{backend['module']['module_ref']}/{backend['model_id']}"),
        )
    runner_cls = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        class_name,
    ).with_options(gpu=backend["gpu"])
    runner = runner_cls(
        ref=backend["module"]["module_ref"],
        model_id=backend["model_id"],
        content_digest=backend["module"]["content_digest"],
        resolved_commit_sha=backend["module"].get("resolved_commit_sha", ""),
        gpu=backend["gpu"],
        inference_precision=runner_precision_for_build(
            backend["architecture"], backend["runner_build"], backend["gpu"]
        ),
    )
    return _RemoteRunner(
        runner, resources.max_containers * resources.max_concurrent_inputs
    )


def _worker_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"input", "decoded_bytes"}:
        raise ValueError("durable runner returned an invalid publication envelope")
    return {
        "input": value["input"],
        "decoded_bytes": require_nonnegative_safe_integer(
            value["decoded_bytes"], "decoded_bytes"
        ),
    }


def _execute_core_halo_input(
    model_run_id: str,
    request: Mapping[str, Any],
    input_id: int,
    plan: Mapping[str, Any],
    limits: Mapping[str, Any],
    completed_units: dict[tuple[int, int], dict[str, Any]],
    lifecycle: Any,
    runner: Any,
) -> dict[str, Any] | None:
    """Prepare shared statistics only when predictions are missing, then resume."""

    expected_units = [window["unit_id"] for window in plan["windows"]]
    model_input = next(
        item for item in request["inputs"] if item["input_id"] == input_id
    )
    retained = {
        unit_id: completed_units[(input_id, unit_id)]
        for unit_id in expected_units
        if (input_id, unit_id) in completed_units
    }
    parallelism = _core_halo_parallelism(runner)
    if len(retained) != len(expected_units):
        shared_target = uses_target_normalization(
            request["expected_backend"]["architecture"], plan["type"]
        )
        target_bindings = None
        channel_percentiles = None
        if shared_target:
            target_bindings = _wait_for_source_bindings(
                model_run_id, model_input, expected_units, lifecycle
            )
            if target_bindings is None:
                return None
            channel_percentiles = runner.calculate_core_halo_percentiles(
                model_run_id, request, input_id, limits, target_bindings
            )
            if channel_percentiles is None:
                return None

        def publish(unit_id: int, publication: Any) -> bool:
            if publication is None:
                return False
            if not lifecycle.publish_unit(model_run_id, publication):
                return False
            retained[unit_id] = publication
            completed_units[(input_id, unit_id)] = publication
            return True

        def predict(unit_id, stop_waiting):
            if shared_target:
                if (
                    stop_waiting.is_set()
                    or not lifecycle.source_delivery(model_run_id)["active"]
                ):
                    return None
                return runner.execute_core_halo_unit(
                    request,
                    input_id,
                    unit_id,
                    limits,
                    source_bindings=target_bindings,
                    channel_percentiles=channel_percentiles,
                )
            bindings = _wait_for_source_bindings(
                model_run_id, model_input, unit_id, lifecycle, stop_waiting
            )
            if bindings is None:
                return None
            return runner.execute_core_halo_unit(
                request, input_id, unit_id, limits, source_bindings=bindings
            )

        if (
            _run_unit_operations(
                model_run_id,
                input_id,
                [unit_id for unit_id in expected_units if unit_id not in retained],
                parallelism,
                lifecycle,
                predict,
                publish,
            )
            is None
        ):
            return None

    ordered = [retained[unit_id] for unit_id in expected_units]
    bindings = []
    if "capture" in model_input["raster_input"]["source_view"]:
        delivery = lifecycle.source_delivery(model_run_id)
        if not delivery["active"]:
            return None
        bindings = delivery["source_bindings"]
    return _worker_result(
        runner.finalize_core_halo_input(
            request, input_id, limits, ordered, source_bindings=bindings
        )
    )


def _run_unit_operations(
    model_run_id: str,
    input_id: int,
    unit_ids: list[int],
    parallelism: int,
    lifecycle: Any,
    operation: Callable[[int, Event], Any],
    accept: Callable[[int, Any], bool],
) -> dict[int, Any] | None:
    """Keep bounded slots occupied, draining submitted work after observed failure."""

    results: dict[int, Any] = {}
    stop_waiting = Event()
    remaining = iter(unit_ids)
    acceptance_lost = False
    acceptance_error: Exception | None = None
    failures: dict[int, tuple[GatewayFailure, Exception | None]] = {}
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {}
        while True:
            # Process all ready completions before refill, so a success cannot hide
            # a ready failure and trigger another submission.
            ready = [future for future in futures if future.done()]
            if not ready and not stop_waiting.is_set():
                while len(futures) < parallelism:
                    unit_id = next(remaining, None)
                    if unit_id is None:
                        break
                    futures[pool.submit(operation, unit_id, stop_waiting)] = unit_id
            if not futures:
                break
            if not ready:
                wait(futures, return_when=FIRST_COMPLETED)
                ready = [future for future in futures if future.done()]
            for future in sorted(ready, key=futures.__getitem__):
                unit_id = futures.pop(future)
                try:
                    value = future.result()
                except GatewayFailure as error:
                    failure = error
                    cause = None
                except Exception as error:
                    failure = GatewayFailure(
                        GatewayErrorCode.RUNNER_FAILED,
                        "durable CoreHalo unit execution failed",
                    )
                    cause = error
                else:
                    # Publish immediately: another submitted CPU operation may
                    # still be waiting for its image to arrive from QuPath.
                    try:
                        accepted = accept(unit_id, value)
                    except Exception as error:
                        accepted = False
                        if acceptance_error is None:
                            acceptance_error = error
                    if accepted:
                        results[unit_id] = value
                    else:
                        acceptance_lost = True
                        stop_waiting.set()
                    continue
                failures[unit_id] = (failure, cause)
                # Stop peers that have not received an image, but drain any GPU
                # calls already in flight so their complete work can be retained.
                stop_waiting.set()

    # Account for every submitted peer, including successes after a failure, before
    # applying acceptance-error precedence and canonical failure selection.
    if acceptance_error is not None:
        raise acceptance_error
    if failures:
        unit_id = min(failures)
        failure, cause = failures[unit_id]
        publication = {
            "input_id": input_id,
            "unit_id": unit_id,
            "failure": failure.error,
        }
        if not lifecycle.publish_unit_failure(model_run_id, publication):
            return None
        if cause is not None:
            raise failure from cause
        raise failure
    if acceptance_lost:
        return None
    return results


def _wait_for_source_bindings(
    model_run_id: str,
    model_input: Mapping[str, Any],
    unit_id: int | list[int],
    lifecycle: Any,
    stop_waiting: Event | None = None,
) -> list[dict[str, Any]] | None:
    """Wait on CPU for all requested units' accepted images, without using a GPU.

    Return its file binding, an empty list if the request already names a sealed
    image, or None if the run ended or a peer stopped the wait.
    """

    if "capture" not in model_input["raster_input"]["source_view"]:
        return []
    required = set(unit_id if isinstance(unit_id, list) else [unit_id])
    stop_waiting = stop_waiting if stop_waiting is not None else Event()
    while not stop_waiting.is_set():
        delivery = lifecycle.source_delivery(model_run_id)
        if not delivery["active"]:
            return None
        bindings = [
            binding
            for binding in delivery["source_bindings"]
            if binding["input_id"] == model_input["input_id"]
            and binding["unit_id"] in required
        ]
        if {binding["unit_id"] for binding in bindings} == required:
            return bindings
        stop_waiting.wait(1)
    return None


def _core_halo_parallelism(runner: Any) -> int:
    value = getattr(runner, "core_halo_parallelism", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("core_halo_parallelism must be a positive integer")
    return value
