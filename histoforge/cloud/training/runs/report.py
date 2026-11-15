"""Closed immutable report contract for managed training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_finite_number,
    require_int32,
    require_nonnegative_int32,
    require_nonnegative_safe_integer,
    require_positive_finite_number,
    require_positive_int32,
    require_sha256,
)
from histoforge.cloud._errors import validate_gateway_error
from histoforge.cloud.inference.identity import validate_module_content_identity
from histoforge.cloud.modules.schema import (
    model_output_target_classes,
    validate_model_output,
)
from histoforge.cloud.training.class_schema import (
    validate_class_initialization,
    validate_training_class_transition,
)
from histoforge.cloud.training.contract import (
    require_training_model_id,
    require_training_stable_id,
    require_training_text,
    validate_instance_training_settings,
    validate_training_augmentation_selection_syntax,
    validate_training_field_profile,
    validate_training_settings,
    validate_training_task,
)
from histoforge.cloud.training.datasets.split import validate_training_split_plan_syntax
from histoforge.cloud.training.limits import validate_training_api_limits

_SERIES_FIELDS = {
    "series_id",
    "display_name",
    "partition",
    "value_domain",
    "definition",
}
_POLICY_FIELDS = {
    "series",
    "primary_validation_series_id",
    "primary_selection_direction",
}
_METRIC_VALUE_FIELDS = {"series_id", "value"}
_EPOCH_FIELDS = {"epoch", "elapsed_seconds", "metrics"}
_UNAVAILABLE_FIELDS = {"series_id", "reason"}
_BEST_SELECTION_FIELDS = {
    "type",
    "series_id",
    "value",
    "epoch",
    "selected_weight_digest",
}
_FINAL_SELECTION_FIELDS = {
    "type",
    "reason",
    "epoch",
    "selected_weight_digest",
}
_COMPONENT_FIELDS = {"profile_id", "parameters"}
_AUGMENTATION_PRESENTATION_FIELDS = {
    "base_profile_id",
    "base_profile_display_name",
    "default_enabled_operator_ids",
    "enabled_operators",
}
_ENABLED_OPERATOR_FIELDS = {"operator_id", "display_name"}
TRAINER_CONFIGURATION_FIELDS = {
    "optimizer",
    "scheduler",
    "sampler",
    "objective",
    "augmentation",
    "preprocessing",
}
_COUNTS_FIELDS = {
    "image_count",
    "roi_count",
    "negative_roi_count",
    "target_object_counts",
}
_MESSAGE_FIELDS = {"severity", "message"}
_RESPONSE_FIELDS = {"report_revision", "report_digest", "report"}
_REPORT_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "state",
    "dataset_digest",
    "dataset_image_count",
    "split_plan",
    "source_module",
    "selected_model_id",
    "starting_weight_digest",
    "class_initialization",
    "source_model_output",
    "source_model_microns_per_pixel",
    "result_form",
    "result_display_name",
    "result_inference_gpu",
    "task",
    "trainer_build",
    "training_seed",
    "trainer_settings",
    "total_epochs",
    "field_profile",
    "augmentation",
    "trainer_configuration",
    "training_counts",
    "metric_policy",
    "epochs",
    "unavailable_metrics",
    "messages",
}
_REPORT_OPTIONAL_FIELDS = {
    "validation_counts",
    "model_selection",
    "failure",
    "min_instance_pixels",
}
_STATES = {"QUEUED", "RUNNING", "FINALIZING", "SUCCEEDED", "FAILED", "CANCELLED"}
_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
_RESULT_FORMS = {"selected_model_only", "complete_source_workflow"}


def validate_trainer_metric_policy(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate the frozen metric catalog and checkpoint-selection rule."""

    limits = validate_training_api_limits(api_limits)["report"]
    item = _exact_object(value, _POLICY_FIELDS, "trainer metric policy")
    values = _nonempty_array(item["series"], "metric series")
    if len(values) > limits["max_metric_series"]:
        raise ValueError("metric series exceed the training report limit")
    series = [_metric_series(entry, limits["max_text_bytes"]) for entry in values]
    ids = [entry["series_id"] for entry in series]
    if len(ids) != len(set(ids)):
        raise ValueError("metric series IDs must be unique")
    primary = require_training_stable_id(
        item["primary_validation_series_id"], "primary_validation_series_id"
    )
    matches = [entry for entry in series if entry["series_id"] == primary]
    if len(matches) != 1 or matches[0]["partition"] != "validation":
        raise ValueError("primary metric must name one validation series")
    direction = _enum(
        item["primary_selection_direction"],
        {"minimize", "maximize"},
        "primary_selection_direction",
    )
    return canonical_tree(
        {
            "series": series,
            "primary_validation_series_id": primary,
            "primary_selection_direction": direction,
        }
    )


def validate_resolved_trainer_configuration(
    value: Any, api_limits: Any
) -> dict[str, Any]:
    """Validate bounded evidence for every resolved trainer component."""

    item = _exact_object(
        value, TRAINER_CONFIGURATION_FIELDS, "resolved trainer configuration"
    )
    limits = validate_training_api_limits(api_limits)["report"]
    return canonical_tree(
        {
            name: validate_resolved_trainer_component(item[name], limits, name)
            for name in (
                "optimizer",
                "scheduler",
                "sampler",
                "objective",
                "augmentation",
                "preprocessing",
            )
        }
    )


def validate_training_report(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate one complete report revision and its cross-field invariants."""

    limits = validate_training_api_limits(api_limits)
    report_limits = limits["report"]
    item = _object_with_optional(
        value, _REPORT_REQUIRED_FIELDS, _REPORT_OPTIONAL_FIELDS, "training report"
    )
    if require_int32(item["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported training report schema_version")
    state = _enum(item["state"], _STATES, "state")
    split_plan = validate_training_split_plan_syntax(item["split_plan"])
    task = validate_training_task(item["task"])
    initialization = validate_class_initialization(item["class_initialization"])
    source_output = validate_model_output(item["source_model_output"])
    validate_training_class_transition(
        source_output, task["model_output"], initialization
    )
    instance_settings = validate_instance_training_settings(item, task["model_output"])
    total_epochs = require_positive_int32(item["total_epochs"], "total_epochs")
    if total_epochs > report_limits["max_epochs"]:
        raise ValueError("total_epochs exceeds the training report limit")
    trainer_settings = validate_training_settings(item["trainer_settings"])
    if trainer_settings["total_epochs"] != total_epochs:
        raise ValueError(
            "trainer_settings total_epochs differs from report total_epochs"
        )
    field_profile = validate_training_field_profile(item["field_profile"])
    configuration = validate_resolved_trainer_configuration(
        item["trainer_configuration"], limits
    )
    if configuration["preprocessing"]["profile_id"] != field_profile["preprocessing"]:
        raise ValueError("resolved preprocessing differs from field_profile")
    augmentation = validate_training_augmentation_selection_syntax(item["augmentation"])
    _validate_augmentation_presentation(
        configuration["augmentation"]["parameters"],
        augmentation,
        report_limits["max_text_bytes"],
    )
    # Import locally so the generic report structures remain independent of
    # deployment-time profile construction while every persisted report still
    # receives the backend-owned closed-schema check.
    from histoforge.cloud.training.profile_schema import (
        validate_managed_training_report_configuration,
    )

    validate_managed_training_report_configuration(
        configuration,
        task,
        augmentation,
        total_epochs,
        trainer_settings,
    )
    if trainer_settings.get("flow_loss_weighting") != configuration["objective"][
        "parameters"
    ].get("flow_loss_weighting"):
        raise ValueError("resolved flow_loss_weighting differs from trainer_settings")
    if (
        configuration["scheduler"]["parameters"]["warmup_epochs"]
        != trainer_settings["warmup_epochs"]
    ):
        raise ValueError("resolved scheduler differs from trainer_settings warmup")
    policy = validate_trainer_metric_policy(item["metric_policy"], limits)
    series_by_id = {entry["series_id"]: entry for entry in policy["series"]}
    target_classes = model_output_target_classes(task["model_output"])
    training_counts = _partition_counts(item["training_counts"], target_classes)
    validation_enabled = split_plan["type"] == "split"
    validation_counts = (
        _partition_counts(item["validation_counts"], target_classes)
        if "validation_counts" in item
        else None
    )
    if validation_enabled != (validation_counts is not None):
        raise ValueError("validation_counts presence must match split_plan")

    epochs = _epochs(item["epochs"], policy, total_epochs)
    unavailable = _unavailable_metrics(
        item["unavailable_metrics"], series_by_id, report_limits
    )
    observed_ids = {
        metric["series_id"] for epoch in epochs for metric in epoch["metrics"]
    }
    unavailable_ids = {entry["series_id"] for entry in unavailable}
    if observed_ids & unavailable_ids:
        raise ValueError("an observed metric cannot also be unavailable")
    messages = _messages(item["messages"], report_limits)
    selection = (
        _model_selection(
            item["model_selection"], policy, epochs, validation_enabled, total_epochs
        )
        if "model_selection" in item
        else None
    )
    if (
        state == "SUCCEEDED"
        and selection is not None
        and selection["type"] == "final-weights"
        and selection["reason"] == "primary_metric_unavailable"
        and policy["primary_validation_series_id"] not in unavailable_ids
    ):
        raise ValueError("unavailable primary metric requires a reason")
    failure = validate_gateway_error(item["failure"]) if "failure" in item else None
    if failure is not None:
        require_training_text(
            failure["message"], "failure.message", report_limits["max_text_bytes"]
        )
        if "resource_id" in failure:
            require_training_text(
                failure["resource_id"],
                "failure.resource_id",
                report_limits["max_text_bytes"],
            )
    _validate_report_state(
        state,
        total_epochs,
        epochs,
        unavailable,
        selection,
        failure,
    )
    if not validation_enabled:
        used_ids = observed_ids | unavailable_ids
        if any(
            series_by_id[series_id]["partition"] == "validation"
            for series_id in used_ids
        ):
            raise ValueError("a no-validation report cannot contain validation values")
    if state == "SUCCEEDED":
        applicable_ids = {
            series_id
            for series_id, series in series_by_id.items()
            if validation_enabled or series["partition"] == "training"
        }
        if observed_ids | unavailable_ids != applicable_ids:
            raise ValueError(
                "a successful report must observe or explain every applicable metric"
            )

    normalized: dict[str, Any] = {
        "schema_version": 1,
        "run_id": require_canonical_uuid(item["run_id"], "run_id"),
        "state": state,
        "dataset_digest": require_sha256(item["dataset_digest"], "dataset_digest"),
        "dataset_image_count": require_positive_int32(
            item["dataset_image_count"], "dataset_image_count"
        ),
        "split_plan": split_plan,
        "source_module": validate_module_content_identity(item["source_module"]),
        "selected_model_id": require_training_model_id(
            item["selected_model_id"], "selected_model_id"
        ),
        "starting_weight_digest": require_sha256(
            item["starting_weight_digest"], "starting_weight_digest"
        ),
        "result_form": _enum(item["result_form"], _RESULT_FORMS, "result_form"),
        "result_display_name": require_training_text(
            item["result_display_name"],
            "result_display_name",
            report_limits["max_text_bytes"],
        ),
        "result_inference_gpu": require_training_stable_id(
            item["result_inference_gpu"], "result_inference_gpu"
        ),
        "task": task,
        "class_initialization": initialization,
        "source_model_output": source_output,
        "source_model_microns_per_pixel": require_positive_finite_number(
            item["source_model_microns_per_pixel"], "source_model_microns_per_pixel"
        ),
        "trainer_build": parse_software_build_identity(item["trainer_build"]),
        "training_seed": require_nonnegative_safe_integer(
            item["training_seed"], "training_seed"
        ),
        "trainer_settings": trainer_settings,
        "total_epochs": total_epochs,
        "field_profile": field_profile,
        **instance_settings,
        "augmentation": augmentation,
        "trainer_configuration": configuration,
        "training_counts": training_counts,
        "metric_policy": policy,
        "epochs": epochs,
        "unavailable_metrics": unavailable,
        "messages": messages,
    }
    if validation_counts is not None:
        normalized["validation_counts"] = validation_counts
    if selection is not None:
        normalized["model_selection"] = selection
    if failure is not None:
        normalized["failure"] = failure
    result = canonical_tree(normalized)
    if len(canonical_bytes(result)) > report_limits["max_report_bytes"]:
        raise ValueError("training report exceeds max_report_bytes")
    return result


def validate_training_report_response(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate one advertised report wrapper returned by the run route."""

    item = _exact_object(value, _RESPONSE_FIELDS, "training report response")
    limits = validate_training_api_limits(api_limits)
    report = validate_training_report(item["report"], limits)
    digest = require_sha256(item["report_digest"], "report_digest")
    if digest != training_report_digest(report, limits):
        raise ValueError("report_digest does not match report")
    return canonical_tree(
        {
            "report_revision": require_positive_int32(
                item["report_revision"], "report_revision"
            ),
            "report_digest": digest,
            "report": report,
        }
    )


def validate_training_report_response_for_run(
    value: Any, run: Any, api_limits: Any
) -> dict[str, Any]:
    """Bind one immutable report response to the run that advertises it."""

    # Local import keeps the standalone report schema independent of run state.
    from histoforge.cloud.training.runs.contract import validate_training_run_record

    limits = validate_training_api_limits(api_limits)
    response = validate_training_report_response(value, limits)
    accepted = validate_training_run_record(run, limits)
    report = response["report"]
    request = accepted["request"]
    expected_report_fields = {
        "run_id": accepted["run_id"],
        "dataset_digest": request["dataset_digest"],
        "split_plan": request["split_plan"],
        "source_module": request["source_module"],
        "selected_model_id": request["selected_model_id"],
        "class_initialization": request["class_initialization"],
        "result_form": request["result_form"],
        "result_display_name": request["result_display_name"],
        "result_inference_gpu": request["result_inference_gpu"],
        "trainer_build": accepted["trainer_build"],
        "training_seed": request["training_seed"],
        "trainer_settings": request["trainer_settings"],
        "total_epochs": accepted["progress"]["total_epochs"],
        "field_profile": request["field_profile"],
        "min_instance_pixels": request.get("min_instance_pixels"),
        "augmentation": request["augmentation"],
    }
    if any(
        report.get(field) != expected
        for field, expected in expected_report_fields.items()
    ):
        raise ValueError("training report content differs from advertised run")
    allowed_report_states = {
        "QUEUED": {"QUEUED"},
        "RUNNING": {"QUEUED", "RUNNING"},
        "FINALIZING": {"FINALIZING"},
        "SUCCEEDED": {"SUCCEEDED"},
        "FAILED": {"FAILED"},
        "CANCELLED": {"CANCELLED"},
    }
    if (
        report["run_id"] != accepted["run_id"]
        or response["report_revision"] != accepted["report_revision"]
        or report["state"] not in allowed_report_states[accepted["state"]]
        or _last_report_epoch(report) != accepted["progress"]["completed_epochs"]
    ):
        raise ValueError("training report response differs from advertised run state")

    selection = report.get("model_selection")
    finalization = accepted.get("finalization")
    if finalization is None:
        if selection is not None:
            raise ValueError("training report selects weights before finalization")
    elif (
        selection is None
        or selection["selected_weight_digest"] != finalization["selected_weight_digest"]
    ):
        raise ValueError("training report selection differs from finalization")

    expected_digest = None
    if accepted["state"] in {"FAILED", "CANCELLED"}:
        expected_digest = accepted["result"]["report_digest"]
    elif accepted["state"] == "SUCCEEDED":
        expected_digest = finalization["report_digest"]
    if expected_digest is not None and response["report_digest"] != expected_digest:
        raise ValueError("terminal training report digest differs from run")
    if accepted["state"] == "FAILED" and (
        report["failure"] != accepted["result"]["error"]
    ):
        raise ValueError("failed training report error differs from run result")
    return response


def training_report_digest(value: Any, api_limits: Any) -> str:
    """Return SHA-256 of the validated report's canonical bytes."""

    return canonical_digest(validate_training_report(value, api_limits))


def _last_report_epoch(report: Mapping[str, Any]) -> int:
    return report["epochs"][-1]["epoch"] if report["epochs"] else 0


def _metric_series(value: Any, max_text_bytes: int) -> dict[str, Any]:
    item = _exact_object(value, _SERIES_FIELDS, "trainer metric series")
    return {
        "series_id": require_training_stable_id(item["series_id"], "series_id"),
        "display_name": require_training_text(
            item["display_name"], "display_name", max_text_bytes
        ),
        "partition": _enum(item["partition"], {"training", "validation"}, "partition"),
        "value_domain": _enum(
            item["value_domain"], {"unit_interval", "real"}, "value_domain"
        ),
        "definition": require_training_text(
            item["definition"], "definition", max_text_bytes
        ),
    }


def validate_resolved_trainer_component(
    value: Any, limits: Mapping[str, int], path: str
) -> dict[str, Any]:
    """Validate one bounded component for private configuration or a run report."""

    item = _exact_object(value, _COMPONENT_FIELDS, f"{path} component")
    parameters = item["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError(f"{path}.parameters must be an object")
    canonical = canonical_tree(parameters)
    if len(canonical_bytes(canonical)) > limits["max_component_parameters_bytes"]:
        raise ValueError(f"{path}.parameters exceed their byte limit")
    depth, entries = _shape(canonical)
    if depth > limits["max_component_parameter_depth"]:
        raise ValueError(f"{path}.parameters exceed their depth limit")
    if entries > limits["max_component_collection_entries"]:
        raise ValueError(f"{path}.parameters exceed their collection-entry limit")
    return {
        "profile_id": require_training_stable_id(item["profile_id"], "profile_id"),
        "parameters": canonical,
    }


def _validate_augmentation_presentation(
    parameters: Mapping[str, Any],
    selection: Mapping[str, Any],
    max_text_bytes: int,
) -> None:
    missing = _AUGMENTATION_PRESENTATION_FIELDS - set(parameters)
    if missing:
        raise ValueError(
            f"resolved augmentation presentation is missing fields: {sorted(missing)}"
        )
    base_profile_id = require_training_stable_id(
        parameters["base_profile_id"], "base_profile_id"
    )
    if base_profile_id != selection["profile_id"]:
        raise ValueError("resolved augmentation base profile differs from selection")
    require_training_text(
        parameters["base_profile_display_name"],
        "base_profile_display_name",
        max_text_bytes,
    )
    default_ids = [
        require_training_stable_id(value, "default_enabled_operator_ids item")
        for value in _array(
            parameters["default_enabled_operator_ids"],
            "default_enabled_operator_ids",
        )
    ]
    if len(default_ids) != len(set(default_ids)):
        raise ValueError("default enabled augmentation operators must be unique")

    enabled = []
    for value in _array(parameters["enabled_operators"], "enabled_operators"):
        item = _exact_object(value, _ENABLED_OPERATOR_FIELDS, "enabled operator")
        enabled.append(
            {
                "operator_id": require_training_stable_id(
                    item["operator_id"], "operator_id"
                ),
                "display_name": require_training_text(
                    item["display_name"], "display_name", max_text_bytes
                ),
            }
        )
    enabled_ids = [item["operator_id"] for item in enabled]
    if len(enabled_ids) != len(set(enabled_ids)):
        raise ValueError("enabled augmentation operators must be unique")
    if enabled_ids != selection["enabled_operator_ids"]:
        raise ValueError(
            "resolved enabled augmentation operators differ from selection"
        )


def _shape(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        children = [_shape(item) for item in value.values()]
        return (
            1 + max((depth for depth, _ in children), default=0),
            len(value) + sum(entries for _, entries in children),
        )
    if isinstance(value, list):
        children = [_shape(item) for item in value]
        return (
            1 + max((depth for depth, _ in children), default=0),
            len(value) + sum(entries for _, entries in children),
        )
    return 0, 0


def _partition_counts(value: Any, target_classes: Sequence[str]) -> dict[str, Any]:
    item = _exact_object(value, _COUNTS_FIELDS, "training partition counts")
    image_count = require_positive_int32(item["image_count"], "image_count")
    roi_count = require_positive_int32(item["roi_count"], "roi_count")
    negative_count = require_nonnegative_int32(
        item["negative_roi_count"], "negative_roi_count"
    )
    values = item["target_object_counts"]
    if not isinstance(values, dict):
        raise ValueError("target_object_counts must be an object")
    if set(values) != set(target_classes):
        raise ValueError("target_object_counts must match task target PathClasses")
    target_counts = {
        path_class: require_nonnegative_safe_integer(
            values[path_class], f"target_object_counts[{path_class!r}]"
        )
        for path_class in target_classes
    }
    if image_count > roi_count or negative_count > roi_count:
        raise ValueError("training partition counts are inconsistent")
    return {
        "image_count": image_count,
        "roi_count": roi_count,
        "negative_roi_count": negative_count,
        "target_object_counts": target_counts,
    }


def _epochs(
    value: Any, policy: Mapping[str, Any], total_epochs: int
) -> list[dict[str, Any]]:
    values = _array(value, "epochs")
    series_order = [entry["series_id"] for entry in policy["series"]]
    series_by_id = {entry["series_id"]: entry for entry in policy["series"]}
    result = []
    previous_epoch = 0
    previous_elapsed = 0
    for value_item in values:
        item = _exact_object(value_item, _EPOCH_FIELDS, "training epoch metric record")
        epoch = require_positive_int32(item["epoch"], "epoch")
        elapsed = require_nonnegative_safe_integer(
            item["elapsed_seconds"], "elapsed_seconds"
        )
        if epoch <= previous_epoch or epoch > total_epochs:
            raise ValueError("report epochs must be increasing and within total_epochs")
        if elapsed < previous_elapsed:
            raise ValueError("epoch elapsed_seconds must be nondecreasing")
        metrics = [_metric_value(entry) for entry in _array(item["metrics"], "metrics")]
        ids = [entry["series_id"] for entry in metrics]
        if len(ids) != len(set(ids)) or any(entry not in series_order for entry in ids):
            raise ValueError("epoch metrics must uniquely reference the metric policy")
        if ids != [entry for entry in series_order if entry in set(ids)]:
            raise ValueError("epoch metrics must follow metric policy order")
        if any(
            series_by_id[metric["series_id"]]["value_domain"] == "unit_interval"
            and not 0 <= metric["value"] <= 1
            for metric in metrics
        ):
            raise ValueError("unit-interval metric values must be from 0 through 1")
        result.append({"epoch": epoch, "elapsed_seconds": elapsed, "metrics": metrics})
        previous_epoch = epoch
        previous_elapsed = elapsed
    return result


def _metric_value(value: Any) -> dict[str, Any]:
    item = _exact_object(value, _METRIC_VALUE_FIELDS, "trainer metric value")
    return {
        "series_id": require_training_stable_id(item["series_id"], "series_id"),
        "value": require_finite_number(item["value"], "value"),
    }


def _unavailable_metrics(
    value: Any,
    series_by_id: Mapping[str, Mapping[str, Any]],
    limits: Mapping[str, int],
) -> list[dict[str, str]]:
    values = _array(value, "unavailable_metrics")
    if len(values) > limits["max_metric_series"]:
        raise ValueError("unavailable metrics exceed the report limit")
    result = []
    for value_item in values:
        item = _exact_object(value_item, _UNAVAILABLE_FIELDS, "unavailable metric")
        series_id = require_training_stable_id(item["series_id"], "series_id")
        if series_id not in series_by_id:
            raise ValueError("unavailable metric is not declared by the policy")
        result.append(
            {
                "series_id": series_id,
                "reason": require_training_text(
                    item["reason"], "reason", limits["max_text_bytes"]
                ),
            }
        )
    ids = [entry["series_id"] for entry in result]
    order = list(series_by_id)
    if len(ids) != len(set(ids)) or ids != [
        entry for entry in order if entry in set(ids)
    ]:
        raise ValueError("unavailable metrics must be unique and follow policy order")
    return result


def _messages(value: Any, limits: Mapping[str, int]) -> list[dict[str, str]]:
    values = _array(value, "messages")
    if len(values) > limits["max_messages"]:
        raise ValueError("training report messages exceed their limit")
    result = []
    for value_item in values:
        item = _exact_object(value_item, _MESSAGE_FIELDS, "training report message")
        result.append(
            {
                "severity": _enum(item["severity"], {"info", "warning"}, "severity"),
                "message": require_training_text(
                    item["message"], "message", limits["max_text_bytes"]
                ),
            }
        )
    return result


def select_training_weight_epoch(
    policy: Mapping[str, Any],
    epochs: Sequence[Mapping[str, Any]],
    *,
    validation_enabled: bool,
    total_epochs: int,
) -> dict[str, Any]:
    """Derive selection fields from validated history, before hashing weights.

    The live worker and persisted-report validator share this rule. Every defined
    primary observation participates, sparse later epochs retain earlier scores,
    and exact ties select the earliest epoch. No eligible score selects the final
    epoch; the caller must finish that epoch before publishing a result.
    """

    primary = policy["primary_validation_series_id"]
    observations = [
        (epoch["epoch"], metric["value"])
        for epoch in epochs
        for metric in epoch["metrics"]
        if metric["series_id"] == primary
    ]
    if validation_enabled and observations:
        choose = min if policy["primary_selection_direction"] == "minimize" else max
        best_epoch, best_value = choose(observations, key=lambda item: item[1])
        return {
            "type": "best-validation",
            "series_id": primary,
            "value": best_value,
            "epoch": best_epoch,
        }
    return {
        "type": "final-weights",
        "reason": (
            "primary_metric_unavailable" if validation_enabled else "no_validation"
        ),
        "epoch": total_epochs,
    }


def _model_selection(
    value: Any,
    policy: Mapping[str, Any],
    epochs: Sequence[Mapping[str, Any]],
    validation_enabled: bool,
    total_epochs: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("training model selection must be an object")
    expected = select_training_weight_epoch(
        policy,
        epochs,
        validation_enabled=validation_enabled,
        total_epochs=total_epochs,
    )
    kind = value.get("type")
    if kind == "best-validation":
        item = _exact_object(value, _BEST_SELECTION_FIELDS, "best validation selection")
        if expected["type"] != kind or item["series_id"] != expected["series_id"]:
            raise ValueError("best-validation selection has no eligible primary value")
        value_number = require_finite_number(item["value"], "value")
        epoch_number = require_positive_int32(item["epoch"], "epoch")
        if value_number != expected["value"] or epoch_number != expected["epoch"]:
            raise ValueError(
                "best-validation selection does not match report observations"
            )
        return {
            "type": kind,
            "series_id": expected["series_id"],
            "value": value_number,
            "epoch": epoch_number,
            "selected_weight_digest": require_sha256(
                item["selected_weight_digest"], "selected_weight_digest"
            ),
        }
    if kind == "final-weights":
        item = _exact_object(value, _FINAL_SELECTION_FIELDS, "final weights selection")
        reason = _enum(
            item["reason"],
            {"no_validation", "primary_metric_unavailable"},
            "reason",
        )
        if expected["type"] != kind or reason != expected["reason"]:
            raise ValueError(
                "final-weights reason does not match validation observations"
            )
        epoch = require_positive_int32(item["epoch"], "epoch")
        if epoch != total_epochs:
            raise ValueError("final weights must name the final epoch")
        return {
            "type": kind,
            "reason": reason,
            "epoch": epoch,
            "selected_weight_digest": require_sha256(
                item["selected_weight_digest"], "selected_weight_digest"
            ),
        }
    raise ValueError("unsupported training model selection type")


def _validate_report_state(
    state: str,
    total_epochs: int,
    epochs: Sequence[Mapping[str, Any]],
    unavailable: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
) -> None:
    if state == "QUEUED" and epochs:
        raise ValueError("QUEUED report cannot contain completed epochs")
    if state not in _TERMINAL_STATES and unavailable:
        raise ValueError("unavailable_metrics are terminal-only")
    if state == "FAILED":
        if failure is None:
            raise ValueError("FAILED report requires failure")
    elif failure is not None:
        raise ValueError("only FAILED report may contain failure")
    if state in {"QUEUED", "RUNNING", "CANCELLED"} and selection is not None:
        raise ValueError(f"{state} report cannot contain model_selection")
    if state in {"FINALIZING", "SUCCEEDED"}:
        if selection is None:
            raise ValueError(f"{state} report requires model_selection")
        if not epochs or epochs[-1]["epoch"] != total_epochs:
            raise ValueError(f"{state} report requires the final epoch record")


def _enum(value: Any, choices: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"unsupported {path}")
    return value


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value


def _object_with_optional(
    value: Any, required: set[str], optional: set[str], path: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    fields = set(value)
    if not required <= fields or not fields <= required | optional:
        raise ValueError(f"{path} has missing or unknown fields")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _nonempty_array(value: Any, path: str) -> list[Any]:
    result = _array(value, path)
    if not result:
        raise ValueError(f"{path} must be nonempty")
    return result
