"""Connect completed epochs to inference weights and immutable result publication.

The acknowledged run/report remains the only history. One live worker owns a
fresh local temporary directory containing at most final and best safetensors.
These files are not resume checkpoints and never authorize a run-state change.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from histoforge.cloud.modules.files import file_identity
from histoforge.cloud.training.runs.report import (
    select_training_weight_epoch,
    validate_training_report,
)
from histoforge.cloud.training.runs.result_module import TrainingResultCandidate
from histoforge.cloud.training.runs.worker import (
    publish_training_epoch,
    read_running_training_report,
)

if TYPE_CHECKING:
    from torch import nn

_BEST_WEIGHTS = "best.safetensors"
_FINAL_WEIGHTS = "final.safetensors"


def record_training_epoch(
    run: Mapping[str, Any],
    epoch: Mapping[str, Any],
    model: nn.Module,
    *,
    weights_directory: Path,
    control: Any,
    run_files: Any,
    api_limits: Any,
) -> dict[str, Any]:
    """Retain eligible post-update weights and advertise a completed epoch.

    The synchronous trainer callback supplies the canonical, unwrapped model and
    the complete epoch record, including any validation values actually measured.
    It decides neither validation cadence nor metric IDs here. The caller retains
    the returned run and reuses its private temporary directory for later epochs.

    Any save, copy, or publication error propagates and stops training without
    retry. Local files may then be incomplete, but no result candidate exists and
    ordinary stopped-worker reconciliation fails the run. The caller's temporary
    directory scope removes the files when the worker exits normally or by error.
    """

    report = read_running_training_report(
        run, run_files=run_files, api_limits=api_limits
    )
    report["state"] = "RUNNING"
    report["epochs"].append(dict(epoch))
    report = validate_training_report(report, api_limits)
    selection = _selection(report)

    is_final = epoch["epoch"] == report["total_epochs"]
    is_best = (
        selection["type"] == "best-validation" and selection["epoch"] == epoch["epoch"]
    )
    if is_final or is_best:
        path = weights_directory / (_FINAL_WEIGHTS if is_final else _BEST_WEIGHTS)
        _save_inference_weights(model, path)
        if is_final and is_best:
            shutil.copyfile(path, weights_directory / _BEST_WEIGHTS)
    return publish_training_epoch(
        run,
        report["epochs"][-1],
        control=control,
        run_files=run_files,
        api_limits=api_limits,
    )


def publish_training_result_candidate(
    run: Mapping[str, Any],
    model: nn.Module,
    *,
    weights_directory: Path,
    source_root: Path,
    source_marker: Mapping[str, Any],
    run_files: Any,
    result_files: Any,
    api_limits: Any,
    unavailable_metrics: Sequence[Mapping[str, str]],
    load_model_state: Callable[[Any, Mapping[str, Any]], None],
    project_model: Callable[..., dict[str, Any]],
    messages: Sequence[Mapping[str, str]] = (),
) -> TrainingResultCandidate:
    """Select, exact-load, and publish the completed run's candidate output.

    The caller supplies terminal metric explanations, never invented scores. The
    report validator requires the final epoch and every applicable series to have
    an observation or an explanation. Reloading once through inference's exact
    tensor loader supplied by the implementation checks the saved state against
    the same constructed model and leaves it at the selected epoch. No optimizer
    state is loaded.

    Publication preserves the existing source/module ownership and failure rules.
    It does not install a module or mark the run SUCCEEDED; the existing finalizer
    adopts this complete candidate and owns those transitions. Cancellation may
    still win before adoption. Failed or ambiguous publication is not retried here.
    """

    from safetensors.torch import load_file

    report = read_running_training_report(
        run, run_files=run_files, api_limits=api_limits
    )
    selection = _selection(report)
    filename = (
        _BEST_WEIGHTS if selection["type"] == "best-validation" else _FINAL_WEIGHTS
    )
    selected_path = weights_directory / filename
    report.update(
        state="SUCCEEDED",
        model_selection={
            **selection,
            "selected_weight_digest": file_identity(filename, selected_path)["sha256"],
        },
        unavailable_metrics=list(unavailable_metrics),
        messages=[*report["messages"], *messages],
    )
    report = validate_training_report(report, api_limits)
    load_model_state(model, load_file(str(selected_path)))
    return result_files.publish_candidate(
        run["run_id"],
        source_root=source_root,
        source_marker=source_marker,
        selected_weights=selected_path,
        succeeded_report=report,
        request=run["request"],
        project_model=project_model,
    )


def _selection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Use the accepted policy and acknowledged observations, not new defaults."""

    return select_training_weight_epoch(
        report["metric_policy"],
        report["epochs"],
        validation_enabled=report["split_plan"]["type"] != "none",
        total_epochs=report["total_epochs"],
    )


def _save_inference_weights(model: nn.Module, path: Path) -> None:
    """Snapshot all parameters and persistent buffers in inference's file format."""

    import torch
    from safetensors.torch import save_file

    # An explicit copy also breaks shared storage on CPU. Keep frozen backbone
    # weights and architecture/classification buffers, not just trainable tensors.
    state = {
        name: tensor.detach().to(
            device="cpu", copy=True, memory_format=torch.contiguous_format
        )
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(path))
