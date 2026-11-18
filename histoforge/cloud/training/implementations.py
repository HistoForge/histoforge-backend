"""Private, explicit bindings from trainer admission to concrete execution.

Component profile IDs describe scientific behavior; they never select Python
code. Adding an implementation here does not advertise it: deployment still
owns the concrete compatible entries in the capability registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from histoforge.cloud.architectures.dinov3_training import (
    dinov3_class_initializations,
    project_dinov3_training_model,
)
from histoforge.cloud.architectures.instance_runtime import load_instance_model_state
from histoforge.cloud.architectures.semantic.runtime import load_semantic_model_state
from histoforge.cloud.training.bindings.dinov3_dense import (
    execute_dense_training,
    validate_dense_trainer_profile,
)
from histoforge.cloud.training.bindings.dinov3_instance import (
    execute_instance_training,
    validate_initial_managed_trainer_profile,
)
from histoforge.cloud.training.trainers.dense.configuration import (
    prepare_dense_configuration,
)
from histoforge.cloud.training.trainers.dense.policy import (
    prepare_dense_metric_policy,
)
from histoforge.cloud.training.trainers.instance.configuration import (
    prepare_instance_configuration,
)
from histoforge.cloud.training.trainers.instance.policy import (
    prepare_instance_metric_policy,
)

DINOV3_INSTANCE_TRAINER_ID = "dinov3-instance-v1"
DINOV3_DENSE_TRAINER_ID = "dinov3-dense-v1"


@dataclass(frozen=True)
class TrainingImplementation:
    """Explicit policy, execution, and model-state loading for one trainer.

    Validation and configuration are admission operations. Execution loads and
    trains the model, publishes complete epochs through the supplied callback,
    and returns its network for the common candidate-publication path. That path
    uses the bound model loader to verify the selected inference checkpoint.
    """

    validate_profile: Callable[[str, Mapping[str, Any], Mapping[str, Any]], None]
    prepare_configuration: Callable[..., dict[str, Any]]
    prepare_metric_policy: Callable[..., dict[str, Any]]
    execute: Callable[..., Any]
    load_model_state: Callable[[Any, Mapping[str, Any]], None]
    class_initializations: Callable[[str, str], tuple[str, ...]]
    project_model: Callable[..., dict[str, Any]]


_IMPLEMENTATIONS = {
    DINOV3_DENSE_TRAINER_ID: TrainingImplementation(
        validate_profile=validate_dense_trainer_profile,
        prepare_configuration=prepare_dense_configuration,
        prepare_metric_policy=prepare_dense_metric_policy,
        execute=execute_dense_training,
        load_model_state=load_semantic_model_state,
        class_initializations=dinov3_class_initializations,
        project_model=project_dinov3_training_model,
    ),
    DINOV3_INSTANCE_TRAINER_ID: TrainingImplementation(
        validate_profile=validate_initial_managed_trainer_profile,
        prepare_configuration=prepare_instance_configuration,
        prepare_metric_policy=prepare_instance_metric_policy,
        execute=execute_instance_training,
        load_model_state=load_instance_model_state,
        class_initializations=dinov3_class_initializations,
        project_model=project_dinov3_training_model,
    ),
}


def resolve_training_implementation(implementation_id: str) -> TrainingImplementation:
    """Reject unknown implementations; never substitute a different trainer."""

    try:
        return _IMPLEMENTATIONS[implementation_id]
    except (KeyError, TypeError) as error:
        raise ValueError("training implementation is not deployed") from error
