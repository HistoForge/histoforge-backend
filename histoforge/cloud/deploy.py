"""Register every per-architecture runner Cls + the gateway with the Modal app.

Deploy through `histoforge deploy` or `histoforge upgrade`. The helper selects
the image builder and performs the authenticated deployment handoff. Direct
`modal deploy` bypasses those required steps.

The gateway resolves an incoming served model (`<owner>/<name>/<model-id>`) →
architecture via the recipe at `/modules/<ref>/recipe.json`, then spawns the
matching runner Cls (parameterized by `(ref, model_id)`). Adding a new
architecture = one deployed entry in `architectures.registry.runner_registry`
+ one runner module + one import here.
"""

from histoforge.cloud._app import app  # noqa: F401

# One short serialized lane for administrative cap/conflict admission.
from histoforge.cloud.admin_jobs.admission import (  # noqa: F401
    AdminJobAdmissionLane,
)

# Per-architecture runners — one @app.cls per DINOv3-adapter head family.
from histoforge.cloud.architectures.cellpose.runner import CellposeRunner  # noqa: F401
from histoforge.cloud.architectures.hovernet.runner import HovernetRunner  # noqa: F401
from histoforge.cloud.architectures.point.runner import PointRunner  # noqa: F401
from histoforge.cloud.architectures.semantic.runner import SemanticRunner  # noqa: F401

# One serialized boundary for source/run references, publication and cleanup.
from histoforge.cloud.artifacts.lifecycle import (  # noqa: F401
    ArtifactLifecycleLane,
    cleanup_expired_inference_files,
)

# Stateless gateway routing inference by ref.
from histoforge.cloud.gateway import fastapi_app  # noqa: F401
from histoforge.cloud.gateway_state_service import GatewayStateLane  # noqa: F401
from histoforge.cloud.inference.runs.assembly import (  # noqa: F401
    assemble_model_run_input,
)
from histoforge.cloud.inference.runs.executor import (  # noqa: F401
    execute_model_run,
)

# One serialized Module-Volume mutation lane for install/publish/delete.
from histoforge.cloud.modules.mutation import ModuleMutationLane  # noqa: F401

# Managed training: one serialized control lane and one worker entry point.
from histoforge.cloud.training.state_lane import TrainingStateLane  # noqa: F401
from histoforge.cloud.training.worker import (
    execute_training_run,  # noqa: F401
)
