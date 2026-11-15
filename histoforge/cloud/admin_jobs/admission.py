"""Short serialized admission boundary for administrative jobs.

Admission is deliberately separate from the long-running module mutation lane:
capacity and module-ref ownership must be decided serially, but an HTTP request
must not wait behind a download or publication merely to reach that decision.
"""

from typing import Any

import modal

from histoforge.cloud._app import app, image
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.admin_jobs import store as admin_jobs
from histoforge.cloud.gateway_state_client import GatewayStateClient


@app.cls(image=image, timeout=60, max_containers=1, retries=0)
@modal.concurrent(max_inputs=1)
class AdminJobAdmissionLane:
    """Atomically order cap/conflict checks, QUEUED publication, and spawn."""

    @modal.method()
    def accept(
        self,
        deployment_id: str,
        auth_revision: int,
        job_type: str,
        requested_module_ref: str,
        conflict_refs: list[str],
        payload: dict[str, Any],
    ) -> dict[str, str]:
        jobs = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.admin_jobs_dict_name, create_if_missing=True
        )
        auth = GatewayStateClient().current
        from histoforge.cloud._auth import require_current_auth_generation

        require_current_auth_generation(auth, deployment_id, auth_revision)
        return admin_jobs.accept_job(
            jobs,
            deployment_id=deployment_id,
            auth_revision=auth_revision,
            job_type_value=job_type,
            requested_module_ref=requested_module_ref,
            conflict_refs=conflict_refs,
            payload=payload,
        )
