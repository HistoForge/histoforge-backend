"""The single writer of permanent gateway identity and its committed cache."""

from typing import Any

import modal

from histoforge.cloud._app import app, artifact_volume, image
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.gateway_state import GatewayStateStore


@app.cls(
    image=image,
    timeout=60,
    volumes={"/artifacts": artifact_volume},
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
class GatewayStateLane:
    """Serialize credential rotation and cache refill without running other work."""

    @modal.method()
    def load_current(self) -> dict[str, Any] | None:
        return self._store().load_current()

    @modal.method()
    def rotate(
        self,
        token_sha256: str,
        expected_deployment_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        return self._store().rotate(
            token_sha256, expected_deployment_id, expected_revision
        )

    def _store(self) -> GatewayStateStore:
        store = getattr(self, "_gateway_state_store", None)
        if store is None:
            store = GatewayStateStore(
                "/artifacts",
                modal.Dict.from_name(
                    DEPLOYMENT_CONFIG.gateway_state_cache_dict_name,
                    create_if_missing=True,
                ),
                reload_volume=artifact_volume.reload,
                commit_volume=artifact_volume.commit,
            )
            self._gateway_state_store = store
        return store
