"""Read committed gateway state, refilling its disposable cache when needed."""

from __future__ import annotations

from typing import Any

import modal

from histoforge.cloud import _auth
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.gateway_state import cached_state


class GatewayStateClient:
    """Own cache reads and validated calls to the sole gateway-state writer."""

    def __init__(self) -> None:
        self._cache = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.gateway_state_cache_dict_name,
            create_if_missing=True,
        )
        self._class = None
        self._lane = None

    def available(self) -> bool:
        """Only confirmed class absence permits the one-time legacy bridge."""

        try:
            self._get_class().hydrate()
        except modal.exception.NotFoundError:
            return False
        return True

    def current(self) -> dict[str, Any] | None:
        state = cached_state(self._cache)
        if state is not None:
            return state
        value = self._get_lane().load_current.remote()
        return None if value is None else _auth.parse_gateway_state(value)

    def rotate(
        self,
        token_sha256: str,
        *,
        expected_deployment_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        value = self._get_lane().rotate.remote(
            token_sha256,
            expected_deployment_id,
            expected_revision,
        )
        state = _auth.parse_gateway_state(value)
        if expected_deployment_id is not None and (
            state["deployment_id"] != expected_deployment_id
        ):
            raise ValueError("gateway returned a different deployment identity")
        if expected_revision is not None and (
            state["auth_revision"] != expected_revision + 1
        ):
            raise ValueError("gateway returned a different auth revision")
        return state

    def _get_class(self):
        if self._class is None:
            self._class = modal.Cls.from_name(
                DEPLOYMENT_CONFIG.gateway_app_name,
                DEPLOYMENT_CONFIG.gateway_state_class_name,
            )
        return self._class

    def _get_lane(self):
        if self._lane is None:
            self._lane = self._get_class()()
        return self._lane
