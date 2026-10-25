"""Runtime adapter for the Module Volume mount and frozen store ceilings."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from histoforge.cloud._config import (
    DEPLOYMENT_CONFIG,
    MAX_INSTALLED_BYTES,
    MAX_MODULE_BYTES,
)
from histoforge.cloud.modules.current import CurrentModules
from histoforge.cloud.modules.store import ModuleStore

_module_volume_lock = Lock()


def mounted_module_store() -> ModuleStore:
    """Construct the store owner for the current mounted Module Volume view."""

    return ModuleStore(
        Path(os.environ.get("MODULES_ROOT", "/modules")),
        max_module_bytes=MAX_MODULE_BYTES,
        max_installed_bytes=MAX_INSTALLED_BYTES,
        max_catalog_encoded_bytes=DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes,
        training_api_limits=DEPLOYMENT_CONFIG.training_api_limits.wire_value(),
    )


@contextmanager
def current_module_snapshot() -> Iterator[CurrentModules]:
    """Keep this container's mount stable until all dependent reads finish.

    Modal temporarily empties a mount during reload. Hold the same lock across
    reload and filesystem consumption, including any paths read after a yield.
    Only detached values may outlive this context. Other containers and gateway
    requests that do not read this mount remain independent.
    """

    from histoforge.cloud._app import model_volume

    with _module_volume_lock:
        model_volume.reload()
        yield mounted_module_store().current
