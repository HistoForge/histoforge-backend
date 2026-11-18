"""Process-local serialization for every mounted Artifact Volume operation."""

from __future__ import annotations

import threading
from functools import wraps

# One gate for ordinary operations and context-managed readers alike. It is deliberately
# non-reentrant: close a reader scope before invoking another mounted operation.
artifact_volume_gate = threading.Lock()


def mounted_artifact_volume_operation(function):
    """Keep reload, file access, and commit inside one shared process gate."""

    @wraps(function)
    def guarded(self, *args, **kwargs):
        with artifact_volume_gate:
            return function(self, *args, **kwargs)

    return guarded
