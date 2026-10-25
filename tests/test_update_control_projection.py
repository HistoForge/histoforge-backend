"""The local updater reads job identity/state, never scientific geometry or pixels."""

import copy
import subprocess
import sys

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.inference.runs.records import (
    _RECORD_FIELDS,
    MODEL_RUN_RECORD_SCHEMA_VERSION,
    ModelRunRecords,
)

RUN_ID = "66666666-6666-4666-8666-666666666666"


def control(phase):
    # Other scientific values are intentionally opaque to this control projection.
    return dict.fromkeys(_RECORD_FIELDS) | {
        "schema_version": MODEL_RUN_RECORD_SCHEMA_VERSION,
        "request": {"model_run_id": RUN_ID},
        "phase": phase,
    }


@pytest.mark.parametrize(
    "phase",
    ["QUEUED", "RUNNING", "PARTIAL", "COMPLETE", "FAILED", "CANCELLED", "TIMED_OUT"],
)
def test_active_ids_uses_existing_record_key_and_phase_without_mutation(phase):
    store = {"model-run:v1:" + RUN_ID: control(phase), "unrelated": {}}
    before = copy.deepcopy(store)
    assert ModelRunRecords(store).active_ids() == (
        [RUN_ID] if phase in {"QUEUED", "RUNNING"} else []
    )
    assert store == before
    with pytest.raises(GatewayFailure):
        ModelRunRecords(store).get(RUN_ID)  # full scientific validation is not weakened


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("schema_version", 3),
        ("phase", "unknown"),
        ("phase", {}),
        ("request", None),
        ("request", {"model_run_id": "wrong"}),
        ("extra", True),
    ],
)
def test_invalid_control_header_fails_update(field, value):
    with pytest.raises(GatewayFailure):
        ModelRunRecords(
            {"model-run:v1:" + RUN_ID: control("RUNNING") | {field: value}}
        ).active_ids()


def test_control_lookup_and_training_page_parser_need_no_scientific_stack():
    script = """
import sys
from importlib.abc import MetaPathFinder
class NoScientificDependencies(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'shapely', 'numpy', 'torch'}:
            raise AssertionError('Local update imported ' + fullname)
sys.meta_path.insert(0, NoScientificDependencies())
from histoforge import _cloud_update
from histoforge.cloud.inference.runs.records import (
    ModelRunRecords, _RECORD_FIELDS, MODEL_RUN_RECORD_SCHEMA_VERSION)
run_id = '66666666-6666-4666-8666-666666666666'
record = dict.fromkeys(_RECORD_FIELDS) | {
    'schema_version': MODEL_RUN_RECORD_SCHEMA_VERSION,
    'request': {'model_run_id': run_id}, 'phase': 'RUNNING'}
assert ModelRunRecords({'model-run:v1:' + run_id: record}).active_ids() == [run_id]
_cloud_update.validate_training_run_page({'items': []},
    _cloud_update.DEPLOYMENT_CONFIG.training_api_limits.wire_value())
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
