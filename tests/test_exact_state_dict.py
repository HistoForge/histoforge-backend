"""Runner activation requires exact executable checkpoint tensor identity."""

from collections import OrderedDict

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.common import (
    StateDictIdentityError,
    load_state_dict_exact,
)


class _TinyNet(torch.nn.Module):
    def __init__(self, head_code=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float32))
        self.register_buffer("head_type_code", torch.tensor(head_code))


def test_exact_and_consistently_prefixed_state_dicts_load():
    state = _TinyNet().state_dict()
    target = _TinyNet()
    load_state_dict_exact(target, state)
    prefixed = OrderedDict((f"module.{key}", value) for key, value in state.items())
    load_state_dict_exact(_TinyNet(), prefixed)


def test_checkpoint_must_include_current_head_metadata():
    state = _TinyNet().state_dict()
    state.pop("head_type_code")
    with pytest.raises(StateDictIdentityError, match="missing=.*head_type_code"):
        load_state_dict_exact(_TinyNet(head_code=0), state)


def test_checkpoint_head_metadata_must_match_the_constructed_model():
    state = _TinyNet(head_code=1).state_dict()

    with pytest.raises(StateDictIdentityError, match="head_type_code=1"):
        load_state_dict_exact(_TinyNet(head_code=0), state)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape", "dtype"])
def test_checkpoint_key_shape_and_dtype_mismatches_fail(mutation):
    state = _TinyNet().state_dict()
    if mutation == "missing":
        state.pop("weight")
    elif mutation == "unexpected":
        state["other"] = torch.zeros(1)
    elif mutation == "shape":
        state["weight"] = torch.zeros(3, 2)
    else:
        state["weight"] = state["weight"].to(torch.float64)

    with pytest.raises(StateDictIdentityError):
        load_state_dict_exact(_TinyNet(), state)


def test_mixed_data_parallel_prefixes_fail():
    state = _TinyNet().state_dict()
    mixed = OrderedDict(
        {
            "module.weight": state["weight"],
            "head_type_code": state["head_type_code"],
        }
    )
    with pytest.raises(StateDictIdentityError, match="mixes"):
        load_state_dict_exact(_TinyNet(), mixed)
