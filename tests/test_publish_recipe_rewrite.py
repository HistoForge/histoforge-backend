"""publish/_recipe_rewrite — the vendor transform (local → vendored).

Narrow contract: `local` weights become `{type: vendored, file:
<model-id>.safetensors}`; `referenced` stays pinned; `vendored` is unchanged;
nothing else is added or removed; the input is never mutated (deep copy).
"""

from histoforge.cloud.modules.publish._recipe_rewrite import to_published_recipe


def test_local_becomes_vendored():
    r = {
        "models": {"foo": {"weights": {"type": "local", "ref": "local/foo"}}},
        "steps": [],
    }
    out = to_published_recipe(r)
    assert out["models"]["foo"]["weights"] == {
        "type": "vendored",
        "file": "foo.safetensors",
    }


def test_referenced_left_pinned():
    r = {
        "models": {
            "enc": {"weights": {"type": "referenced", "ref": "M/U", "revision": "abc"}}
        },
        "steps": [],
    }
    out = to_published_recipe(r)
    assert out["models"]["enc"]["weights"] == {
        "type": "referenced",
        "ref": "M/U",
        "revision": "abc",
    }


def test_vendored_filename_is_preserved():
    r = {
        "models": {
            "c": {"weights": {"type": "vendored", "file": "c.safetensors"}},
            "d": {"weights": {"type": "vendored", "file": "weird-name.safetensors"}},
        },
        "steps": [],
    }
    out = to_published_recipe(r)
    assert out["models"]["c"]["weights"]["file"] == "c.safetensors"
    assert out["models"]["d"]["weights"]["file"] == "weird-name.safetensors"
    assert all(m["weights"]["type"] == "vendored" for m in out["models"].values())


def test_deep_copy_does_not_mutate_input():
    r = {
        "models": {"foo": {"weights": {"type": "local", "ref": "local/foo"}}},
        "steps": [],
    }
    to_published_recipe(r)
    assert r["models"]["foo"]["weights"]["type"] == "local"


def test_other_fields_preserved():
    r = {
        "schemaVersion": 3,
        "name": "N",
        "models": {
            "m": {
                "arch": "dinov3-adapter-cellpose",
                "gpu": "A10G",
                "weights": {"type": "local", "ref": "local/m"},
                "input": {
                    "type": "field",
                    "micronsPerPixel": 0.5,
                    "largeRegion": {
                        "coreWidthModelPx": 8192,
                        "coreHeightModelPx": 8192,
                        "haloPerSideModelPx": 256,
                    },
                },
                "output": {
                    "type": "instance_segmentation_and_classification",
                    "classChannels": [
                        {"channel": 0, "role": "background"},
                        {"channel": 1, "role": "emitted", "pathClass": "C"},
                    ],
                },
                "params": {"num_classes": 2},
            }
        },
        "steps": [{"model": "m"}, {"script": "s.groovy"}],
    }
    out = to_published_recipe(r)
    assert out["schemaVersion"] == 3 and out["name"] == "N"
    assert out["steps"] == [{"model": "m"}, {"script": "s.groovy"}]
    m = out["models"]["m"]
    assert m["arch"] == "dinov3-adapter-cellpose"
    assert m["input"] == r["models"]["m"]["input"]
    assert m["output"] == r["models"]["m"]["output"]
    assert m["params"] == {"num_classes": 2}
    assert m["weights"] == {"type": "vendored", "file": "m.safetensors"}
