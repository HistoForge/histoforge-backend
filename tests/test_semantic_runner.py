"""Runner-native semantic parameters reach warmup and ordinary inference."""

import weakref

import numpy as np
from histoforge._vendor.histoforge_models.pathology import semantic_inference
from histoforge.cloud.architectures.semantic.runner import SemanticRunner
from histoforge.cloud.architectures.semantic.runtime import SemanticModelRuntime
from histoforge.cloud.rasterization.rasterizer import RasterizedImage


class _FakeNet:
    def to(self, _device):
        return self

    def eval(self):
        return self


def test_semantic_batch_size_reaches_warmup_and_model_call(monkeypatch):
    calls = []
    raster = RasterizedImage(
        np.zeros((4, 5, 3), dtype=np.uint8), np.ones((4, 5), dtype=bool)
    )
    raster.real_pixel_mask[:, -1] = False
    model = {
        "arch": "dinov3-adapter-semantic",
        "input": {"type": "field", "micronsPerPixel": 0.22},
        "params": {
            "num_classes": 4,
            "bsize": 64,
            "batch_size": 1,
            "tile_overlap": 0.25,
        },
        "output": {
            "type": "dense_classification",
            "classChannels": [
                {"channel": 3, "role": "emitted", "pathClass": "Positive"},
                {"channel": 1, "role": "training_only", "pathClass": "Other"},
                {"channel": 0, "role": "background"},
                {"channel": 2, "role": "emitted", "pathClass": "Negative"},
            ],
        },
    }

    def fake_inference(_net, image, **kwargs):
        calls.append(kwargs)
        if kwargs["real_pixel_mask"] is not None:
            assert image is raster.pixels
        class_map = np.zeros(image.shape[:2], dtype=np.int32)
        class_map[0, 0] = 3
        class_map[0, 1] = 1
        class_map[0, 2] = 2
        return {"class_map": class_map}

    monkeypatch.setattr(
        "histoforge.cloud.inference.activation.activate_model",
        lambda *_args: {
            "network": _FakeNet(),
            "model": model,
            "backend": {"identity": "backend"},
            "capabilities": {"identity": "capabilities"},
        },
    )
    monkeypatch.setattr(semantic_inference, "run_semantic_inference", fake_inference)

    def execute(request, activation, source_store, files, infer):
        assert request == {"request": True}
        assert activation["model"] == model
        assert source_store == "source-store"
        assert files == "source-files"
        return infer(
            raster,
            {
                "max_transferred_bytes": 1_000_000,
                "max_decoded_bytes": 1_000_000,
                "max_record_bytes": 100_000,
                "max_features": 100,
                "max_vertices": 1_000,
            },
        )

    monkeypatch.setattr(
        "histoforge.cloud.inference.finite.runtime.execute_finite_request", execute
    )

    runner_type = SemanticRunner._get_user_cls()
    runner = runner_type()
    runner.gpu = "A10G"
    runner.ref = "owner/module"
    runner.model_id = "semantic"
    runner.content_digest = "a" * 64
    runner.resolved_commit_sha = "b" * 40
    runner_type.__dict__["load_model"]._get_raw_f()(runner)

    assert runner.runtime.batch_size == 1
    assert calls == [
        {
            "num_classes": 4,
            "bsize": 64,
            "batch_size": 1,
            "tile_overlap": 0.25,
            "real_pixel_mask": None,
            "include_diagnostics": False,
            "network_precision": "fp32",
        }
    ]

    runner.source_store = "source-store"
    runner.source_files = "source-files"
    result = runner_type.__dict__["infer"]._get_raw_f()(runner, {"request": True})

    assert calls[-1]["batch_size"] == 1
    assert calls[-1]["real_pixel_mask"] is raster.real_pixel_mask
    assert {feature["properties"]["path_class"] for feature in result["features"]} == {
        "Positive",
        "Negative",
    }
    assert all(
        feature["properties"]["path_class"] != "Other" for feature in result["features"]
    )


def test_finite_semantic_prediction_releases_fields_and_skips_unused_softmax(
    monkeypatch,
):
    from histoforge.cloud.inference import semantic_features

    expected = np.array([[0, 1, 2, 3, 2, 0], [3, 2, 1, 0, 3, 1]], np.int32)
    references = []

    def fields(*_args, **_kwargs):
        logits = np.zeros((*expected.shape, 7), np.float32)
        np.put_along_axis(logits, expected[..., None], 2, axis=-1)
        logits[..., 4:] = 1_000_000
        references.append(weakref.ref(logits))
        return logits

    def reject_softmax(*_args, **_kwargs):
        raise AssertionError("compact semantic inference computed unused softmax")

    monkeypatch.setattr(semantic_inference, "run_model_fields", fields)
    monkeypatch.setattr(semantic_inference.np, "exp", reject_softmax)
    runner = SemanticRunner._get_user_cls()()
    runner.runtime = SemanticModelRuntime(
        network=None,
        num_classes=4,
        field_options={"bsize": 32, "tile_overlap": 0.25},
        warmup_side=32,
        batch_size=1,
        emitted_classes=("Positive", "Negative"),
        class_by_channel={2: "Positive", 3: "Negative"},
    )
    image = RasterizedImage(
        np.zeros((*expected.shape, 3), np.uint8), np.ones(expected.shape, bool)
    )
    result = runner.runtime.predict(image)
    assert set(result) == {"class_map"}
    np.testing.assert_array_equal(result["class_map"], expected)
    assert all(reference() is None for reference in references)
    convert = semantic_features.feature_collection_from_class_map

    def geometry(labels, *args, **kwargs):
        assert all(reference() is None for reference in references)
        return convert(labels, *args, **kwargs)

    monkeypatch.setattr(
        semantic_features, "feature_collection_from_class_map", geometry
    )
    collection = runner._infer_features(
        image,
        {
            "max_features": 100,
            "max_vertices": 1_000,
            "max_record_bytes": 100_000,
            "max_transferred_bytes": 1_000_000,
            "max_decoded_bytes": 1_000_000,
        },
    )
    assert collection["features"]
