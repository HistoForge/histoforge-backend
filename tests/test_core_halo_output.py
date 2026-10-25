"""CPU handoff preserves the serial window output, including typed failures."""

import copy
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.inference.feature_stream import (
    collection_budget_for_stream,
    decode_feature_stream,
    encode_canonical_feature_records,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
)
from histoforge.cloud.inference.mask_features import feature_collection_from_mask
from histoforge.cloud.inference.runs.core_halo import (
    InsufficientCoreHaloContext,
    select_canonical_core_halo_unit_features,
    translate_canonical_core_regions,
)
from histoforge.cloud.inference.runs.core_halo_output import (
    encode_core_halo_collection,
    encode_instance_window,
    encode_semantic_window,
)
from histoforge.cloud.inference.semantic_features import (
    feature_collection_from_class_map,
)

CLASSES = ("Tubule", 'Tübu"le\\cell: 🧪')
WINDOW = {
    "unit_id": 1,
    "input_rect": {"x": 90, "y": 90, "width": 80, "height": 80},
    "core_rect": {"x": 100, "y": 100, "width": 60, "height": 60},
}
RASTER = {"width": 2000, "height": 2000}
LIMITS = {
    "max_transferred_bytes": 1_000_000,
    "max_decoded_bytes": 1_000_000,
    "max_record_bytes": 100_000,
    "max_features": 100,
    "max_vertices": 1_000,
}


@pytest.fixture(scope="module")
def cpu_pool():
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        assert pool.submit(os.getpid).result(timeout=30) != os.getpid()
        yield pool


def _old_collection_output(collection, output_type, limits=LIMITS):
    """The pre-handoff runtime's exact selection and stream-encoding sequence."""

    select = (
        translate_canonical_core_regions
        if output_type == "dense_classification"
        else select_canonical_core_halo_unit_features
    )
    owned = select(
        WINDOW,
        collection,
        CLASSES,
        target_width=RASTER["width"],
        target_height=RASTER["height"],
        result_budget=collection_budget_for_stream(limits),
    )
    return encode_canonical_feature_records(owned.encoded_records(), limits)


def _instances(empty=False):
    labels = np.zeros((80, 80), dtype=np.int32)
    if not empty:
        labels[15:40, 15:40] = 1
        labels[22:27, 22:27] = 0  # One owned instance with a hole.
        labels[45:50, 20:25] = 2
        labels[45:50, 35:40] = 2  # One ID with disconnected components.
        labels[2:6, 2:6] = 3  # Context, outside this unit's ownership core.
        labels[20:24, 55:60] = 4  # Excluded when the class map omits this ID.
    return labels


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("single_class", [False, True])
def test_instance_output_matches_old_serial_inline_and_spawn(
    cpu_pool, empty, single_class
):
    labels = _instances(empty)
    classes = (
        CLASSES[0] if single_class else {1: CLASSES[0], 2: CLASSES[1], 3: CLASSES[0]}
    )
    prediction = (labels, classes)
    before = labels.copy(), copy.deepcopy(classes)
    collection = feature_collection_from_mask(
        labels, classes, CLASSES, collection_budget_for_stream(LIMITS)
    )
    expected = _old_collection_output(collection, "instance_segmentation")
    args = prediction, WINDOW, CLASSES, RASTER, LIMITS

    assert encode_instance_window(*args) == expected
    assert cpu_pool.submit(encode_instance_window, *args).result(timeout=30) == expected
    np.testing.assert_array_equal(labels, before[0])
    assert classes == before[1]
    decoded = decode_feature_stream(expected.data, CLASSES, LIMITS)
    ids = [feature["id"] for feature in decoded.collection["features"]]
    assert ids == sorted(set(ids))
    assert expected.feature_count == (0 if empty else 3 if single_class else 2)
    if not empty:
        geometries = [feature["geometry"] for feature in decoded.collection["features"]]
        assert any(geometry["type"] == "MultiPolygon" for geometry in geometries)
        assert any(
            geometry["type"] == "Polygon" and len(geometry["coordinates"]) == 2
            for geometry in geometries
        )


@pytest.mark.parametrize("empty", [False, True])
def test_semantic_core_crop_matches_old_serial_inline_and_spawn(cpu_pool, empty):
    # Semantic prediction has already cropped to the 60x60 ownership core.
    labels = np.zeros((60, 60), dtype=np.int32)
    if not empty:
        labels[2:25, 2:25] = 1
        labels[8:13, 8:13] = 0
        labels[30:35, 35:40] = 2
        labels[40:45, 45:50] = 7  # Winning but non-emitted channel.
    channels = {1: CLASSES[0], 2: CLASSES[1]}
    collection = feature_collection_from_class_map(
        labels,
        channels,
        CLASSES,
        collection_budget_for_stream(LIMITS),
        x_offset=10,
        y_offset=10,
    )
    expected = _old_collection_output(collection, "dense_classification")
    args = (labels, channels), WINDOW, CLASSES, RASTER, LIMITS

    assert encode_semantic_window(*args) == expected
    assert cpu_pool.submit(encode_semantic_window, *args).result(timeout=30) == expected
    assert expected.feature_count == (0 if empty else 2)


def _rectangle(x, y, width=5, height=5):
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]]
        ],
    }


@pytest.mark.parametrize(
    "output_type",
    ["instance_segmentation", "instance_segmentation_and_classification"],
)
def test_collection_output_preserves_duplicate_occurrence_ids(cpu_pool, output_type):
    duplicate = (CLASSES[0], _rectangle(20, 20))
    collection = build_feature_collection(
        [duplicate, (CLASSES[1], _rectangle(40, 40)), duplicate], CLASSES
    )
    expected = _old_collection_output(collection, output_type)
    args = collection, WINDOW, CLASSES, output_type, RASTER, LIMITS

    assert encode_core_halo_collection(*args) == expected
    assert (
        cpu_pool.submit(encode_core_halo_collection, *args).result(timeout=30)
        == expected
    )
    assert expected.feature_count == 3
    features = decode_feature_stream(expected.data, CLASSES, LIMITS).collection[
        "features"
    ]
    assert len({feature["id"] for feature in features}) == 3


@pytest.mark.parametrize("field", list(LIMITS))
def test_collection_output_keeps_exact_stream_limits_across_spawn(cpu_pool, field):
    collection = build_feature_collection(
        [(CLASSES[0], _rectangle(20, 20)), (CLASSES[1], _rectangle(40, 40))], CLASSES
    )
    expected = _old_collection_output(collection, "instance_segmentation")
    features = decode_feature_stream(expected.data, CLASSES, LIMITS).collection[
        "features"
    ]
    exact = {
        "max_transferred_bytes": len(expected.data),
        "max_decoded_bytes": expected.decoded_bytes,
        "max_record_bytes": max(len(canonical_bytes(feature)) for feature in features),
        "max_features": expected.feature_count,
        "max_vertices": expected.vertex_count,
    }
    args = collection, WINDOW, CLASSES, "instance_segmentation", RASTER, exact
    assert encode_core_halo_collection(*args) == expected
    assert (
        cpu_pool.submit(encode_core_halo_collection, *args).result(timeout=30)
        == expected
    )
    too_small = {**exact, field: exact[field] - 1}
    with pytest.raises(ResultLimitExceeded) as serial_error:
        _old_collection_output(collection, "instance_segmentation", too_small)
    for spawn in (False, True):
        args = collection, WINDOW, CLASSES, "instance_segmentation", RASTER, too_small
        with pytest.raises(ResultLimitExceeded) as actual:
            if spawn:
                cpu_pool.submit(encode_core_halo_collection, *args).result(timeout=30)
            else:
                encode_core_halo_collection(*args)
        assert type(actual.value) is type(serial_error.value)
        assert str(actual.value) == str(serial_error.value)


@pytest.mark.parametrize("failure", ["artificial_edge", "candidate_limit", "bad_label"])
def test_instance_failures_keep_exact_type_and_message_after_spawn(cpu_pool, failure):
    labels = _instances()
    limits = dict(LIMITS)
    if failure == "artificial_edge":
        labels.fill(0)
        labels[20:35, :35] = 1
        expected_type = InsufficientCoreHaloContext
    elif failure == "candidate_limit":
        limits["max_features"] = 1
        expected_type = ResultLimitExceeded
    else:
        labels[0, 0] = -1
        expected_type = ValueError

    with pytest.raises(expected_type) as serial_error:
        collection = feature_collection_from_mask(
            labels, CLASSES[0], CLASSES, collection_budget_for_stream(limits)
        )
        _old_collection_output(collection, "instance_segmentation", limits)
    args = (labels, CLASSES[0]), WINDOW, CLASSES, RASTER, limits
    for spawn in (False, True):
        with pytest.raises(expected_type) as actual:
            if spawn:
                cpu_pool.submit(encode_instance_window, *args).result(timeout=30)
            else:
                encode_instance_window(*args)
        assert type(actual.value) is type(serial_error.value)
        assert str(actual.value) == str(serial_error.value)

    # A normal job exception must not break the healthy worker or poison its peer.
    empty_args = (_instances(empty=True), CLASSES[0]), WINDOW, CLASSES, RASTER, LIMITS
    assert (
        cpu_pool.submit(encode_instance_window, *empty_args)
        .result(timeout=30)
        .feature_count
        == 0
    )


def test_semantic_collection_outside_core_still_fails_after_spawn(cpu_pool):
    collection = build_feature_collection([(CLASSES[0], _rectangle(3, 3))], CLASSES)
    args = collection, WINDOW, CLASSES, "dense_classification", RASTER, LIMITS
    with pytest.raises(ValueError, match="outside its ownership core") as serial_error:
        _old_collection_output(collection, "dense_classification")
    with pytest.raises(ValueError) as actual:
        cpu_pool.submit(encode_core_halo_collection, *args).result(timeout=30)
    assert type(actual.value) is type(serial_error.value)
    assert str(actual.value) == str(serial_error.value)


def _encode_without_mounted_io(kind, arguments):
    """Fail a fresh child's conversion if it reaches a mount or GPU/store import."""

    import builtins
    from unittest.mock import patch

    forbidden_modules = (
        "torch",
        "modal",
        "histoforge.cloud._app",
        "histoforge.cloud.artifacts.store",
        "histoforge.cloud.sources.files",
    )

    def forbidden(name):
        return any(
            name == module or name.startswith(module + ".")
            for module in forbidden_modules
        )

    assert not any(forbidden(name) for name in sys.modules)
    original_import = builtins.__import__

    def import_cpu_only(name, *args, **kwargs):
        assert not forbidden(name), f"CPU output imported {name}"
        return original_import(name, *args, **kwargs)

    def reject_mount_access(event, arguments):
        if event == "open" or event.startswith("os."):
            for argument in arguments:
                if isinstance(argument, (str, bytes)):
                    path = os.fsdecode(argument)
                    assert not any(
                        path == mount or path.startswith(mount + "/")
                        for mount in ("/artifacts", "/modules")
                    ), f"CPU output attempted mounted IO: {event} {path}"

    sys.addaudithook(reject_mount_access)
    # Verify the guard itself before observing the real conversion.
    for path in ("/artifacts/forbidden", "/modules/forbidden"):
        with pytest.raises(AssertionError, match="attempted mounted IO"):
            open(path, "rb")
    operation = encode_instance_window if kind == "instance" else encode_semantic_window
    with patch("builtins.__import__", import_cpu_only):
        encoded = operation(*arguments)
    assert not any(forbidden(name) for name in sys.modules)
    return encoded, os.getpid()


@pytest.mark.parametrize("kind", ["instance", "semantic"])
def test_spawned_output_uses_no_mounted_io_or_cuda_import(kind):
    if kind == "instance":
        prediction = _instances(), CLASSES[0]
        expected = encode_instance_window(prediction, WINDOW, CLASSES, RASTER, LIMITS)
    else:
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[4:9, 4:9] = 1
        prediction = labels, {1: CLASSES[0]}
        expected = encode_semantic_window(prediction, WINDOW, CLASSES, RASTER, LIMITS)
    arguments = prediction, WINDOW, CLASSES, RASTER, LIMITS
    # A separate pool gives this permanent audit hook one short-lived child.
    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        encoded, pid = pool.submit(_encode_without_mounted_io, kind, arguments).result(
            timeout=30
        )
    assert pid != os.getpid()
    assert encoded == expected


def test_runtime_closes_source_reader_before_encoding_and_publication(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from histoforge.cloud.artifacts.store import ArtifactFileStore
    from histoforge.cloud.artifacts.volume import artifact_volume_gate
    from histoforge.cloud.inference.runs.core_halo_runtime import execute_core_halo_unit
    from histoforge.cloud.rasterization.partitioned import PartitionedRaster
    from histoforge.cloud.sources.files import SourceFileStore
    from tests.test_core_halo_runtime import _values

    request, activation, limits, plan = _values()
    events = []
    reader_open = False
    identity = {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x": 0.0,
        "translate_y": 0.0,
    }
    raster = request["inputs"][0]["raster_input"]["frame"]["transform"]["raster"]
    dimensions = {key: raster[key] for key in ("width", "height")}

    def open_reader(*_args):
        nonlocal reader_open
        assert artifact_volume_gate.locked()
        reader_open = True
        events.append("open")

        def read(_series, _level, rectangle):
            assert artifact_volume_gate.locked() and reader_open
            events.append("read")
            return np.zeros((rectangle["height"], rectangle["width"], 3), np.uint8)

        def close():
            nonlocal reader_open
            assert artifact_volume_gate.locked() and reader_open
            reader_open = False
            events.append("close")

        return (
            SimpleNamespace(
                selected_level_pixels_to_level_zero=lambda *_args: identity,
                read_rgb_view=read,
                close=close,
            ),
            {
                "validated_levels": [
                    {"series_index": 0, "resolution_level": 0, **dimensions}
                ]
            },
        )

    source_files = SourceFileStore(tmp_path)
    monkeypatch.setattr(source_files, "_open_validated_source", open_reader)

    @contextmanager
    def open_pixels():
        with source_files.source_view_reader({}, {}, 0, 0, {}) as read:
            yield lambda rectangle: read(rectangle)["pixels"]

    prepared = PartitionedRaster(
        {"raster": raster, "raster_to_slide": identity},
        identity,
        {"x": 0, "y": 0, **dimensions},
        dimensions,
        dimensions["width"] * dimensions["height"],
        open_pixels,
        (0.0, 0.0, float(dimensions["width"]), float(dimensions["height"])),
    )
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )

    def predict(read_window_image, _plan, window):
        image = read_window_image()
        assert not reader_open and not artifact_volume_gate.locked()
        assert image.pixels.shape[:2] == (
            window["input_rect"]["height"],
            window["input_rect"]["width"],
        )
        events.append("predict")
        labels = np.zeros(image.pixels.shape[:2], np.int32)
        labels[4:8, 4:8] = 1
        return labels, "Tubule"

    def encode(*args):
        assert not reader_open and not artifact_volume_gate.locked()
        events.append("encode")
        return encode_instance_window(*args)

    def reload_volume():
        assert not reader_open and artifact_volume_gate.locked()
        events.append("reload")

    def commit_volume():
        assert not reader_open and artifact_volume_gate.locked()
        events.append("commit")

    files = ArtifactFileStore(
        tmp_path, reload_volume=reload_volume, commit_volume=commit_volume
    )
    publication = execute_core_halo_unit(
        request,
        0,
        0,
        limits,
        activation,
        object(),
        source_files,
        files,
        predict,
        encode,
    )
    assert events == ["open", "read", "close", "predict", "encode", "reload", "commit"]
    assert publication["unit_id"] == plan["windows"][0]["unit_id"]
    reference = publication["reusable_state"]
    decoded = decode_feature_stream(
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"]),
        {"Tubule"},
        request["result_budget"],
    )
    assert decoded.feature_count == 1
