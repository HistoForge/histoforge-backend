"""Compact assembly preserves exact output and retained cores after failure."""

import gzip
import io
import weakref

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference import feature_stream, features
from histoforge.cloud.inference.feature_stream import (
    decode_feature_stream,
    encode_canonical_feature_records,
    encode_feature_stream,
)
from histoforge.cloud.inference.features import build_feature_collection
from histoforge.cloud.inference.runs import core_halo_state
from histoforge.cloud.inference.runs.core_halo_runtime import finalize_core_halo_input
from histoforge.cloud.inference.runs.publications import (
    model_run_execution_identity_digest,
)
from tests.test_core_halo_runtime import _geometry, _values


def _case(tmp_path, populations, classes=("Tubule",)):
    request, activation, limits, plan = _values()
    files = ArtifactFileStore(tmp_path)
    publications = []
    for unit_id, instances in enumerate(populations):
        reference = core_halo_state.publish_core_halo_unit_state(
            encode_canonical_feature_records(
                features.validate_feature_records(
                    build_feature_collection(instances, classes)["features"], classes
                ).encoded_records(),
                limits["intermediate_result"],
            ),
            files,
        )
        publications.append(
            {
                "input_id": 0,
                "unit_id": unit_id,
                "reusable_state": reference,
                "execution_identity_digest": model_run_execution_identity_digest(
                    request, 0, unit_id, limits
                ),
            }
        )
    return request, activation, limits, plan, files, publications


def _merge(case, classes=("Tubule",), output_type="instance_segmentation"):
    request, _activation, limits, plan, files, publications = case
    return core_halo_state.merge_core_halo_unit_states(
        publications,
        request=request,
        input_id=0,
        plan=plan,
        model_run_limits=limits,
        declared_classes=classes,
        output_type=output_type,
        result_budget=request["result_budget"],
        target_raster=request["inputs"][0]["raster_input"]["frame"]["transform"][
            "raster"
        ],
        files=files,
    )


@pytest.mark.parametrize(
    "output_type", ["instance_segmentation", "instance_segmentation_and_classification"]
)
def test_compact_merge_keeps_global_duplicates_holes_multipart_and_escaped_classes(
    tmp_path, output_type
):
    special_class = 'Tübu"le\\cell: 🧪'
    classes = ("Tubule", special_class)
    hole = _geometry(20, 20, width=40, height=40)
    hole["coordinates"].append([[30, 30], [30, 40], [40, 40], [40, 30], [30, 30]])
    multipart = {
        "type": "MultiPolygon",
        "coordinates": [
            _geometry(70, 20)["coordinates"],
            _geometry(90, 20)["coordinates"],
        ],
    }
    same = ("Tubule", _geometry(4, 4))
    populations = [
        [same, same, (special_class, hole)],
        [same, (special_class, same[1]), ("Tubule", multipart)],
    ]
    expected = encode_feature_stream(
        build_feature_collection(
            [item for population in populations for item in population], classes
        ),
        classes,
    )

    actual = _merge(_case(tmp_path, populations, classes), classes, output_type)

    assert actual == expected
    decoded = decode_feature_stream(actual.data, classes, _values()[0]["result_budget"])
    assert len({feature["id"] for feature in decoded.collection["features"]}) == 6


@pytest.mark.parametrize("empty", [False, True])
def test_empty_last_core_is_valid_at_exhausted_aggregate_limits(tmp_path, empty):
    first = [] if empty else [("Tubule", _geometry(4, 4))]
    case = _case(tmp_path, [first, []])
    expected = encode_feature_stream(
        build_feature_collection(first, {"Tubule"}), {"Tubule"}
    )
    limits = case[2]["intermediate_result"]
    limits.update(
        max_features=max(1, expected.feature_count),
        max_vertices=max(1, expected.vertex_count),
        max_decoded_bytes=max(1, expected.decoded_bytes),
    )
    limits["max_transferred_bytes"] = sum(
        item["reusable_state"]["size_bytes"] for item in case[5]
    )

    assert _merge(case) == expected


@pytest.mark.parametrize(
    "field",
    [
        "max_transferred_bytes",
        "max_decoded_bytes",
        "max_record_bytes",
        "max_features",
        "max_vertices",
    ],
)
def test_finalization_failure_keeps_cores_and_can_repeat_without_prediction(
    tmp_path, field
):
    case = _case(
        tmp_path, [[("Tubule", _geometry(4, 4))], [("Tubule", _geometry(260, 4))]]
    )
    request, activation, limits, _plan, files, publications = case
    expected = _merge(case)
    decoded = decode_feature_stream(expected.data, {"Tubule"}, request["result_budget"])
    ceiling = {
        "max_transferred_bytes": len(expected.data),
        "max_decoded_bytes": expected.decoded_bytes,
        "max_record_bytes": max(
            len(canonical_bytes(feature)) for feature in decoded.collection["features"]
        ),
        "max_features": expected.feature_count,
        "max_vertices": expected.vertex_count,
    }[field]
    request["result_budget"][field] = ceiling
    assert _merge(case) == expected
    original_ids = set(files.artifact_ids())
    request["result_budget"][field] -= 1

    with pytest.raises(GatewayFailure, match=field) as failure:
        finalize_core_halo_input(request, 0, limits, publications, activation, files)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert set(files.artifact_ids()) == original_ids
    for publication in publications:
        reference = publication["reusable_state"]
        assert files.load_artifact_ref(reference["artifact_id"]) == reference

    request["result_budget"][field] = ceiling
    result = finalize_core_halo_input(
        request, 0, limits, publications, activation, files
    )
    reference = result["input"]["feature_stream_artifact"]
    assert (
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"])
        == expected.data
    )


@pytest.mark.parametrize(
    "field",
    [
        "max_transferred_bytes",
        "max_decoded_bytes",
        "max_record_bytes",
        "max_features",
        "max_vertices",
    ],
)
def test_each_core_is_bounded_even_when_the_aggregate_would_fit(
    tmp_path, monkeypatch, field
):
    case = _case(
        tmp_path,
        [
            [("Tubule", _geometry(4, 4))],
            [("Tubule", _geometry(260, 4)), ("Tubule", _geometry(270, 4))],
        ],
    )
    request, activation, limits, _plan, files, publications = case
    reference = publications[-1]["reusable_state"]
    decoded = decode_feature_stream(
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"]),
        {"Tubule"},
        request["result_budget"],
    )
    ceiling = {
        "max_transferred_bytes": reference["size_bytes"],
        "max_decoded_bytes": decoded.decoded_bytes,
        "max_record_bytes": max(
            len(canonical_bytes(feature)) for feature in decoded.collection["features"]
        ),
        "max_features": decoded.feature_count,
        "max_vertices": decoded.vertex_count,
    }[field]
    processing = dict(limits["intermediate_result"])
    processing[field] = ceiling
    monkeypatch.setattr(
        core_halo_state, "processing_result_limits", lambda _architecture: processing
    )
    assert _merge(case).feature_count == 3
    original_ids = set(files.artifact_ids())
    processing[field] -= 1
    with pytest.raises(GatewayFailure) as failure:
        finalize_core_halo_input(request, 0, limits, publications, activation, files)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert set(files.artifact_ids()) == original_ids
    assert all(
        files.load_artifact_ref(item["reusable_state"]["artifact_id"])
        == item["reusable_state"]
        for item in publications
    )


def test_per_core_recovery_ceiling_is_checked_before_reading(tmp_path, monkeypatch):
    case = _case(
        tmp_path, [[("Tubule", _geometry(4, 4))], [("Tubule", _geometry(260, 4))]]
    )
    _request, _activation, limits, _plan, files, publications = case
    limits["max_recovery_state_bytes_per_unit"] = (
        min(item["reusable_state"]["size_bytes"] for item in publications) - 1
    )
    monkeypatch.setattr(
        files,
        "read_artifact_block",
        lambda *_args: pytest.fail("oversized core must fail before reading"),
    )
    with pytest.raises(ValueError, match="per-unit"):
        _merge(case)


@pytest.mark.parametrize("x,y", [(-1, 4), (510, 4), (4, -1), (4, 382)])
def test_compact_merge_still_rejects_out_of_target_coordinates(tmp_path, x, y):
    case = _case(
        tmp_path, [[("Tubule", _geometry(4, 4))], [("Tubule", _geometry(x, y))]]
    )
    with pytest.raises(ValueError, match="outside the target raster"):
        _merge(case)


@pytest.mark.parametrize(
    "corruption",
    [
        "digest",
        "gzip",
        "unknown_class",
        "id",
        "trailing_record",
        "crc",
        "truncated",
        "member",
        "padding",
    ],
)
def test_late_corrupt_core_never_publishes_a_final_result(tmp_path, corruption):
    case = _case(
        tmp_path, [[("Tubule", _geometry(4, 4))], [("Tubule", _geometry(260, 4))]]
    )
    request, activation, limits, _plan, files, publications = case
    reference = publications[-1]["reusable_state"]
    path = tmp_path / "files" / reference["artifact_id"] / "data.bin"
    data = path.read_bytes()
    if corruption == "digest":
        path.write_bytes(b"x" * len(data))
    else:
        payload = gzip.decompress(data)
        if corruption == "unknown_class":
            payload = payload.replace(b'"path_class":"Tubule"', b'"path_class":"Other"')
        elif corruption == "id":
            feature = decode_feature_stream(
                data, {"Tubule"}, limits["intermediate_result"]
            ).collection["features"][0]
            payload = payload.replace(feature["id"].encode(), b"0" * 64)
        elif corruption == "trailing_record":
            payload += b'\x1e{"type":\n'
        output = io.BytesIO()
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=output,
            mtime=1 if corruption == "gzip" else 0,
            compresslevel=9,
        ) as stream:
            stream.write(payload)
        damaged = output.getvalue()
        if corruption == "crc":
            damaged = damaged[:-8] + bytes([damaged[-8] ^ 1]) + damaged[-7:]
        elif corruption == "truncated":
            damaged = damaged[:-1]
        elif corruption == "member":
            damaged += damaged
        elif corruption == "padding":
            damaged += b"\0"
        publications[-1]["reusable_state"] = files.publish_artifact_bytes(
            damaged,
            media_type=reference["media_type"],
            content_encoding=reference["content_encoding"],
        )
    original_ids = set(files.artifact_ids())
    with pytest.raises(GatewayFailure) as failure:
        finalize_core_halo_input(request, 0, limits, publications, activation, files)
    assert failure.value.error["code"] == (
        "SERVICE_UNAVAILABLE" if corruption == "digest" else "RUNNER_FAILED"
    )
    assert set(files.artifact_ids()) == original_ids
    assert (
        files.load_artifact_ref(publications[0]["reusable_state"]["artifact_id"])
        == publications[0]["reusable_state"]
    )


def test_completed_core_is_released_before_decoding_the_next(tmp_path, monkeypatch):
    case = _case(
        tmp_path, [[("Tubule", _geometry(4, 4))], [("Tubule", _geometry(260, 4))]]
    )
    decode = core_halo_state.decode_feature_stream
    previous = None
    calls = 0

    def checked_decode(*args):
        nonlocal previous, calls
        assert previous is None or previous() is None
        decoded = decode(*args)
        previous = weakref.ref(decoded)
        calls += 1
        return decoded

    monkeypatch.setattr(core_halo_state, "decode_feature_stream", checked_decode)
    _merge(case)
    assert calls == 2
    assert previous() is None


def test_coordinate_json_is_encoded_once_per_feature_and_reused_in_assembly(
    tmp_path, monkeypatch
):
    case = _case(
        tmp_path, [[("Tubule", _geometry(4, 4))], [("Tubule", _geometry(260, 4))]]
    )
    encode = features.canonical_bytes
    coordinate_encodes = 0

    def counted_encode(value):
        nonlocal coordinate_encodes
        if isinstance(value, list):
            coordinate_encodes += 1
        return encode(value)

    monkeypatch.setattr(features, "canonical_bytes", counted_encode)
    monkeypatch.setattr(
        feature_stream,
        "canonical_bytes",
        lambda *_args: pytest.fail(
            "assembly must reuse the validated coordinate bytes"
        ),
    )
    assert _merge(case).feature_count == 2
    assert coordinate_encodes == 2
