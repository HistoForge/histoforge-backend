"""Cross-language finite-inference identity and envelope contract fixtures."""

import copy
import json
import math
from pathlib import Path

import pytest
from fastapi import FastAPI, Query
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes, canonical_digest, sha256_bytes
from histoforge.cloud.inference.contract import (
    build_backend_input_identity,
)
from histoforge.cloud.inference.finite.contract import (
    BACKEND_CALL_QUERY_PARAMETER,
    encode_backend_call_query,
    require_backend_call_id,
    validate_backend_result_meta,
    validate_finite_inference_request,
    validate_finite_route_request,
    validate_inference_acceptance,
    validate_inference_accepted,
    validate_inference_cancellation_acceptance,
    validate_inference_cancellation_accepted,
    validate_inference_pending,
    validate_inference_pending_response,
    validate_inference_result,
)
from histoforge.cloud.inference.identity import backend_model_identity_digest
from histoforge.cloud.rasterization.inputs import (
    build_model_raster_input,
    build_raster_frame,
    build_source_view,
    validate_raster_frame,
    validate_source_view,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "finite-inference-v1.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _wire(case, name):
    return json.loads(case[f"{name}Canonical"])


def _assert_exact(case, name, value):
    encoded = canonical_bytes(value)
    assert encoded.decode("utf-8") == case[f"{name}Canonical"], case["name"]
    assert sha256_bytes(encoded) == case[f"{name}Sha256"], case["name"]


def test_shared_fixture_binds_spatial_identity_and_all_finite_envelopes():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["cases"]:
        raw_request = _wire(case, "request")
        route = case["route"]
        request = validate_finite_route_request(
            raw_request, route["moduleRef"], route["modelId"]
        )
        _assert_exact(case, "request", request)

        model_input = request["raster_input"]
        view = model_input["source_view"]
        frame = model_input["frame"]
        digests = case["expectedDigests"]
        assert (
            frame["source_revision"]["image_data_revision"]["image_source_digest"]
            == digests["imageSourceDigest"]
        )
        assert view["view_digest"] == digests["sourceViewDigest"]
        assert frame["inclusion_mask_digest"] == digests["inclusionMaskDigest"]
        assert frame["frame_digest"] == digests["frameDigest"]
        assert model_input["input_digest"] == digests["modelRasterInputDigest"]
        assert (
            backend_model_identity_digest(request["expected_backend"])
            == digests["backendModelIdentityDigest"]
        )

        view_preimage = copy.deepcopy(view)
        view_preimage.pop("view_digest")
        assert view["view_digest"] == canonical_digest(view_preimage)
        assert build_source_view(view_preimage) == view
        changed_creation = copy.deepcopy(view_preimage)
        changed_creation["artifact"]["creation_action_id"] = (
            "99999999-9999-4999-8999-999999999999"
        )
        assert build_source_view(changed_creation)["view_digest"] != view["view_digest"]
        frame_preimage = copy.deepcopy(frame)
        frame_preimage.pop("inclusion_mask_digest")
        frame_preimage.pop("frame_digest")
        assert build_raster_frame(frame_preimage) == frame
        assert (
            build_model_raster_input(view, frame, model_input["rasterizer"])
            == model_input
        )

        accepted = validate_inference_acceptance(request, _wire(case, "accepted"))
        pending = validate_inference_pending_response(
            _wire(case, "pending"), accepted["backend_call_id"]
        )
        cancellation = validate_inference_cancellation_acceptance(
            _wire(case, "cancellation"), accepted["backend_call_id"]
        )
        _assert_exact(case, "accepted", accepted)
        _assert_exact(case, "pending", pending)
        _assert_exact(case, "cancellation", cancellation)

        result = validate_inference_result(
            _wire(case, "result"),
            request,
            accepted["backend_call_id"],
            case["declaredClasses"],
        )
        _assert_exact(case, "result", result)
        assert result["meta"]["input_identity"] == build_backend_input_identity(
            model_input
        )

    empty = fixture["cases"][1]
    assert not _wire(empty, "result")["geojson"]["features"]


@pytest.mark.parametrize("schema_version", [1, 3])
def test_old_and_future_versions_are_rejected_by_every_finite_envelope(schema_version):
    case = _fixture()["cases"][0]
    for value, validate in (
        (_wire(case, "request"), validate_finite_inference_request),
        (_wire(case, "accepted"), validate_inference_accepted),
        (_wire(case, "pending"), validate_inference_pending),
        (_wire(case, "cancellation"), validate_inference_cancellation_accepted),
        (_wire(case, "result")["meta"], validate_backend_result_meta),
    ):
        value["schema_version"] = schema_version
        with pytest.raises(
            ValueError, match="unsupported finite inference schema_version"
        ):
            validate(value)


def test_validated_request_and_result_are_detached_from_input_trees():
    case = _fixture()["cases"][0]
    raw_request = _wire(case, "request")
    request = validate_finite_inference_request(raw_request)
    raw_request["raster_input"]["source_view"]["pixel_rect"]["x"] = 999
    assert request["raster_input"]["source_view"]["pixel_rect"]["x"] == 110

    raw_result = _wire(case, "result")
    result = validate_inference_result(
        raw_result,
        request,
        _wire(case, "accepted")["backend_call_id"],
        case["declaredClasses"],
    )
    raw_result["geojson"]["features"][0]["geometry"]["coordinates"][0][0][0] = 999
    assert result["geojson"]["features"][0]["geometry"]["coordinates"][0][0][0] != 999


def test_shared_backend_call_id_byte_boundary_covers_every_finite_envelope():
    fixture = _fixture()
    base = fixture["cases"][0]
    accepted = _wire(base, "accepted")
    pending = _wire(base, "pending")
    cancellation = _wire(base, "cancellation")
    meta = _wire(base, "result")["meta"]

    for case in fixture["backendCallIdCases"]:
        call_id = case["unit"] * case["repetitions"]
        assert len(call_id.encode("utf-8")) == case["utf8Bytes"], case["name"]
        if "canonicalStringBytes" in case:
            assert len(canonical_bytes(call_id)) == case["canonicalStringBytes"]

        values = []
        for source in (accepted, pending, cancellation):
            value = copy.deepcopy(source)
            value["backend_call_id"] = call_id
            values.append(value)
        result_meta = copy.deepcopy(meta)
        result_meta["backend_call_id"] = call_id

        validators = (
            lambda: require_backend_call_id(call_id),
            lambda: encode_backend_call_query(call_id),
            lambda: validate_inference_accepted(values[0]),
            lambda: validate_inference_pending(values[1]),
            lambda: validate_inference_cancellation_accepted(values[2]),
            lambda: validate_backend_result_meta(result_meta),
        )
        if case["valid"]:
            for validate in validators:
                validate()
        else:
            for validate in validators:
                with pytest.raises(ValueError, match="nonblank|UTF-8 bytes"):
                    validate()


def test_shared_backend_call_query_encoding_round_trips_through_fastapi():
    fixture = _fixture()
    assert BACKEND_CALL_QUERY_PARAMETER == fixture["backendCallQueryParameter"]
    app = FastAPI()

    @app.get("/v1/inference/calls")
    def read_call(backend_call_id: str = Query()) -> dict[str, str]:
        return {"backend_call_id": require_backend_call_id(backend_call_id)}

    client = TestClient(app)
    for case in fixture["backendCallAddressCases"]:
        call_id = case["unit"] * case["repetitions"]
        expected = (
            f"{fixture['backendCallQueryParameter']}="
            + case["encodedUnit"] * case["repetitions"]
        )
        query = encode_backend_call_query(call_id)

        assert query == expected, case["name"]
        response = client.get(f"/v1/inference/calls?{query}")
        assert response.status_code == 200, case["name"]
        assert response.json() == {"backend_call_id": call_id}, case["name"]


def test_request_identity_and_spatial_mutations_fail_closed():
    case = _fixture()["cases"][0]
    request = _wire(case, "request")

    unsupported_view = copy.deepcopy(request["raster_input"]["source_view"])
    unsupported_view.pop("view_digest")
    unsupported_view["plane"]["z"] = 1
    with pytest.raises(ValueError, match="only plane z=0, t=0"):
        build_source_view(unsupported_view)

    mutations = []
    wrong = copy.deepcopy(request)
    wrong["extra"] = True
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["schema_version"] = 1
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["action_id"] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["producer"]["step_id"] = "lumen"
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["expected_backend"]["rasterizer"]["rasterization_contract_digest"] = "a" * 64
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["raster_input"]["source_view"]["view_digest"] = "a" * 64
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["raster_input"]["frame"]["inclusion_mask_digest"] = "a" * 64
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["raster_input"]["frame"]["frame_digest"] = "a" * 64
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["raster_input"]["input_digest"] = "a" * 64
    mutations.append(wrong)
    wrong = copy.deepcopy(request)
    wrong["raster_input"]["frame"]["source_revision"]["image_data_revision"][
        "image_source_digest"
    ] = "a" * 64
    mutations.append(wrong)

    for mutation in mutations:
        with pytest.raises(ValueError):
            validate_finite_inference_request(mutation)

    with pytest.raises(ValueError, match="route module_ref"):
        validate_finite_route_request(
            request, "histoforge/lumen", case["route"]["modelId"]
        )
    with pytest.raises(ValueError, match="route model_id"):
        validate_finite_route_request(request, case["route"]["moduleRef"], "lumen")

    frame_preimage = copy.deepcopy(request["raster_input"]["frame"])
    frame_preimage.pop("inclusion_mask_digest")
    frame_preimage.pop("frame_digest")

    unsupported_plane = copy.deepcopy(frame_preimage)
    unsupported_plane["transform"]["raster"]["plane"]["z"] = 1
    pixel_source = unsupported_plane["source_revision"]["pixel_source"]
    pixel_source["z_slices"] = 2
    unsupported_plane["source_revision"]["image_data_revision"][
        "image_source_digest"
    ] = canonical_digest(pixel_source)
    with pytest.raises(ValueError, match="only plane z=0, t=0"):
        build_raster_frame(unsupported_plane)

    frame_preimage["inclusion_mask_raster"]["coordinates"][0][1][0] = 513
    with pytest.raises(ValueError, match="outside the target raster"):
        build_raster_frame(frame_preimage)

    wrong_scale = copy.deepcopy(frame_preimage)
    wrong_scale["inclusion_mask_raster"]["coordinates"][0][1][0] = 500
    scale = wrong_scale["transform"]["raster_to_slide"]["scale_x"]
    wrong_scale["transform"]["raster_to_slide"]["scale_x"] = math.nextafter(
        scale, math.inf
    )
    with pytest.raises(ValueError, match="target_mpp"):
        build_raster_frame(wrong_scale)

    slide_frame = copy.deepcopy(frame_preimage)
    slide_frame["inclusion_mask_raster"]["coordinates"][0][1][0] = 500
    slide_frame["region"] = {"type": "slide"}
    model_input = request["raster_input"]
    slide_input = build_model_raster_input(
        model_input["source_view"],
        build_raster_frame(slide_frame),
        model_input["rasterizer"],
    )
    slide_request = copy.deepcopy(request)
    slide_request["raster_input"] = slide_input
    with pytest.raises(ValueError, match="region must be roi"):
        validate_finite_inference_request(slide_request)


def test_acceptance_result_echoes_and_result_limits_fail_closed():
    case = _fixture()["cases"][0]
    request = _wire(case, "request")
    accepted = _wire(case, "accepted")
    result = _wire(case, "result")
    call_id = accepted["backend_call_id"]

    for field, replacement in (
        ("action_id", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        ("output_generation_id", "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
    ):
        wrong_accepted = copy.deepcopy(accepted)
        wrong_accepted[field] = replacement
        with pytest.raises(ValueError, match=field):
            validate_inference_acceptance(request, wrong_accepted)

    with pytest.raises(ValueError, match="addressed call"):
        validate_inference_pending_response(_wire(case, "pending"), "other-call")
    with pytest.raises(ValueError, match="addressed call"):
        validate_inference_cancellation_acceptance(
            _wire(case, "cancellation"), "other-call"
        )

    meta_mutations = {
        "action_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "output_generation_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "backend_call_id": "other-call",
        "input_id": 99,
    }
    for field, replacement in meta_mutations.items():
        wrong = copy.deepcopy(result)
        wrong["meta"][field] = replacement
        with pytest.raises(ValueError, match="result meta"):
            validate_inference_result(wrong, request, call_id, case["declaredClasses"])

    wrong = copy.deepcopy(result)
    wrong["meta"]["producer"]["step_id"] = "lumen"
    with pytest.raises(ValueError, match="result meta"):
        validate_inference_result(wrong, request, call_id, case["declaredClasses"])

    wrong = copy.deepcopy(result)
    wrong["meta"]["backend"]["gpu"] = "L4"
    with pytest.raises(ValueError, match="result meta"):
        validate_inference_result(wrong, request, call_id, case["declaredClasses"])

    wrong = copy.deepcopy(result)
    wrong["meta"]["input_identity"]["source_manifest_digest"] = "a" * 64
    with pytest.raises(ValueError, match="result meta"):
        validate_inference_result(wrong, request, call_id, case["declaredClasses"])

    wrong = copy.deepcopy(result)
    wrong["meta"]["coordinate_space"] = "SLIDE"
    with pytest.raises(ValueError, match="coordinate_space"):
        validate_inference_result(wrong, request, call_id, case["declaredClasses"])

    wrong = copy.deepcopy(result)
    wrong["geojson"]["features"][0]["properties"]["path_class"] = "Lumen"
    with pytest.raises(ValueError, match="not declared"):
        validate_inference_result(wrong, request, call_id, case["declaredClasses"])

    wrong = copy.deepcopy(result)
    out_of_bounds = _wire(case, "outOfBoundsGeojson")
    _assert_exact(case, "outOfBoundsGeojson", out_of_bounds)
    wrong["geojson"] = out_of_bounds
    with pytest.raises(ValueError, match="outside the target raster"):
        validate_inference_result(wrong, request, call_id, case["declaredClasses"])

    feature_bytes = len(canonical_bytes(result["geojson"]["features"][0]))
    for budget_field, limit in (
        ("max_features", 1),
        ("max_vertices", 7),
        ("max_record_bytes", feature_bytes - 1),
    ):
        bounded_request = copy.deepcopy(request)
        bounded_request["result_budget"][budget_field] = limit
        with pytest.raises(ValueError, match=budget_field):
            validate_inference_result(
                result, bounded_request, call_id, case["declaredClasses"]
            )


@pytest.mark.parametrize("axis", ["z", "t"])
def test_nonzero_source_view_plane_fails_even_with_valid_identity(axis):
    view = _wire(_fixture()["cases"][0], "request")["raster_input"]["source_view"]
    view.pop("view_digest")
    view["plane"][axis] = 1

    with pytest.raises(ValueError, match="only plane z=0, t=0"):
        build_source_view(view)
    with pytest.raises(ValueError, match="only plane z=0, t=0"):
        validate_source_view({**view, "view_digest": canonical_digest(view)})


@pytest.mark.parametrize("axis,dimension", [("z", "z_slices"), ("t", "timepoints")])
def test_nonzero_target_plane_fails_even_inside_multiplane_source(axis, dimension):
    frame = _wire(_fixture()["cases"][0], "request")["raster_input"]["frame"]
    frame.pop("frame_digest")
    frame["transform"]["raster"]["plane"][axis] = 1
    revision = frame["source_revision"]
    revision["pixel_source"][dimension] = 2
    revision["image_data_revision"]["image_source_digest"] = canonical_digest(
        revision["pixel_source"]
    )

    with pytest.raises(ValueError, match="only plane z=0, t=0"):
        validate_raster_frame({**frame, "frame_digest": canonical_digest(frame)})
    frame.pop("inclusion_mask_digest")
    with pytest.raises(ValueError, match="only plane z=0, t=0"):
        build_raster_frame(frame)


def test_result_budget_fields_are_positive_safe_integers():
    request = _wire(_fixture()["cases"][0], "request")
    for field in request["result_budget"]:
        wrong = copy.deepcopy(request)
        wrong["result_budget"][field] = 0
        with pytest.raises(ValueError, match=field):
            validate_finite_inference_request(wrong)

        wrong = copy.deepcopy(request)
        wrong["result_budget"][field] = 9_007_199_254_740_992
        with pytest.raises(ValueError, match=field):
            validate_finite_inference_request(wrong)
