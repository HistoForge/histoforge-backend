"""Administrative rectangles do not restrict biological annotation geometry."""

import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.training.datasets.io import read_training_roi_geometry
from shapely.geometry import mapping
from shapely.wkt import loads
from tests.test_training_dataset_content import _admit, _dataset, _object_feature


@pytest.mark.parametrize(
    "wkt",
    [
        "POLYGON ((10 10, 30 10, 10 30, 10 10))",
        "POLYGON ((20 10, 30 20, 20 30, 10 20, 20 10))",
        "POLYGON ((10 10, 30 10, 30 20, 20 20, 20 30, 10 30, 10 10))",
        "POLYGON ((10 10, 30 10, 30 30, 10 30, 10 10),"
        " (15 15, 15 20, 20 20, 20 15, 15 15))",
        "MULTIPOLYGON (((10 10, 15 10, 15 15, 10 15, 10 10)),"
        " ((20 20, 30 20, 30 30, 20 30, 20 20)))",
    ],
)
def test_dataset_admission_rejects_nonrectangular_administrative_rois(tmp_path, wkt):
    extracted = _dataset(tmp_path)
    _write_roi(tmp_path, wkt)
    with pytest.raises(ValueError, match="axis-aligned rectangle"):
        _admit(extracted)


@pytest.mark.parametrize(
    "wkt",
    [
        "POLYGON ((10.5 10, 30 10, 30 30, 10.5 30, 10.5 10))",
        "POLYGON ((10 10, 20 10, 30 10, 30 30, 10 30, 10 10))",
        "MULTIPOLYGON (((10 10, 30 10, 30 30, 10 30, 10 10)))",
    ],
)
def test_rectangular_area_is_accepted_without_changing_its_encoding(tmp_path, wkt):
    expected = _write_roi(tmp_path, wkt)
    assert read_training_roi_geometry(tmp_path, 0, 4096) == expected


def test_polygonal_biological_labels_remain_supported(tmp_path):
    geometry = canonical_geometry(
        mapping(loads("POLYGON ((12 12, 20 12, 12 20, 12 12))"))
    )
    feature = _object_feature("11111111-1111-4111-8111-111111111111", geometry)
    admitted = _admit(_dataset(tmp_path, objects=[feature]))
    assert not admitted.roi_label_summaries[0].is_negative


def _write_roi(root, wkt):
    geometry = canonical_geometry(mapping(loads(wkt)))
    path = root / "rois/0/roi.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_bytes({"type": "Feature", "geometry": geometry, "properties": {}})
    )
    return geometry
