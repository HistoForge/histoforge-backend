# Scored point results

HistoForge extension 0.2.4 and backend 0.1.5 support scored points from finite
`dinov3-adapter-point` inference through QuPath save and Metrics. Detection
requires an explicit recipe and an FP32 point checkpoint. Managed training,
BF16, large-region detection and a MIDOG module are outside this support.

## Recipe and finite execution

Detection recipes require `schemaVersion: 4`. Their output is
`point_detection`, with contiguous `classChannels` from zero, one independent
sigmoid channel per emitted PathClass. Background is implicit; `background` and
`training_only` channels are rejected. Scores are not an argmax across classes.

`dinov3-adapter-point` requires `input.type: field` and an explicit existing
`variable-field-v2` profile. Its complete runner-native parameters are:

| Parameter | Meaning |
| --- | --- |
| `num_classes` | Positive channel count, equal to `output.classChannels` length |
| `score_threshold` | Required finite value in `[0,1]`; retain a returned float32 score exactly when it is at least this value |
| `peak_radius_model_px` | Required positive integer; square neighborhood radius used to select class-specific peaks |
| `batch_size` | Optional neural-field batch size; runner default is 8 |

The strict loader constructs the point head and checks every checkpoint key,
shape, dtype and head identity. It does not convert a segmentation checkpoint.
Inference uses FP32, normalizes real pixels once, blends raw overlapping field
logits, and decodes peaks once for the complete supplied raster. A connected
equal-valued plateau emits its first pixel in row-major order; a plateau touching
a higher value or losing a neighborhood comparison emits nothing. Unsupported
pixels neither emit nor suppress a peak. Returned centers are pixel centers,
`(column + 0.5, row + 0.5)`.

The caller's smaller feature/vertex cap bounds peak extraction. A count overflow
raises `PointDetectionLimitExceeded`, translated to `RESULT_LIMIT_EXCEEDED`.
Record and total byte caps are enforced by the shared feature builder and finite
response encoder. No result is truncated. Other malformed or nonfinite outputs
remain `RUNNER_FAILED`.

Only `FINITE_CALL` and `POINT_DETECTION` are advertised by this source runner.
Slide/oversized targets require unsupported durable execution and fail before
dispatch; `input.largeRegion` is rejected. Training availability remains
unavailable. Registry resource limits are admission policy, not GPU qualification.

The paired handshake uses module schema 4 and finite schema 3. Published schema-3
segmentation recipes remain readable and executable without rewriting their
bytes or digests. They cannot declare point detection. Schema-2 snapshots retain
their existing inspection-only path. Old clients/backends require a matching
update; no wire fallback occurs. Source, durable, training and saved-project
schemas are unchanged. A future release must use new package versions and pass
upgrade qualification; existing public release files remain immutable.

MIDOG's research scorer uses strict `score > threshold`; this runner uses
`score >= threshold`. A selected operating point must be translated and verified
at exact float32 ties before module publication. Native QuPath coordinates never
receive the MIDOG dataset's annotation-specific half-pixel conversion.

## Records and identity

One result is one GeoJSON `Point` with two finite raster-local coordinates and
exactly these properties:

```json
{"object_type":"DETECTION","path_class":"Mitotic figure","score":0.5}
```

The class must be declared by the model. The score is required, finite, in `[0,1]`
and exactly representable as float32. It is the model's score, not a calibrated
probability. The trusted Python builder converts NumPy real scalars to Python
numbers; wire validation does not accept strings or booleans as numbers.

The feature ID is lowercase SHA-256 over JCS:

```text
[pathClass, [x, y], score, occurrenceIndex]
```

The occurrence index starts at zero for each identical class, coordinate and score
tuple. Coincident results retain separate IDs. Records are sorted by ID.
Each point consumes one vertex; byte limits include its score. Empty collections
still require their complete JSON envelope. Durable stream framing and compression
are unchanged. Existing polygon records, IDs and default admission are unchanged.

## Admission and native storage

1. Require the raster-local center inside `[0,width) × [0,height)`.
2. Transform its original binary64 XY into slide coordinates.
3. Retain it only if the exact source parent, when present, and target both
   `cover` that position. A boundary point is included. Points are never clipped
   into outlines or filtered by positive-area manual overlap.
4. Convert XY once to the float32 coordinates used by QuPath's native `PointsROI`
   serialization. Never repeat admission after that projection.
5. Stage one native detection per point. Preserve its UUID, class, source,
   generation and score through ordinary hierarchy commit and public project save.

The numeric measurement `HistoForge: Detection score` stores the score separately
from QuPath class probability. Zero is present; a manual point with no score stays
missing. Ordinary recipe scripts may change declared classes, not scores or
coordinates. Point objects cannot supply area crops or area/shape measurements.

QuPath's generic point-to-JTS conversion rounds to a 0.01-pixel grid. Scientific
spatial reads instead use native point coordinates directly. That conversion
preserves every mark in a native counting container, but recipe and Metrics
population capture admits only singleton point ROIs. Empty and multi-mark counting
annotations remain untouched and excluded; this slice does not assign all their
marks to one region using an arbitrary representative or centroid.

## Metrics and save/reopen

Detached Metrics inputs retain singleton XY and optional score. The point-only
binary input record binds coordinates, plane, score presence and value; existing
polygon encodings remain byte-identical. Reconstructed script objects receive the
same native point and score. Counts, region-area density and score distributions
use the existing metric schema and workbook format; no marker area is invented.

Logical ROI selection still follows provenance. Later slide/annotation-union
Metrics selection and biological-parent normalization use the current saved
geometry, as specified by the Metrics contract. Float32 projection can move a
center across an oblique boundary: original inference admission and a later
geometric selection can therefore differ. No tolerance or hidden raw-coordinate
store overrides that rule. Save/reopen must preserve the projected coordinates,
score, IDs, provenance, generation records and complete image revision.

Python `features.py`/`feature_stream.py` and Java `InferenceFeatureCodec`/
`FeatureStreamDecoder` independently enforce record identity. Shared synthetic
fixtures live at `tests/fixtures/contracts/inference-point-features-v1.json`
and the extension's matching test resource. Native integration tests cover
staging, public save, a fresh JVM read, metrics, empty Redo and failed-result
preservation. These tests establish software behavior, not model accuracy.
`point-finite-inference-v1.json` additionally binds recipe, request and result
bytes across Python and Java, including empty results and coincident scored points.
