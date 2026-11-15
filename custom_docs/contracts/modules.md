# Module format

A module contains a recipe, scripts, and model artifacts. It is stored in a
private HistoForge workspace or published to Hugging Face, then installed in the
user's Modal workspace.

This document defines recipe schema 3 and its file layout. Catalog, inference,
and training API records are separate contracts. A valid recipe does not imply
that every model can run: the installed backend must provide its declared runner
and capabilities.

---

## 1. The unit — one type

There is **one type**: a **module** = one self-contained module content root. "model" and
"pipeline" are *shapes* read from the recipe, one step versus many, never a stored
discriminator. There is no `moduleType`, and no model-versus-pipeline branch anywhere in
the code.

A published module content root is one Hugging Face repository at an exact commit. A
successful training run instead creates one private, commitless content root at the
reserved `workspace/run-<id>` ref. That workspace module is create-only and immutable;
its content digest and backend-private training attestation provide its exact identity.
Publication is therefore a storage and identity transition, not what makes the content a
module.

A module is **self-contained**: it owns every weight that it may redistribute as a
vendored file, so there is no separate installed copy to drift against. A gated or
non-redistributable external weight remains **SHA-pinned** as `referenced`. Published
content is versioned by its Hugging Face commit; a run-derived workspace module is
versioned by its immutable content digest and attestation.

---

## 2. Module content layout

The published Hugging Face repository root is the module content root. File-list display
order is not an execution order: `recipe.json` declares the models and ordered steps.

```
<owner>/<name>/                  e.g. histoforge/leydig or workspace/run-<id>
├── recipe.json                  # the whole module definition (header + models + steps)
├── <model-id>.safetensors       # one vendored weights file per vendored model
├── <script>.groovy              # one file per script step
├── training-report.json         # optional canonical report declared by trainingReportDigest
├── README.md                    # optional HF model card, retained unchanged
└── …                            # all other repository files, retained unchanged
```

- **One file defines the module**: `recipe.json`. (There is no separate `manifest.json` —
  the display header is folded in; see §3.1.)
- **Weights are vendored files**, one per model, named `<model-id>.safetensors` (§9).
  A referenced foundation encoder is **not** a file here — it is a SHA-pinned pointer
  fetched once to the Volume and shared (see `weights.referenced`, §4).
- **Installation retains every repository file**, including licence notices,
  training-source attribution, images, unused scripts, and undeclared checkpoints.
  Only the recipe declares executable steps and model artifacts; retaining a file
  does not make it executable. Keep `README.md` at the root
  so Hugging Face renders it as the card; link any supporting files from that card.
  Supporting files may be grouped in subdirectories, but no `module/` wrapper goes
  around `recipe.json` and its declared content.
- The Volume mirrors this path: `/modules/<owner>/<name>/…` using
  `Path("/modules") / ref`. Supporting files stay in that same directory, not a
  separate document store. The backend inventories and hashes all retained files;
  authors maintain no supporting-file list. Normal path-safety and total module-byte
  limits apply before download. The `.histoforge/` namespace is backend-private and
  cannot occur in the source repository. The reserved `training-report.json` still
  requires its recipe declaration. Saved README presentation is described in §7.

---

## 3. `recipe.json`

Annotated skeleton. Every field is defined in §3–§4.

```jsonc
{
  // ── module header ──────────────────────────────────────────────
  "schemaVersion": 3,
  "name": "Leydig",                       // human display name
  "description": "…",                     // optional — one-line subtitle for display + search
  "upstream": "histoforge/leydig@<sha>",  // optional: exact immediate source; absent on originals
  "trainingReportDigest": "<sha256>",     // optional: exact reserved training-report.json bytes

  // ── models: servable units, keyed by LOCAL id ─────────────────
  "models": {
    "<model-id>": {
      "arch": "dinov3-adapter-cellpose",  // → a curated runner (registry, §4)
      "gpu": "L40S",                      // per-model GPU tier (honored at spawn, §4)
      "weights": { … },                   // where the weights come from (§4)
      "input":   { … },                   // required model raster and crop framing (§4)
      "output":  { … },                   // what the model emits (§4)
      "params":  { … },                   // runner-native config, optional (§4)
      "backbone":{ … }                    // OPTIONAL — composable-head archs only (§4)
    }
  },

  // ── steps: execution order (client-orchestrated) ──────────────
  "steps": [
    { "model": "<model-id>" },
    { "model": "<child-id>", "parentModel": "<parent-id>", "parentClass": ["<Class>"] },
    { "script": "metrics.groovy" }
  ]
}
```

### 3.1 Module header

| Field | Req | Type | Notes |
|---|---|---|---|
| `schemaVersion` | ✓ | int | Contract version. `3` today. |
| `name` | ✓ | string | Human display name. |
| `description` | – | string | Short description for plugin search and the documentation fallback. Not a copy of the README body. |
| `upstream` | – | string | `<owner>/<name>@<sha>`: exact immediate source module. Absent on originals. It records lineage; this contract does not define automatic update or merge behavior. |
| `trainingReportDigest` | – | SHA-256 | Digest of the exact UTF-8 JCS bytes in the reserved `training-report.json` file. Presence requires that one file; absence forbids it. |

There is **no `id` field**. A module is addressed by its module ref: an
`<owner>/<name>` Hugging Face repository or the reserved `workspace/run-<id>` ref. Its exact
version is the resolved Hugging Face commit or, for a private run-derived result, its
immutable content digest and training attestation. A model is identified by its key in
`models`. There is **no `license` or `tags`** field in the recipe. Discovery metadata
belongs in published README frontmatter; licence texts and notices remain ordinary
repository files (§7). There is also no `author` field. Human attribution belongs in the
Hugging Face repository and README when published; executable lineage belongs in
`upstream`, the training report where applicable, and the exact module content identity.

`training-report.json` is immutable module evidence, not executable code. Its bytes
are exactly `JCS(TrainingReport)` using the closed record owned by the
[training report validator](../../histoforge/cloud/training/runs/report.py), and
`trainingReportDigest` equals their SHA-256.
The fixed filename avoids a second path field. Recipe validation includes the
report in the complete file closure and rejects a missing, extra, noncanonical, or
digest-mismatched report. Because the digest is inside `recipe.json`, the normal
recipe and content digests bind the report without adding it to the backend's
`executableFiles` list. A successful training finalizer replaces any older report;
normal publication and installation preserve and revalidate it.

A module-owned training report must describe a successful result, not merely have
valid bytes. Validation requires `state=SUCCEEDED`, an absent `failure`, and a
present `model_selection`. Its `selected_model_id` must name a model in this
recipe, that model's trained primary weights must be vendored module bytes, and
their SHA-256 must equal `model_selection.selected_weight_digest`. These checks
apply equally to a private workspace result and an installed Hugging Face module.
They prevent a digest-valid failed report or unrelated selected weights from being
presented as trained-module evidence.

The report's scientific task must also describe that selected model. Its
`task.input_kind` and `task.model_microns_per_pixel` equal the model's
`input.type` and `input.micronsPerPixel`, and `task.model_output` equals the
model's complete normalized `output` record. The report therefore preserves the
exact channel, role, and PathClass meaning used by training, but never owns or
changes runtime behavior.
For custom-class training this is the **target** schema and resolution. The
report separately records `source_model_output`, `source_model_microns_per_pixel`
and `class_initialization`; they describe the immutable starting model, not the
result's runtime contract. Its annotation mapping is training provenance, not an
inference-time remapping operation.
When the model declares
`input.fieldProfile`, `report.field_profile` is that exact record. When it does
not, the report carries a fixed profile and ordinary selected-runner validation
must prove that profile equals the runner's resolved fixed contract before the
model can execute. A missing runner leaves the schema-valid model catalogable but
unavailable, as it does for every other runner-owned parameter contract. These
bindings prevent valid report bytes for a different task, resolution, or field
planner from being attached to the right weight file.

`result_display_name` and `result_form` describe the module produced by that
training run. They are provenance, not invariants of every later module that may
preserve the report while being renamed or composed. Validation therefore does
not compare them with the current recipe shape or display name.

`result_inference_gpu` is different: it records the concrete recipe default chosen
for the trained model. Validation requires it to equal that selected model's current
`gpu`. A later global inference override is action state and never rewrites this
field or the recipe.

### 3.2 `models` — the map

An object keyed by **local model-id** (short, `snake_case`, unique within the module). Each
value is a **model entry**. Every entry — head, fused net, or FM-backed classifier alike —
has its own `arch` and I/O contract; each is a model-shaped thing.

Model-ids are **local**: `steps`, `parentModel`, and the inference route reference these keys
— never other modules, never HF refs (§3.3, §5). The only HF ref inside a module is a pinned
encoder in `weights.referenced` (§4).

### 3.3 `steps` — execution order

An ordered list. Three shapes:

```jsonc
{ "model": "<id>" }                                              // plain — run on the ROI
{ "model": "<id>", "parentModel": "<id>", "parentClass": […] }   // region-gated (two-wave)
{ "script": "<file>.groovy" }                                    // groovy — at its step position
```

- **Plain `{model}`** — run this model over the region, in the first wave.
- **Region-gated `{model, parentModel, parentClass}`** — run this model within each eligible
  polygonal parent whose class ∈ `parentClass`. An eligible parent is either output attributed to
  `parentModel` or an unattributed manual/foreign-tool object; output attributed to another
  HistoForge producer cannot substitute. A manual parent remains unattributed: it is an input, but
  never counts as `parentModel` output for Fill or enters that producer's Redo replacement set. The
  plugin defines each recipe-framed parent source view and target grid; the backend rasterizes and
  runs the child on that crop. This is the two-wave executor; the crop is shaped by the child
  model's `input` = `crop` (§4). `parentModel` is a **local model-id** of a plain (wave-1) step and
  defines the producer dependency; `parentClass` is a list of PathClass names. Depth 1 only — a
  `parentModel` must itself be a plain step.
- **Ordinary script `{script}`** — a Groovy file other than the reserved `metrics.groovy`, run
  **at its declared position** in the list (not forced last). Before a later region-gated step it
  can re-classify parents; for example, `cross_section.groovy` marks
  `Tubule (Cross Section)` for the germ-cell step to gate on. Its staged population includes
  earlier model output plus unattributed polygonal objects whose current class belongs to an
  earlier model's emitted output classes or an ordinary script's class header at or before this recipe
  position. Those manual inputs keep their absent provenance.
- **Metric script `{script: "metrics.groovy"}`** — the reserved pure metric step. It occurs at
  most once and, when present, is the final step. It computes from the completed region population
  and does not mutate recipe output.

`parentModel`/`parentClass` are a matched pair naming the producer dependency and accepted parent
classes, including a deliberately unattributed manual substitute. The *how-to-crop* half (padding,
size filter) rides the child model's `input` (§4).

---

## 4. Field contracts

### `weights` — where a model's weights come from

The **vendor/reference discriminator rides the weights**, not the model or module. It is a
**license (redistribution) decision**: can we re-host the bytes → `vendored`; gated /
NoDerivatives / no-rehost → `referenced`.

```jsonc
{ "type": "vendored",   "file": "<model-id>.safetensors" }              // a file in this module content root
{ "type": "referenced", "ref": "MahmoodLab/UNI2-h", "revision": "<sha>" } // external, SHA-pinned, shared
{ "type": "local",      "ref": "local/<name>" }                        // Volume working form only
```

- **`vendored`** — the module owns the weight file in its content root. This does not
  imply Hugging Face publication. Both an immutable private `workspace/run-<id>` module
  and a published repository can contain vendored weights. It is the self-contained
  default for user-trained, permissive, and otherwise redistributable weights.
- **`referenced`** — an external HF repo, **SHA-pinned via `revision`** (pinned = content-exact
  = no drift; unpinned is forbidden). For gated foundation encoders (UNI2-h, Virchow2, …). A big
  referenced encoder is fetched **once** to the Volume and **shared/deduped by ref** across every
  module that references it — one shared copy vs N vendored copies.
- **`local`** — points at separate Volume bytes that the module does not own. It is a
  non-executable working form for manual authoring or intermediate development. A
  successful training run does not create a `local` module: checkpoints remain run
  artifacts, and finalization copies the selected weights into an immutable workspace
  module as `vendored`. Publishing an ordinary local working form materializes it into
  `vendored` (§6). `local` never appears in a published repository.

### `input` — the model raster required from supplied source pixels

```jsonc
"input": {
  "type": "field",              // "field" (whole region) | "crop" (cut from a parent instance)
  "micronsPerPixel": 0.22,      // mpp the model trained at; HistoForge rasterizes to this grid
  "fieldProfile": { … },        // optional neural-field planner; absence keeps fixed-field behavior
  "largeRegion": {              // optional processing cores, distinct from neural-network fields
    "coreWidthModelPx": 8192,
    "coreHeightModelPx": 8192,
    "haloPerSideModelPx": 256
  }
  // crop-only, OMITTED on field models:
  // "framing":        { "paddingModelPx": 64 },
  // "acceptCriteria": { "sizeUmRange": [40, 400], "maxAspectRatio": 3.0 }   // forward — not wired yet
}
```

| Field | Where | Notes |
|---|---|---|
| `type` | all | `field` = run on the ROI/region. `crop` = run on a cut from a parent instance (a region-gated child, §3.3). |
| `micronsPerPixel` | all | Target pixel size in µm. The backend rasterizer produces this exact square-pixel grid from the plugin-selected source view. |
| `fieldProfile` | optional | Exact variable-field model planner below. Absence retains the runner's existing fixed-profile contract. |
| `largeRegion` | optional, field or crop | Recommended processing-core dimensions and context per side, in model-resolution pixels. Absence permits finite-size inputs only; it does not select hidden defaults. |
| `framing` | crop only | `{paddingModelPx}` — add N pixels of real slide context at the model's target MPP, convert that physical distance independently to source X/Y pixels, and edge-clamp. The runner then applies its declared field planner. Current fixed `histoforge-models` wrappers normalize the logical crop and pad a smaller tensor with numeric zero in normalized space; this is not white RGB, additional slide context, or `framing`. |
| `acceptCriteria` | crop only, **forward** | `{sizeUmRange:[min,max], maxAspectRatio}` — skip parents too small/large/elongated to run the child on. Deferred: no shipping module declares one (spg included), so the plugin `ModelSpec` record omits it until one needs it. |

There is **no `channels`** field — model inputs are 3-channel RGB produced by the pinned backend
rasterizer from the plugin-supplied source artifact.
A real multi-channel/multiplex input spec is introduced (and *wired*) when that arc is built.

`framing`/`acceptCriteria` are **omitted** on `field` models (absence = not applicable), never
written as `null`.

`largeRegion` is a closed record with exactly the three fields shown above. Core dimensions are
positive signed 32-bit integers; halo is a nonnegative signed 32-bit integer. Zero halo is valid.
Null, decimal number spellings, booleans and unknown fields are rejected. These processing sizes
need not be multiples of the neural-field alignment. `largeRegion` and `fieldProfile` may coexist;
neither replaces crop `framing` or changes the model's target MPP.

The recipe owns these recommendations. The deployed runner still owns support, resources and
limits: declaring this record cannot enable an unsupported large-region method or raise any cap.
For a supported model, the backend combines the recipe values and its method/evidence into the
advertised CoreHalo profile. QuPath's **Inference settings…** reads the resulting default halo and
read-only core dimensions. Its project-specific Custom halo is captured separately, never written
back into the module. Requests and recovery bind the recipe, resolved profile and any override.
Changing the record therefore changes model identity without changing weight bytes.

The maintained Cellpose examples use 8192 × 8192 cores and 256 pixels per side; HoVer examples use
8192 × 8192 and 512 per side. These are model-author recommendations, not a claim that every image
or checkpoint has been qualified at those settings. Fine-tuning and publishing preserve the source
record, including its absence. Training does not invent large-region defaults.

Schema 3 is a paired backend/extension change. Older schema-2 module recipes must be explicitly
updated and reinstalled before use with this release; the installer does not rewrite pinned
content. Saved schema-2 action snapshots remain readable for inspection and abandonment, with
their original identity and captured profile. The existing exact-extension-build fence prevents
resuming those old actions after an upgrade. This is not a second live recipe protocol.

`fieldProfile` is one closed optional record:

```jsonc
{
  "planner": "variable-field-v2",
  "minSideModelPx": 256,
  "maxSideModelPx": 1024,
  "preferredTileSideModelPx": 1024,
  "alignmentModelPx": 16,
  "outerArtificialPaddingModelPx": 8,
  "smallRoiContext": true,
  "tileOverlap": 0.25,
  "aggregation": "<qualified-id>",
  "preprocessing": "<qualified-id>"
}
```

`planner` is the sole discriminator and is exactly `variable-field-v2` in schema 3.
All side and alignment values are positive integers, every side is divisible by
`alignmentModelPx`, and
`minSideModelPx <= preferredTileSideModelPx <= maxSideModelPx`. Height and width
may vary independently inside that range. `tileOverlap` is a finite binary64 value
in `[0, 1)`. `aggregation` and `preprocessing` are nonblank bounded versioned IDs
validated by the deployed runner. `smallRoiContext` permits a small direct field ROI
to read nearby real pixels while supervising and committing only inside the ROI. It
must be `false` for `type=crop`, where `framing.paddingModelPx` is the sole real-slide
context owner.

`outerArtificialPaddingModelPx` is required and exactly `8`. It is immutable model
content, not a user setting. Complete-ROI validation, finite inference, and
core/halo inference center the logical target in a virtual eight-model-pixel border
before using the ordinary per-axis field planner. The virtual border does not read
neighboring source pixels, enter normalization statistics, enlarge the returned
target, or own output. It supplies normalized-zero input only where a planned field
extends beyond the logical target. Optimizer training fields continue to be sampled
from the natural supervised raster; complete held-out validation uses the normal
bordered inference path. This field is distinct from both real-slide context
(`smallRoiContext` or crop `framing`) and centered minimum/alignment tensor padding.

Presence of `fieldProfile` selects variable-field behavior and forbids legacy
`params.bsize` and `params.tile_overlap`. Absence selects the existing fixed runner
contract and requires any fixed-size fields owned by that runner. No second `kind`
or per-inference override exists.

### `output` — what a model emits

An unclassified instance model has no neural class head. It declares the one
PathClass applied after instance decoding:

```jsonc
"output": {
  "type": "instance_segmentation",
  "classes": ["Lumen"]
}
```

A model with a class head declares every tensor channel explicitly:

```jsonc
"output": {
  "type": "instance_segmentation_and_classification",
  "classChannels": [
    { "channel": 0, "role": "background" },
    { "channel": 1, "role": "emitted", "pathClass": "Tubule" },
    { "channel": 2, "role": "training_only", "pathClass": "Other" }
  ]
}
```

`type` is one of `instance_segmentation`,
`instance_segmentation_and_classification`, `dense_classification`, or
`region_classification`.

`instance_segmentation` requires exactly `type` plus one `classes` value and
forbids `classChannels`. The class is not a tensor channel.

Every other output type requires `classChannels` and forbids `classes`. Each
record owns an explicit nonnegative `channel`; array position has no meaning.
Channels are unique and contiguous from zero, PathClasses are unique, and at
least one channel is `emitted`. A `background` record has no `pathClass`; an
`emitted` or `training_only` record requires one. At most one background channel
is permitted by the generic module schema. Current DINOv3 class-head runners
further require background at channel zero. A future classification runner whose
head has no background can instead declare every channel as emitted or
training-only.

The runner makes its normal decision over the complete head, then returns only
winners whose role is `emitted`. Background and training-only winners are
suppressed; removing them before argmax or instance voting would change the
prediction. Parsers normalize records by `channel`, so authored array order cannot
change behavior or the normalized recipe/model-config digest. Exact authored
module bytes still retain their ordinary content identity.

### `backbone` — nested encoder (composable-head archs only)

```jsonc
"backbone": {
  "arch": "vit-encoder",
  "weights": { "type": "referenced", "ref": "MahmoodLab/UNI2-h", "revision": "<sha>" }
}
```

A **compute dependency**: the encoder runs first → features feed the head → **one forward pass,
one served result, one step**. Distinct from `parentModel` (a *spatial* dependency — separate
inference calls).

- Present **only** on composable-head archs (`abmil`/`mlp`/`linear` on a foundation encoder),
  where the same head can sit on different encoders, so the encoder must be named + pinned.
- **Fused `dinov3-adapter-*` models OMIT it** — their encoder is baked into the weights and named
  by the arch, so there is nothing to swap or reference.

### `params` — runner-native config (optional)

A pass-through bag of the runner's own tuning knobs. **This is the one `snake_case` island**
(everything else in the recipe is `camelCase`) — deliberately, because `params` maps 1:1 to the
Python runner's kwargs. The boundary is a rule (§9), not drift.

```jsonc
// dinov3-adapter-cellpose:  { "num_classes": 6, "bsize": 1024, "batch_size": 1,
//                             "min_instance_pixels": 15 }
// dinov3-adapter-hovernet:  { "num_classes": 3, "bsize": 1024, "tile_overlap": 0.25,
//                             "edge_threshold": 0.5, "fg_threshold": 0.15,
//                             "object_size": 2000, "morph_kernel": 21,
//                             "min_instance_pixels": 0 }
```

Discipline (§9): **each runner owns and documents its own `params` keys, and the recipe carries
nothing the runner doesn't read.** `num_classes` is the one spelling everywhere (no `nclasses`
alias). For current class-head runners it must equal the number of
`output.classChannels`; for a pure instance model it remains zero. It is a
runner-native construction parameter validated against the output contract, not
a second source of class meaning. `params` never carries `backbone` — the encoder
is in the arch name.

`batch_size` is specifically the number of runner-internal model tiles stacked into one GPU
forward pass. It is not a number of user actions, images, ROIs, logical inputs, or backend
processing windows; changing it controls GPU memory and throughput, not scientific scope. An
explicit value is part of the exact model entry and therefore `modelConfigDigest`. If omitted, the
version-pinned runner owns the default through its build/behavior manifest.

`min_instance_pixels` is required for the current Cellpose and HoVer instance runners. It is the
minimum instance area in model-grid pixels. Cellpose preserves upstream cleanup order after
dynamics and flow-quality filtering: remove undersized masks, fill holes, then remove undersized
masks again. HoVer applies the threshold to final watershed instances. Both complete cleanup
before class voting or feature conversion. Zero disables these size filters, not Cellpose hole
filling or other architecture-owned decoding. The value is model behavior, so it contributes to
`modelConfigDigest`; it is never inferred from image resolution or supplied as a per-inference
override. HoVer's `object_size` remains a separate watershed-marker parameter and does not replace
its final instance-area filter. Core/halo processing exempts provisional artificial-window-edge
fragments from size filtering until ownership and completeness are checked; real target edges
receive no exemption.

### `arch` → runner (the registry)

`arch` maps to a **curated runner baked into the Modal image** — our code (timm / transformers /
torch / clean-room), never fetched per-module, never `trust_remote_code` on user content
("models are data, not code"). The catalog still validates and returns a module whose `arch` has no
deployed runner; that model is marked unavailable. Activation or inference requiring it fails with
the architecture contract's typed unsupported-capability result rather than rejecting the module
content or relying on a diagnostic HTTP message.

| `arch` | Runner | Status |
|---|---|---|
| `dinov3-adapter-cellpose` | `CellposeRunner` | live |
| `dinov3-adapter-hovernet` | `HovernetRunner` | live |
| `dinov3-adapter-semantic` | `SemanticRunner` | live |
| `vit-encoder` | ViT foundation encoder | forward |
| `abmil` · `mlp` · `linear` | pooling / MLP / linear heads | forward |
| `torchscript-classifier` | curated TorchScript (wsinfer) | forward |
| `ctranspath` | clean-room Swin | forward |

The model's nonblank `gpu` value is its recipe-owned default. The plugin setting
**Default inference GPU** initially uses **Use model setting (recipe)**; a user may instead choose
one HistoForge-curated Modal GPU override. The selected action resolves this preference once and
stores the exact result in `BackendModelIdentity.gpu`. A later settings change cannot affect a
running or recovered action, and dispatch never falls back silently if Modal rejects the resolved
GPU. The recipe model and `modelConfigDigest` remain unchanged by an override.

The backend accepts only the recipe value or a value in its small versioned inference-GPU set,
then applies the resolved value as both a Modal class option and a class parameter so GPU is part
of warm-container identity and worker provenance. `arch` still selects runner code and
capabilities rather than a per-architecture GPU allowlist. The override list is a provider/resource
choice, not a guarantee that every model fits every offered device; Modal or runner startup may
reject an insufficient configuration explicitly. Build-owned CPU, memory, concurrency, timeout,
and default GPU values provision the base runner class.

**Settings → Inference precision** is a global user preference, not a recipe field.
**Default** uses the checkpoint's native precision through the architecture's
supported loader. Current DINOv3 loaders require FP32 tensors and reject a dtype
mismatch rather than casting it. **BF16 (if applicable)**, initially selected,
uses BF16 for the network forward when the runner and selected GPU support it;
otherwise the new action captures Default. Weights and returned prediction fields
remain FP32 on the current runners. BF16 can change individual results and is not
an accuracy claim. Unsupported execution fails; it does not retry in Default.
The effective runner build is part of the saved action identity. Training, existing
actions, checkpoint files and saved results do not change with this preference.

A managed training request separately selects `result_inference_gpu`. The source
model's exact `gpu` is its initial and always-admissible value; another value must
belong to the same curated inference-GPU set. Finalization writes that value to the
trained model's `gpu` and binds it through the training report. This is independent
of both the training worker GPU and the user's global inference override.

`dinov3-adapter-*` = **DINOv3 ViT-L/16 + fixed five-level 256-channel LN-SPM + four
bidirectional CTI stages + stride-4 W2 readout → task head** (Cellpose flow, HoVer, or dense
semantic classification). There is no selectable decoder or adapter variant. Architectural
switches do not belong in `params`; a materially different architecture requires its own curated
runner key and checkpoint contract.

---

## 5. Serving — one model within a module

```
POST /v1/inference/<owner>/<name>/<model-id>
```

Three segments, always. The gateway loads that model's weights and picks the runner by its `arch`;
the internal warm pool keys on `(module ref, model-id, installed content digest)`. The digest is an
execution fence, not a URL segment: the public target remains the three segments above, while a
reinstalled ref cannot reuse a container holding different bytes. A module is **never** a single
inference target — the client `RecipeExecutor` orchestrates the module's `steps` (multi-model +
region-gating + scripts). A nested `backbone` runs as the encoder stage of its head's one forward
pass, inside that model's container.

`GET /v1/modules` returns each installed module's `recipe.json` + script sources inline (weights
stay on the Volume), so the plugin renders and runs without extra fetches.

**Single-model modules** are served the same way: `…/leydig_nuclear/leydig_nuclear`. The visual
repetition is only the same-name coincidence (module named after its sole model); the moment a
model lives in a differently-named or multi-model module it is gone. No special-casing.

---

## 6. Publish — the vendor-transform

Publish commits the retained module repository in **one atomic `create_commit`**:
`recipe.json`, its declared scripts and weights, the declared training report when
present, and supporting files under the preservation rules below. The recipe owns
the model vendor transform; supporting files need no extra manifest.
Per model:

```jsonc
{ "type": "local", "ref": "local/foo" }        →  { "type": "vendored", "file": "foo.safetensors" }  // read Volume, add file
{ "type": "referenced", "ref": "…@sha" }       →  unchanged (left pinned)
{ "type": "vendored", "file": "…" }            →  re-added
```

The `local -> vendored` transform applies only to an ordinary authoring working form.
A completed `workspace/run-<id>` training result already owns vendored weights and is
published without changing that discriminator.

Publish is **pure HF by default** — it reads the Volume and commits to HF, without writing the
Volume. Publication snapshots every retained source file, applies the vendor transform,
attempts `create_repo(exist_ok=False)`, then resolves the target's current head and file tree
regardless of whether creation returned success or `409`.

| File ownership | Publication rule |
|---|---|
| Incoming `recipe.json`, declared weights/scripts, declared training report | Write the prepared module's bytes. |
| Any other source file, absent from target | Copy unchanged. |
| Any other source file, already present at the same target path | Preserve the existing HF bytes unchanged. |
| Target-only supporting file | Preserve unchanged. |
| Weight declared by the previous target recipe, absent from the prepared result | Delete that exact path. |
| Reserved training report, no longer declared by the incoming recipe | Delete that exact path. |

README, licences, notices, images, and unused scripts all follow the same supporting-file rule.
There is no document parser, suffix-based deletion, or duplicate-document namespace. In particular,
an HF-side licence edit wins over a different source copy at the same path. The publisher owns the
review of those terms and of inherited documentation; HistoForge does not assess compatibility.
If neither source nor target contains `README.md` and the target also lacks `recipe.json`, the
commit adds the supplied fallback card without assigning a licence. Otherwise no card is generated.
The merged file tree and total byte size are validated before commit.

An opt-in **sync-locally** checkbox installs the complete target from the **returned HF commit**,
not source staging: retained HF-side files may differ from the source. The normal downloader and
ModuleStore validation own this installation. Failure after HF publication produces `PARTIAL` and
preserves the prior local installation; the committed HF revision is not rolled back.

A run-derived `workspace/run-<id>` module is already installed and runnable. Its publish dialog
therefore does not offer **sync locally**, and the backend rejects `syncLocally=true` for that source.
Publication leaves the workspace module unchanged and returns the exact target repository and commit.
Installing the published `owner/name` identity later is a separate normal Install action. Other module
sources retain the optional sync-local behavior above.

A publication whose staged module bytes differ creates **one new commit on `main`**, fenced by the
resolved parent commit; the message is `"histoforge publish <target_repo>"`. A same-byte publication
or retry may instead return the existing head because the pinned HF client suppresses empty commits.
The publisher accepts that result only after reading the returned revision, verifying the complete
expected file inventory and the exact bytes of every uploaded file. Unmodified target files remain
at the fenced parent's bytes. Thus any returned SHA contains the planned merged repository,
while changed commits keep old `/commit/<sha>` revisions valid. There is no
model-only gate (any module publishes) and no `id` rewrite (there is no `id`).

---

## 7. HF card — `README.md`

The README's YAML frontmatter is the **only discovery surface** (there is no `facets`):

```yaml
---
license: apache-2.0
tags: [histoforge, dinov3-adapter-cellpose, testis, HE]
---
# Leydig Nuclei
…
```

- HistoForge preserves the existing card and licence files. There is **no licence selector or
  default licence assignment**. Authors edit terms and notices directly on Hugging Face.
- A generated fallback card seeds `tags` with `histoforge` + each model's `arch`, but no `license`.
  Existing cards are not rewritten. **Domain tags** (organ, stain, …) are **author-written**.
  HF reads `license:` + `tags:` when supplied by the author in the
  committed README and marks the repo — no server-side `metadata_update`, one commit covers it.

The plugin's own Modules-tab search is free-text over `ref + name + description` (structured
filtering of a small *installed* set was over-engineering; tag discovery happens on HF at install
time).

### Installed module cards

Install captures the publisher's `README.md` at the exact resolved commit, including private
repositories, using the administration-only HF credential. `README.md` is an ordinary retained
file: its exact bytes participate in module content identity and source snapshots. Optional
`readme_markdown` in the installed record/catalog is derived from that file for presentation, not a
second document store or recipe field. It is limited to **256 KiB of strict UTF-8** and counts toward
existing storage/catalog limits. README changes affect module content identity, but not recipe,
model or script digests.

| README at the resolved revision | Installation behavior |
|---|---|
| Missing | Install normally; use the recipe description. |
| Present, valid text | Save the exact text; render it when the module is selected. Empty/frontmatter-only text uses the description fallback. |
| Metadata/download failure, invalid UTF-8 or too large | Fail clearly before replacing the currently installed module. |

Publication preserves the target's existing README; a saved source README fills a missing target
path as described in §6. Optional local sync reads the actual target card at the verified commit.

The QuPath Modules tab uses a searchable module list and one selected detail pane. **Documentation**
shows the saved README text, tables and links, or the description fallback. **Workflow** shows the
ordered models, processing scripts and measurements from the validated recipe and script headers;
each model's technical settings are collapsed. Read-only viewers expose the complete installed recipe
and exact script sources. The footer shows the installed HF revision when present; workspace results
are explicitly unpublished, not assigned a synthetic revision. Publication, removal, source links and
the training-report action remain available for the selected module. Catalog loading/failure does not
discard the previous selection or details.

Raw HTML is displayed as text. Scripts and embedded resources are not executed by the documentation
view; images are explicit links, not downloads. Only HTTP(S) destinations can open in the user's
browser; relative repository files and the source link are pinned to the installed commit. Opening
details performs no network request. The plugin never receives the HF token.

Workspace training results retain the source README bytes for later publication, but their QuPath
card uses current configuration and a training-report action rather than presenting inherited
source-validation claims as new results. A publisher must review the retained README on Hugging
Face for the new model. Generated fallback cards omit inherited recipe descriptions when a training
report is declared; source-model descriptions are not evidence for a newly trained result.

---

## 8. Example fixture

The [synthetic catalog fixture](../../tests/fixtures/contracts/module-catalog-v1.json)
contains a module definition and its resolved catalog entry. It demonstrates the
schema and identity checks; it does not include runnable model weights.

---

## 9. Conventions

- **Field-name casing.** Recipe field names are `camelCase` (`schemaVersion`, `micronsPerPixel`,
  `parentModel`, `paddingModelPx`). The **one exception is `params`**, whose keys are the runner's
  native `snake_case` kwargs, passed through verbatim. That boundary is intentional: *outside
  `params` = schema (camelCase); inside `params` = runner (snake_case).*
- **Enum values** follow their domain: `arch` is `kebab-case` (`dinov3-adapter-cellpose`);
  `type` enums are single words or `snake_case` (`vendored`, `field`, `instance_segmentation`).
- **Model-ids** are short `snake_case`, unique within a module, and stable (they are URL segments
  and provenance tags).
- **Vendored weight files** are named `<model-id>.safetensors` — uniform for single- and
  multi-model modules (no special `weights.safetensors` case), and stable when a model is later
  vendored into a multi-model module. `weights.file` is explicit, so the runner never depends on the
  convention — but humans do.

---

## 10. Design rules

- **One type; shape is derived.** No `moduleType`. `len(models)`/`len(steps)` is the shape.
- **`weights.type` rides the weights** and records ownership plus redistribution:
  module-owned and re-hostable bytes use `vendored`; gated or non-rehostable bytes use
  `referenced`, SHA-pinned. `local` is an incomplete working form only. Reproducibility
  comes from the module's immutable content identity for vendored bytes plus the pinned
  `revision` for referenced bytes.
- **`steps` / `parentModel` reference local model-ids** — no namespace resolution, no `qualify`.
  Cross-module composition is by **vendoring** weights, never by reference. The only HF ref in a
  module is a pinned encoder in `weights.referenced`.
- **Code is curated runners in the image** — never per-module, never `trust_remote_code` on user
  content. `arch` selects our runner.
- **Publish is pure HF by default; the Volume isn't written.** An explicit *sync-locally* option
  downloads and installs the complete just-published target at the verified HF commit, including
  preserved HF-side files rather than substituting source staging.
  Run-derived workspace modules omit that redundant option. Install is the inverse (HF → Volume).
- **Runner parameters are explicit.** Each runner defines its supported `params`
  keys; recipes must not carry unused settings.
