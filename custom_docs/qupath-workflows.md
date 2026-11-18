# Using HistoForge in QuPath

This guide covers slide import, annotation-scoped inference, and training ROIs.

## Importing slides

Import through QuPath normally; leave its reader selector at **Default (let QuPath
decide)** unless a specific reader choice is part of your study. If QuPath offers
multiple scenes, open the intended tissue image. HistoForge captures that actual
reader, scene, calibration, pyramid, and source identity; it does not silently
switch to scene zero or another reader in the backend.

The qualified RGB examples include SVS (including selected JPEG2000 cases), NDPI,
SCN, CZI, TIFF/OME-TIFF variants, BIF, Trestle, MRXS, and VSI. Keep companion files
and directories together for formats such as MRXS and VSI. This is bounded evidence
for particular files, readers, and codecs, not universal support for every file
with those extensions. Non-RGB scanner masks, uncalibrated associated images, and
the retained stock-host AVS/JPEG-XR CZI failures are not qualified cases.

HistoForge chooses unchanged original upload or prepared RGB pixels through its
normal source-selection policy. It may upload a complete companion-file package;
do not rename, move, or edit those source files while an action is active. Original
reader support for inference does not automatically expand training-source support.
Small RGB differences can occur across readers and platforms. Check representative
slides before using a new format, scanner, or reader in a study.

## Inference inside existing annotations

1. In HistoForge **Project**, select the project images and assign their module.
2. Set **Scope → Annotations by class**, then select the exact annotation class,
   for example `Tissue mask`. All matching area annotations in those images become
   separate targets; merely selecting one object in QuPath is not this scope selector.
3. Choose **All models** or specific models. **Fill** skips model work where output
   is present; it does not check whether that output is complete or current.
   **Redo** reruns selected model work and replaces its output after success.

Fill and Redo lock the selected boundaries and save them through QuPath before inference.
Predictions are clipped to each outline. To change a boundary, use QuPath Unlock,
edit it deliberately, then Redo the affected results. **Metrics** recomputes the
selected measurements without model inference; Metrics and Export do not change
boundary locks. **Whole slide** and **HistoForge ROIs** remain separate scope choices.
An inference tissue-mask annotation is not automatically a training ROI.

## Creating training ROIs

Use a separate ordinary QuPath project for curated training data when practical.
There is no special training-project format.

1. Open an image and choose the **HistoForge ROI** tool in the QuPath toolbar
   (outlined rectangle marked ROI). Drag a rectangle over a training example.
   Clicking places the default size from **Project → Draw ROI — W/H**.
2. Draw or review the biological objects inside the rectangle and assign their
   QuPath classes. The ROI tool supplies the administrative marker automatically;
   an ordinary rectangle, a renamed annotation, or a `Tissue mask` class alone does not.
3. In **Train**, choose the module and trainable model, check target classes and
   annotation mappings, then **Refresh** and **Generate split**.
4. Review the split and settings. Only **Start Training** submits training; opening
   Train or refreshing reads local project data and does not launch a GPU job.

Every marked HistoForge ROI across every project image is included, not just the
currently selected rectangle. Rectangles must be inside the image, on z=0/t=0,
and must not overlap by positive area. Ordinary annotation and non-cell detection
objects can supply labels; true QuPath cell objects do not. Model-generated labels
are not automatically ground truth: review them for the intended training task.
Pixels inside a training ROI without a mapped target label contribute as background.

If no marked ROIs are found, **Refresh** shows **No training ROIs found** and
leaves split generation, export, and training disabled. Marked ROIs still undergo
full validation, which can take time on large projects.

Each training ROI must fit the backend's advertised object limit, currently
10,000 mapped target objects. Existing mapped detections count toward this limit.
If validation reports too many objects, resize the ROI or replace it with smaller,
non-overlapping HistoForge ROIs. Review the labels, then click **Refresh**.
Inference limits do not change this separate training limit.

Older installations may show a less detailed validation error.
