# Developer tools

## CLI and dependency maintenance

To install the standalone CLI from this checkout:

```bash
uv tool install --force .
histoforge status
```

Regenerate the helper lock only when changing its dependencies:

```bash
uv pip compile pyproject.toml --universal --python-version 3.12.11 \
  --generate-hashes --only-binary :all: --no-annotate --no-header \
  --output-file requirements-bundled-cli.lock
```

If deployment returns `DEPLOYMENT_CUTOVER_REQUIRED`, keep the old environment
while finishing or explicitly abandoning its work. Deploy matching backend and
extension builds into a fresh Modal environment, then reconnect. Old recovery
records cannot resume there. There is no automatic in-place migration; do not
delete old volumes to bypass the check.

## Exact scientific-source snapshot

`maintenance/sync_histoforge_models_vendor.py` maintains the runtime code under
`histoforge/_vendor/histoforge_models`. Its provenance manifest records the source
revision, file hashes, license, and notices. Model weights are not part of it.

Verify the checked-in files without downloading or changing anything:

```bash
python scripts/maintenance/sync_histoforge_models_vendor.py --check
```

To update them, use an authorized checkout of the recorded upstream repository
and an explicit commit:

```bash
python scripts/maintenance/sync_histoforge_models_vendor.py \
  --source-checkout /path/to/histoforge-models --source-commit COMMIT
```

Review the source, license, and numerical changes together, then run the tests.
Do not edit vendored algorithms directly or remove their attribution.
