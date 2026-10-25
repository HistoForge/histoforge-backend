# Testing

## Environment

Create the development environment once, then install the package:

```bash
conda env create -f environment.yml
conda activate histoforge-backend
python -m pip install -e ".[dev]"
```

### Linux reader runtime

On Linux x86-64, runtime-identity tests require the pinned Java/Bio-Formats files.
Install them once into a new directory using the production hash checks:

```bash
export HISTOFORGE_JAVA_HOME="$PWD/.reader-runtime/java"
export HISTOFORGE_BIOFORMATS_JAR="$PWD/.reader-runtime/bioformats_package.jar"
python - <<'PY'
from pathlib import Path
from histoforge.cloud.runtime.install_readers import install

install(
    Path("histoforge/cloud/runtime/reader-runtime.json"),
    Path(".reader-runtime"),
)
PY
```

Reuse the environment variables on later runs; the installer refuses an existing
destination. This Linux archive is not a macOS or Windows runtime. Normal
QuPath helper setup does not need this step.

## Run the checks

From the repository root:

```bash
ruff check .
ruff format --check .
pytest
```

## Coverage

The suite checks API and storage contracts, source handling, training lifecycle,
local setup, and numerical routines using synthetic inputs and mocked providers.
CI also tests local credential persistence on Linux, macOS, and Windows.

Concurrent-storage tests verify that a Volume reload waits until active file
reads finish and that read failures release the lock. These local checks do not
replace tests against an actual deployment.

Model-specific regression tests, research datasets, weights, and cloud benchmarks
are maintained separately. This suite does not measure model accuracy, GPU cost,
maximum supported workload, or the complete desktop installation experience.
Do not substitute a passing unit test for those checks.

Keep test data synthetic or include its redistribution permission and provenance.
Never commit tokens, project results, patient data, or model weights.
