# Code organization and experiment workflow

This is a small research repository. Its structure separates scientific components
from orchestration and reporting without moving historical models or results.

```text
configs/                  Versioned JSON presets for MVP and investigation
data/
  schema.py               Shared Conversation data structure
  labels.py               Shared weak per-turn label policy
  generate_data.py         Original v1 generator and historical schema alias
  investigation_data.py   Revised templates, real-data assembly and validation
  external/               Provided source snapshots
features/
  cache.py                Frozen encoder inference and content-addressed cache
  extraction.py           Transcript-to-feature assembly
  emotion_features.py     Emotion distributions, deltas and speaker features
models/
  baseline.py             Position/message/aggregate scorers
  trajectory_model.py     GRU definition, optimization and batched inference
  text.py                 TF-IDF and frozen-embedding text references
eval/
  evaluate.py             Calibration, FPR, AUC and early-warning metrics
  verify_outputs.py       Recompute metrics from persisted predictions
experiments/
  config.py               Typed, validated settings
  datasets.py             Assemble the configured train/calibration/test pools
  training.py             GRU fits and causal sequence-order controls
  probes.py               Offline length/style/source diagnostics
  artifacts.py            Models, metrics, predictions and provenance
  runner.py               Coordinate the experiment suite
  run_store.py            Unique directories and success/failure lifecycle
  cli.py                  Argument parsing and lightweight dry runs
reporting/
  summary.py              Tables, bootstrap comparisons, plots and error inventory
  checkpoint.py           Checkpoint PDF from completed artifacts
tests/                    Scientific integrity and workflow regression checks
outputs/                  Immutable completed results and named future runs
output/pdf/               Reviewed submission PDF and its provenance
```

`run_mvp.py`, `run_experiments.py`, `build_report.py`, and the old report-module
paths remain compatibility entry points. Reusable components do not import CLI
scripts. Existing `models.baseline` class paths remain valid for saved joblib
models, and `data.generate_data.Conversation` aliases the shared schema.

## Configuration and runs

Install the checkout editably with `pip install -e .`; dependency declarations
live in `pyproject.toml`. `requirements.txt` delegates to that declaration.
Historical dependency locks and source archives remain with their original runs.
Activate that environment before using the commands below, or invoke their
executables in `.venv/Scripts/` explicitly. Source can be formatted consistently
with the pinned development extra: `pip install -e ".[dev]"`, then `black .`.

```powershell
scam-trajectory --config configs/investigation.json --run-name timing-check --dry-run
scam-trajectory --config configs/investigation.json --run-name timing-check
scam-trajectory-verify outputs/runs/investigation-timing-check
```

Use another name for another attempt. Run directories are claimed atomically and
are never reused, including after failure. A failed run retains its evidence and
`run_state.json`; feature caches can still be reused by the next named run.
`--prepare-only` performs inference/cache setup only; choose a fresh run name for
the subsequent full experiment.

The runner validates data, extracts frozen features, trains each declared model,
saves its predictions and fitted state, generates summaries, then independently
verifies the saved metrics before writing `COMPLETE`. Reporting code consumes
artifacts; it never retrains a model. The PDF template requires the full
investigation suite, including frozen text references. Partial and MVP runs have
their own Markdown report.

## Boundaries that protect the experiment

- Dataset assembly decides partitions before any model fit. Held-out labels do
  not influence the label-policy experiment or calibration.
- Calibration uses synthetic validation twins only. Report observed FPR on each
  test distribution; fitting a correct validation threshold is not a test-FPR
  guarantee.
- Training utilities consume supplied data/features; their functions do not
  choose output paths, download data or build documents.
- Prefix controls use observed history only. Offline full-transcript probes are
  labeled separately from operational turn-by-turn scorers.
- Existing result folders and source archives stay unchanged when code moves.
  New run manifests include resolved settings, source/config files and hashes.

The refactor does not change the benchmark, optimization or published results.
Regression tests compare dataset hashes, archived training behavior and saved
model predictions to guard that claim. It does not repair dataset realism; that
requires a separate, versioned scientific iteration.
