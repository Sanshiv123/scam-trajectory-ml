# Emotional trajectory detection of social engineering

Team 6: Sanchusri Kavitha Babu, Rahul Ajith Kumar, Sisvanthkumar Sathivadivel.

This branch rebuilds the investigation checkpoint from the original October MVP
(commit 67cddb6 on main). Every new result comes from the code and artifacts in
this branch. The previous investigation branch is not merged.

## Setup and commands

Use Python 3.12 and an isolated environment. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -u run_mvp.py
.\.venv\Scripts\python.exe -u run_experiments.py --mode investigation
```

The first feature extraction downloads pretrained model weights. No paid API
or generated LLM conversations are required. Model/text features are cached in
`.cache/`; this directory and `.venv/` are ignored by Git. For exact package
versions use a completed run's `requirements-lock.txt` with the CPU index above.
Model revisions, truncation settings and source hashes are saved in manifest.json.

Default runs use seeds 0, 1 and 2 and 30 epochs. `--seeds`, `--epochs`, and
`--threads` configure runs, and their values are saved. `--prepare-only` downloads
models and caches features without training. `--skip-text` omits frozen DistilBERT
references (it still runs TF-IDF). Completed run folders contain `COMPLETE` and
cannot be overwritten; archive/rename a completed folder before a deliberate rerun.

## Evidence and outputs

- `outputs/corrected-mvp/`: corrected evaluation of the original v1 templates.
- `outputs/investigation/`: new v2 benchmark and all investigation experiments.
- Each run includes `REPORT.md`, `summary.csv`, `manifest.json`, a dependency lock,
  exact `dataset.json`, dataset audit, and error inventory.
- Each experiment/seed saves `metrics.json`, per-turn `predictions.npz`,
  per-conversation CSV, and fitted model weights where applicable.
- `paired_bootstrap.json` contains paired differences conditional on seed-0 fits;
  the report also shows variability across three training seeds.
- `human_audit_PENDING.csv` is an unfilled worksheet, not completed annotations.
- The root-level October PDF and `outputs/report.txt` / original plots are
  **archival results with the old evaluation**, retained for comparison only.

Generate the checkpoint PDF from a completed investigation run with the bundled
artifact Python (which includes ReportLab), or install ReportLab in a separate
reporting environment:

```powershell
python build_report.py --run outputs/investigation
```

Output: `output/pdf/Investigation_Iteration_Checkpoint_Rebuilt.pdf`.
The full machine-generated tables remain in `outputs/investigation/REPORT.md`.

## Corrections to the original MVP

1. Conservative empirical threshold calibration respects `score >= threshold`,
   including tied scores and small validation sets. Report actual FPR and Wilson
   intervals on validation and each test, rather than assuming target FPR is met.
2. Position features use absolute observed turn index; they do not reveal the
   unknown final conversation length.
3. The isolated-message baseline uses emotion distribution and speaker only.
   Models using delta features explicitly have adjacent-turn context.
4. The opening probe fits TF-IDF inside grouped folds, keeping scam/twin pairs
   together. It is an opening-specific check, not proof of general dataset validity.
5. Reports are generated from metrics; no hard-coded historical findings.

The corrected MVP preserves the v1 templates to isolate evaluation changes. Its
unrealistic pre-ask monologues are retained only as a historical comparison.

## Rebuilt investigation

The v2 generator alternates speakers before and after the ask. Scam/twin pairs
share their opening, recipient replies, length, speaker schedule, amount and
deadline. Scenario identities are unique across train/validation/test. Four named
styles are text substitutions, not GPT/Claude/Llama/Qwen model generations.

Experiments include:

- Position, isolated-message, aggregate, and GRU on matched synthetic twins.
- Real-negative-mixed training: 410 real training conversations and 73 diagnostic
  validation examples; 200 separate Enron examples measure false positives.
  Calibration uses synthetic legitimate validation twins in both conditions.
- Five GRU feature configurations, plus corresponding logistic feature variants.
- Causal shuffled-prefix control: shuffle only past turns, retain the current
  turn and recompute deltas; train on prefixes with last-turn supervision.
- Soft pre-ask training labels on urgency/authority turns from training patterns
  only; test ask labels never change.
- Paired long/delayed-ask tests: 20-32 turns, ask at 70-95% of the transcript.
- Whole-transcript length/style shortcut probes, a legitimate-source probe,
  TF-IDF message/prefix baselines, and frozen DistilBERT message/prefix references.
- Pre-ask emotion separation diagnostics, error inventory and pending human audit.

The long test adds neutral setup before pressure; it is not validated subtle or
adaptive evasion. Frozen DistilBERT + logistic regression is not a fine-tuned
DistilBERT ceiling. These distinctions are retained in the checkpoint report.

## Scientific limits

This is a controlled template benchmark. There is no independent real-scam test
set, completed human realism audit, actual multi-LLM generation, or restyled
Enron experiment. Perfect transcript classification can coexist with zero early
recall and does not by itself establish a shortcut. Small samples at a 1% target
are especially uncertain. Paired bootstrap intervals do not include training or
calibration uncertainty; seed ranges address only observed fit variability.

External data are provided JSON snapshots. The Enron reconstruction script is
not available, and presumed legitimate labels/threading are not independently
verified. Inputs are hashed and deduplicated exactly before splitting, but related
threads or semantic near-duplicates may remain. Real conversations are capped
at the first 64 turns; truncation counts and transformer token limits are recorded.

Historic claims from the other checkpoint are not treated as targets. Any claim
in the rebuilt submission must be supported by the new artifacts.
