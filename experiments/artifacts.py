"""Persist metrics, predictions, model state and exact source/data provenance."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from eval.evaluate import full_report

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def save_experiment(out, name, seed, data, feats, scores, weights=None, fitted=None):
    folder = out / name / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    by = {s: [c for c in data if c.split == s] for s in sorted({c.split for c in data})}
    # val_real is reported as a diagnostic only, never mixed into calibration.
    reports = {}
    for alpha in (0.01, 0.05, 0.10):
        report, _ = full_report(by, feats, lambda c, f: scores[c.conv_id], alpha)
        reports[str(alpha)] = report
    dump(folder / "metrics.json", reports)
    np.savez_compressed(folder / "predictions.npz", **{str(k): v for k, v in scores.items()})
    if weights is not None:
        torch.save(weights, folder / "model.pt")
    if fitted is not None:
        import joblib

        joblib.dump(fitted, folder / "model.joblib")
    # No raw external email text copied into the human-readable report.
    with (folder / "conversations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "pair",
                "scenario",
                "split",
                "label",
                "ask",
                "max_score",
                "alert",
                "lead",
            ],
        )
        writer.writeheader()
        th = next(iter(reports["0.05"].values()))["earliness"]["threshold"]
        for c in data:
            crossed = np.flatnonzero(scores[c.conv_id] >= th)
            alert = int(crossed[0]) if len(crossed) else None
            writer.writerow(
                dict(
                    id=c.conv_id,
                    pair=c.pair_id or f"legacy-{c.conv_id//2}",
                    scenario=c.scenario_id,
                    split=c.split,
                    label=c.label,
                    ask=c.harmful_turn_idx,
                    max_score=float(scores[c.conv_id].max()),
                    alert=alert,
                    lead=c.harmful_turn_idx - alert if c.label and alert is not None else None,
                )
            )
    print(f"saved {name} seed {seed}", flush=True)
    return reports


def write_manifest(out, config, data, models):
    files = []
    excluded = {".venv", ".git", ".cache", "outputs", "output", "tmp", "build", "dist"}
    for folder, directories, names in os.walk(ROOT):
        directories[:] = [
            name for name in directories if name not in excluded and not name.endswith(".egg-info")
        ]
        files.extend(Path(folder) / name for name in names if name.endswith(".py"))
    files.sort()
    files += sorted((ROOT / "configs").glob("*.json"))
    files += [
        p
        for p in [ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "README.md"]
        if p.exists()
    ]
    import zipfile

    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, str(p.relative_to(ROOT)))
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        config=config.as_dict(),
        base="67cddb6",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        python=sys.version,
        source_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        external_hashes={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "data/external").glob("*.json")
        },
        dataset_hash=hashlib.sha256(
            json.dumps([asdict(c) for c in data], sort_keys=True).encode()
        ).hexdigest(),
        models=models,
        calibration="Synthetic legitimate validation only; conservative observed FPR, no population guarantee",
        limitations=[
            "Controlled templates, no real scam test set",
            "No completed human audit or LLM restyling",
            "Frozen DistilBERT references are not fine-tuned models",
        ],
    )
    dump(out / "manifest.json", manifest)
    lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    # Installing this checkout editably must not pin a different machine's
    # absolute path or an earlier Git revision in future run locks.
    lock = (
        "\n".join(
            line
            for line in lock.splitlines()
            if not (
                line.startswith("-e ")
                and any(
                    token in line.lower()
                    for token in ("scam_trajectory_ml", "scam-trajectory-ml", str(ROOT).lower())
                )
            )
        )
        + "\n"
    )
    lock = "# Install the matching source checkout first with: pip install -e . --no-deps\n" + lock
    (out / "requirements-lock.txt").write_text(lock, encoding="utf-8")
