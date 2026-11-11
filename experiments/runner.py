"""Experiment orchestration only; model code, configuration and storage live separately."""

from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from data.labels import labels
from eval.evaluate import opening_leakage_check
from features.extraction import extract_features as features
from models.baseline import (
    PositionScorer,
    MessageLevelEmotionScorer,
    position_features,
    running_aggregate,
)
from models.text import fit_text
from experiments.training import fit_gru, score_gru
from experiments.probes import shortcut_probes
from experiments.artifacts import dump, save_experiment, write_manifest
from experiments.config import RunConfig
from experiments.datasets import prepare_dataset


def run_suite(config: RunConfig, out: Path):
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(config.data_seed)
    np.random.seed(config.data_seed)
    torch.manual_seed(config.data_seed)
    bundle = prepare_dataset(config)
    synthetic, real, holdout = bundle.synthetic, bundle.real, bundle.holdout
    data = bundle.conversations
    dump(out / "dataset.json", [asdict(c) for c in data])
    dump(out / "dataset_audit.json", bundle.audit)
    feats, emotion_info = features(data)
    np.savez_compressed(
        out / "frozen_emotion_features.npz", **{str(k): v for k, v in feats.items()}
    )
    embeddings, text_info = (
        (None, {}) if config.skip_text or config.mode == "corrected-mvp" else features(data, "text")
    )
    write_manifest(out, config, data, dict(emotion=emotion_info, distilbert=text_info))
    if config.prepare_only:
        return
    train = [c for c in synthetic if c.split == "train"]
    leak = opening_leakage_check([c for c in synthetic if c.split in ("train", "val")])
    dump(out / "opening_probe.json", leak)
    byid = {c.conv_id: c for c in synthetic}
    diagnostics = {}
    for split in sorted({c.split for c in synthetic}):
        diffs = []
        for c in synthetic:
            if c.label and c.split == split:
                twin = byid[c.conv_id + 1]
                # Direct pairwise separation in frozen emotions before the ask.
                a = feats[c.conv_id][: c.harmful_turn_idx, :7]
                b = feats[twin.conv_id][: c.harmful_turn_idx, :7]
                diffs.append(float(np.abs(a - b).sum(1).mean()))
        diagnostics[split] = dict(
            n_pairs=len(diffs),
            mean_preask_emotion_L1=float(np.mean(diffs)),
            median_preask_emotion_L1=float(np.median(diffs)),
        )
    dump(out / "preask_emotion_diagnostics.json", diagnostics)
    if real:
        dump(out / "shortcut_probes.json", shortcut_probes(synthetic, real, holdout))
    for condition in ["synthetic"] if config.mode == "corrected-mvp" else ["synthetic", "mixed"]:
        tr = train + ([c for c in real if c.split == "train"] if condition == "mixed" else [])
        y = np.concatenate([labels(c) for c in tr])
        pm = PositionScorer().fit(np.concatenate([position_features(len(c.turns)) for c in tr]), y)
        ps = {c.conv_id: pm.predict_proba(position_features(len(c.turns))) for c in data}
        save_experiment(out, f"{condition}/position", 0, data, feats, ps, fitted=pm)
        # The true isolated-message baseline excludes the previous-turn delta.
        msg_cols = list(range(7)) + [14]
        mm = MessageLevelEmotionScorer().fit(
            np.concatenate([feats[c.conv_id][:, msg_cols] for c in tr]), y
        )
        ms = {c.conv_id: mm.predict_proba(feats[c.conv_id][:, msg_cols]) for c in data}
        save_experiment(out, f"{condition}/message", 0, data, feats, ms, fitted=mm)
        save_experiment(
            out,
            f"{condition}/aggregate",
            0,
            data,
            feats,
            {k: running_aggregate(v) for k, v in ms.items()},
            fitted=mm,
        )
        for seed in config.seeds:
            model = fit_gru(
                tr,
                feats,
                seed,
                config.epochs,
                hidden_dim=config.hidden_dim,
                lr=config.learning_rate,
            )
            save_experiment(
                out,
                f"{condition}/gru",
                seed,
                data,
                feats,
                score_gru(model, data, feats),
                model.net.state_dict(),
            )
        if config.mode == "investigation":
            for kind in ("message", "prefix"):
                model, scores = fit_text(tr, data, kind)
                import joblib

                folder = out / f"{condition}/tfidf-{kind}"
                folder.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, folder / "model.joblib")
                save_experiment(out, f"{condition}/tfidf-{kind}", 0, data, feats, scores)
                if embeddings is not None:
                    model, scores = fit_text(tr, data, kind, embeddings)
                    folder = out / f"{condition}/distilbert-frozen-{kind}"
                    folder.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, folder / "model.joblib")
                    save_experiment(
                        out, f"{condition}/distilbert-frozen-{kind}", 0, data, feats, scores
                    )
    if config.mode == "investigation":
        subsets = {
            "no-delta": list(range(7)) + [14],
            "no-speaker": list(range(14)),
            "emotion-only": list(range(7)),
            "delta-speaker": list(range(7, 15)),
        }
        for name, cols in dict(full=list(range(15)), **subsets).items():
            # Delta-bearing logistic variants have adjacent-turn context;
            # they are not the isolated-message baseline in the main comparison.
            model = MessageLevelEmotionScorer().fit(
                np.concatenate([feats[c.conv_id][:, cols] for c in train]),
                np.concatenate([labels(c) for c in train]),
            )
            scores = {c.conv_id: model.predict_proba(feats[c.conv_id][:, cols]) for c in data}
            save_experiment(out, f"ablation/logistic-{name}", 0, data, feats, scores, fitted=model)
        for seed in config.seeds:
            for name, cols in subsets.items():
                model = fit_gru(
                    train,
                    feats,
                    seed,
                    config.epochs,
                    columns=cols,
                    hidden_dim=config.hidden_dim,
                    lr=config.learning_rate,
                )
                save_experiment(
                    out,
                    f"ablation/{name}",
                    seed,
                    data,
                    feats,
                    score_gru(model, data, feats, cols),
                    model.net.state_dict(),
                )
            for name, options in [
                ("preask-label", dict(early=True)),
                ("shuffled-prefix", dict(shuffled=True)),
            ]:
                # Only synthetic training patterns; held-out labels never enter fit.
                model = fit_gru(
                    train,
                    feats,
                    seed,
                    config.epochs,
                    hidden_dim=config.hidden_dim,
                    lr=config.learning_rate,
                    **options,
                )
                scores = score_gru(
                    model, data, feats, shuffled=options.get("shuffled", False), seed=seed
                )
                save_experiment(
                    out, f"investigation/{name}", seed, data, feats, scores, model.net.state_dict()
                )
    from reporting.summary import summarize

    summarize(out)
    from eval.verify_outputs import verify

    verify(out)
    (out / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    print(f"COMPLETE: {out}", flush=True)
