"""Generate narrative and plots only from persisted run artifacts."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_bootstrap(data, base_scores, other_scores, base_threshold, other_threshold, n=400):
    groups = {}
    for i, c in enumerate(data):
        groups.setdefault(c["pair_id"], []).append(i)
    blocks = list(groups.values())
    rng = np.random.default_rng(2026)
    y = np.array([c["label"] for c in data])

    def summarize(scores, threshold):
        maxima = np.array([scores[str(c["conv_id"])].max() for c in data])
        early = []
        for c in data:
            crosses = np.flatnonzero(scores[str(c["conv_id"])] >= threshold)
            early.append(bool(c["label"] and len(crosses) and crosses[0] < c["harmful_turn_idx"]))
        return maxima, np.array(early, dtype=float)

    a, ae = summarize(base_scores, base_threshold)
    b, be = summarize(other_scores, other_threshold)
    aucs, recalls = [], []
    for _ in range(n):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
        positives = idx[y[idx] == 1]
        recalls.append(float(np.mean(be[positives] - ae[positives])))
    return dict(
        direction="candidate minus full GRU",
        n_bootstrap=n,
        unit="matched scenario pair",
        auc_difference=float(roc_auc_score(y, b) - roc_auc_score(y, a)),
        auc_difference_ci95=np.quantile(aucs, [0.025, 0.975]).tolist(),
        early_recall_difference=float(np.mean(be[y == 1] - ae[y == 1])),
        early_recall_difference_ci95=np.quantile(recalls, [0.025, 0.975]).tolist(),
        limitation="Conditional on seed 0 fits and thresholds; does not include calibration or training uncertainty. Exploratory, no multiplicity adjustment.",
    )


def summarize(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(out / "dataset.json")
    manifest = load(out / "manifest.json")
    metrics = {}
    for p in out.glob("**/seed-*/metrics.json"):
        name = str(p.parent.parent.relative_to(out)).replace("\\", "/")
        metrics.setdefault(name, {})[p.parent.name] = load(p)
    splits = [s for s in sorted({c["split"] for c in data}) if s.startswith("test")]
    header = [
        "experiment",
        "seed",
        "split",
        "target_fpr",
        "auc",
        "early_recall",
        "median_lead",
        "actual_fpr",
        "calibration_fpr",
    ]
    rows = []
    for name, seeds in metrics.items():
        for seed, targets in seeds.items():
            for alpha, parts in targets.items():
                for split, r in parts.items():
                    rows.append(
                        dict(
                            experiment=name,
                            seed=seed,
                            split=split,
                            target_fpr=alpha,
                            auc=r["classification"].get("auc"),
                            early_recall=(
                                r["earliness"]["early_recall"] if split.startswith("test") else None
                            ),
                            median_lead=r["earliness"]["median_lead_turns"],
                            actual_fpr=r["false_positives"]["fpr"],
                            calibration_fpr=r["calibration"]["fpr"],
                        )
                    )
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    lines = [
        (
            "# Rebuilt investigation and iteration checkpoint"
            if manifest["config"]["mode"] == "investigation"
            else "# Corrected original MVP rerun"
        ),
        "",
        "These are newly computed results. Historical checkpoint numbers were not reused. The original October output files in the parent directory are archival.",
        "",
        "## Setup",
        "",
        f"Dataset: {len(data)} conversations. Train seeds: {manifest['config']['seeds']}; {manifest['config']['epochs']} epochs, fixed in advance. Emotion model revision: `{manifest['models']['emotion'].get('revision')}`.",
        "",
        "Each scorer is calibrated on synthetic legitimate validation twins only, separately at target 1%, 5%, and 10% conversation FPR. Calibration is conservative for score ties. Test FPR is measured independently and can exceed the calibration target. Accuracy/F1 and alert metrics depend on the threshold; AUC does not.",
        "",
        "Primary tests use matched synthetic scam/legitimate pairs. Different generator names denote text-substitution styles, not real LLMs. Real transcripts are presumed legitimate based on the provided files; their original collection and threading were not independently reproduced.",
        "",
        "## Main results at target 5% FPR",
        "",
        "Values are arithmetic means across available model seeds; deterministic linear models run once. Seed-wise results, actual FPRs, and sample sizes are in summary.csv and metrics.json. Early recall is the share of scams flagged strictly before the ask; median lead is conditional on successful early alerts.",
        "",
    ]
    for split in splits:
        lines.extend(
            [
                f"### {split}",
                "",
                "| Experiment | AUC mean [seed min, max] | Early recall mean | Actual test FPR mean |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, seeds in sorted(metrics.items()):
            rs = [v["0.05"][split] for v in seeds.values() if split in v["0.05"]]
            if not rs:
                continue
            auc = np.array([r["classification"].get("auc", np.nan) for r in rs])
            early = np.mean([r["earliness"]["early_recall"] for r in rs])
            fpr = np.mean([r["false_positives"]["fpr"] for r in rs])
            lines.append(
                f"| {name} | {auc.mean():.3f} [{auc.min():.3f}, {auc.max():.3f}] | {early:.1%} | {fpr:.1%} |"
            )
        lines.append("")
    if "test_slow" in splits:
        lines.extend(
            [
                "## Long / delayed-ask stress test",
                "",
                "test_slow reuses only standard-test scenarios, with 20-32 turns and asks at 70-95% of the transcript. Its legitimate twins have the same length, speaker pattern and request position. This adds neutral setup before the pressure turns; it tests length/delayed timing and is not a validated simulation of subtle or adaptive evasion.",
                "",
            ]
        )
    if any(c["split"] == "secondary_enron" for c in data):
        lines.extend(
            [
                "## Independent real Enron false-positive check",
                "",
                "These held-out examples were excluded from training and calibration. Exact full-transcript duplicates were removed before splitting; near-duplicates and related threads may remain. Rates below are observed on the available sample and do not establish deployment safety.",
                "",
                "| Experiment | Target | Enron FPR mean |",
                "|---|---:|---:|",
            ]
        )
        for name, seeds in sorted(metrics.items()):
            if not name.startswith(("synthetic/", "mixed/")):
                continue
            for target in ("0.05", "0.1"):
                vals = [
                    v[target]["secondary_enron"]["false_positives"]["fpr"] for v in seeds.values()
                ]
                lines.append(f"| {name} | {float(target):.0%} | {np.mean(vals):.1%} |")
        lines.extend(
            [
                "",
                "Wilson intervals and sample counts for each fitted model are saved in metrics.json.",
                "",
            ]
        )
    if (out / "shortcut_probes.json").exists():

        def compact(v):
            if isinstance(v, dict):
                return {k: compact(x) for k, x in v.items() if k != "predictions"}
            return v

        lines.extend(
            [
                "## Shortcut diagnostics",
                "",
                "```json",
                json.dumps(compact(load(out / "shortcut_probes.json")), indent=2),
                "```",
                "",
                "Source style/domain separability is a warning about confounding, not proof that the GRU uses that shortcut. Length features use full transcripts only as an offline diagnostic; the operational position scorer uses observed absolute turn index only.",
                "",
            ]
        )
    bootstrap = {}
    base = out / "synthetic/gru/seed-0"
    if base.exists() and "test_slow" in splits:
        a = np.load(base / "predictions.npz")
        am = load(base / "metrics.json")
        for name in (
            "ablation/no-delta",
            "ablation/no-speaker",
            "investigation/shuffled-prefix",
            "investigation/preask-label",
        ):
            p = out / name / "seed-0"
            if not p.exists():
                continue
            b = np.load(p / "predictions.npz")
            bm = load(p / "metrics.json")
            bootstrap[name] = {}
            for split in ("test", "test_heldout_pattern", "test_slow"):
                cs = [c for c in data if c["split"] == split]
                bootstrap[name][split] = paired_bootstrap(
                    cs,
                    a,
                    b,
                    am["0.05"][split]["earliness"]["threshold"],
                    bm["0.05"][split]["earliness"]["threshold"],
                )
        (out / "paired_bootstrap.json").write_text(
            json.dumps(bootstrap, indent=2), encoding="utf-8"
        )
        lines.extend(
            [
                "## Paired uncertainty checks",
                "",
                "paired_bootstrap.json contains candidate-minus-GRU differences in AUC and early recall with 95% intervals, resampling matched scenario pairs. These intervals are conditional on seed-0 fits and fixed validation thresholds. The multi-seed tables show additional training variability; neither analysis establishes a population false-positive guarantee.",
                "",
            ]
        )
    # Error inventory, including all false positives and late/missed scams.
    errors = []
    if base.exists():
        scores = np.load(base / "predictions.npz")
        ms = load(base / "metrics.json")
        for c in data:
            if c["split"] not in ms["0.05"]:
                continue
            th = ms["0.05"][c["split"]]["earliness"]["threshold"]
            values = scores[str(c["conv_id"])]
            cross = np.flatnonzero(values >= th)
            first = int(cross[0]) if len(cross) else None
            err = (
                "false_positive"
                if c["label"] == 0 and first is not None
                else (
                    "late_or_missed"
                    if c["label"] and (first is None or first >= c["harmful_turn_idx"])
                    else None
                )
            )
            if err:
                errors.append(
                    dict(
                        id=c["conv_id"],
                        split=c["split"],
                        kind=err,
                        first_alert=first,
                        ask=c["harmful_turn_idx"],
                        max_score=float(values.max()),
                    )
                )
    (out / "error_inventory.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    # Human audit is a worksheet, not completed annotation or agreement evidence.
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(data))[:150]
    with (out / "human_audit_PENDING.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "split",
                "label",
                "ask",
                "turns_json",
                "rater1_realism",
                "rater2_realism",
                "rater3_realism",
                "verified_ask",
                "notes",
            ]
        )
        for i in indices:
            c = data[int(i)]
            w.writerow(
                [
                    c["conv_id"],
                    c["split"],
                    c["label"],
                    c["harmful_turn_idx"],
                    json.dumps(c["turns"]),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    if manifest["config"]["mode"] == "investigation":
        lines.extend(
            [
                "## Failure analysis and remaining work",
                "",
                "The paired ablations investigate feature contribution, while the shuffled-prefix control keeps the current turn fixed and permutes only observed history, recomputing deltas. The pre-ask-label model changes soft training labels only; held-out patterns remain excluded from fit. High end-of-conversation AUC can coexist with zero early recall, so one does not establish or refute the other.",
                "",
                "Review error_inventory.json against dataset.json; an inventory is not a completed human error analysis. human_audit_PENDING.csv is an unfilled 150-example worksheet. Human realism ratings, verified ask annotations and inter-rater agreement remain outstanding.",
                "",
                "The text references are TF-IDF and frozen pretrained DistilBERT embeddings with logistic classifiers. The prefix DistilBERT reference averages embeddings of observed turns; it does not jointly encode/fine-tune a full transcript. Finite template vocabulary can give high scores; real-LLM generation, Enron restyling, and a fine-tuned DistilBERT ceiling remain uncompleted. Changing generation method alone would not prove that all shortcuts are gone.",
                "",
                "The original Enron preparation script is unavailable. Source files are hashed for provenance, but their presumed legitimate labels are not verified. Exact deduplication does not exclude all semantic or related-thread leakage. The primary matched synthetic test and separate real FPR check must both be reported.",
                "",
                "All tables above are generated from saved metrics. No historical 40%/56% Enron FPR, 0.78/0.48 slow-burn result, or 1.000 DistilBERT AUC is claimed unless reproduced in these artifacts.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Scope of this corrected rerun",
                "",
                "This run deliberately preserves the original v1 dataset and its unrealistic pre-ask monologues. It corrects evaluation and baseline context, and repeats the GRU with three seeds. It does not run the revised-data ablations, text references, real-data mixing or slow/delayed-ask tests; those are in the separate investigation run.",
                "",
                "The grouped opening probe uses train/validation pairs only. Classification AUC and early recall answer different questions, and a high AUC can coexist with zero early recall. Observed test FPR can exceed the target even with correct empirical calibration; inspect the saved calibration and test intervals.",
                "",
                "The error inventory and unfilled audit worksheet support follow-up; they are not completed human validation.",
                "",
            ]
        )
    # Plot lead times with never-alert counts explicit, avoiding a success-only story.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    test = [c for c in data if c["split"] == "test" and c["label"]]
    for name in ("synthetic/position", "synthetic/message", "synthetic/aggregate", "synthetic/gru"):
        p = out / name / "seed-0"
        s = np.load(p / "predictions.npz")
        m = load(p / "metrics.json")["0.05"]["test"]
        leads = m["earliness"]["per_conversation_leads"]
        vals = [v for v in leads if v is not None]
        if vals:
            axes[0].hist(
                vals,
                bins=np.arange(-35, 20),
                alpha=0.45,
                label=f"{name.split('/')[-1]} (never={sum(v is None for v in leads)})",
            )
        else:
            axes[0].plot([], [], label=f"{name.split('/')[-1]} (never={len(leads)})")
    axes[0].axvline(0, color="black")
    axes[0].set(
        xlabel="Turns before ask (positive = early)",
        ylabel="Scam count",
        title="Standard test, seed 0, target FPR 5%",
    )
    axes[0].legend(fontsize=7)
    for label in (0, 1):
        c = next(c for c in data if c["split"] == "test" and c["label"] == label)
        if label == 0:
            scam = next(c for c in data if c["split"] == "test" and c["label"] == 1)
            c = next(c for c in data if c["pair_id"] == scam["pair_id"] and c["label"] == 0)
        values = np.load(base / "predictions.npz")[str(c["conv_id"])]
        axes[1].plot(values, marker=".", label="scam" if label else "matched twin")
        if label:
            axes[1].axvline(c["harmful_turn_idx"], color="gray", linestyle="--", label="ask")
    th = load(base / "metrics.json")["0.05"]["test"]["earliness"]["threshold"]
    axes[1].axhline(th, color="black", linestyle=":", label="threshold")
    axes[1].set(xlabel="Observed turn index", ylabel="GRU score", title="One matched test pair")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "lead_and_trace.png", dpi=140)
    plt.close(fig)
    lines.extend(["![Lead-time and matched-pair trace](lead_and_trace.png)", ""])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
