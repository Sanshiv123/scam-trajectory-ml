# Feature ablations: the 15-dim per-turn feature vector is [emotion
# distribution (7 dims, idx 0-6), its delta from the previous turn (7 dims,
# idx 7-13), speaker flag (1 dim, idx 14)] -- see features/emotion_features.py.
# This re-trains the message-level and trajectory scorers on ablated subsets
# of that vector (reusing the exact same extracted features and data split,
# computed once) to see which pieces the two content-based scorers actually
# rely on, rather than assuming all 15 dims pull their weight equally.
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_mvp import extract_all_features, stack_train, build_full_dataset
from models.baseline import turn_labels, MessageLevelEmotionScorer
from models.trajectory_model import TrajectoryModel
from eval.evaluate import full_report, SPLITS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

ABLATIONS = {
    "full (emotion_dist + delta + speaker, 15-dim)": list(range(15)),
    "no_delta (emotion_dist + speaker, 8-dim)": list(range(7)) + [14],
    "no_speaker (emotion_dist + delta, 14-dim)": list(range(14)),
    "emotion_dist_only (7-dim)": list(range(7)),
    "delta_only + speaker (8-dim)": list(range(7, 15)),
}


def _slice_feats(turn_feats_by_id, cols):
    return {cid: feats[:, cols] for cid, feats in turn_feats_by_id.items()}


def main():
    data, by_split, _, _ = build_full_dataset()

    print("extracting full (unablated) features once, reused for every ablation...")
    turn_feats_by_id, _ = extract_all_features(data)

    results = {}
    for name, cols in ABLATIONS.items():
        print(f"--- ablation: {name} ---")
        feats = _slice_feats(turn_feats_by_id, cols)
        dim = len(cols)

        Xe, ye = stack_train(by_split["train"], feats)
        msg_model = MessageLevelEmotionScorer().fit(Xe, ye)

        train_feats = [feats[c.conv_id] for c in by_split["train"]]
        train_labels = []
        for c in by_split["train"]:
            train_labels.append(np.array([int(c.label == 1 and i >= c.harmful_turn_idx)
                                           for i in range(len(c.turns))], dtype=float))
        traj_model = TrajectoryModel(input_dim=dim).fit(train_feats, train_labels)

        scorers = {
            "message-level (emotion)": lambda c, f, m=msg_model: m.predict_proba(f),
            "trajectory (emotion, GRU)": lambda c, f, m=traj_model: m.running_score(f),
        }
        ablation_result = {}
        for scorer_name, fn in scorers.items():
            report, _ = full_report(by_split, feats, running_score_fn=fn, target_fpr=0.05)
            ablation_result[scorer_name] = {
                split_name: report[split_name]["classification"].get("auc", float("nan"))
                for split_name in SPLITS
            }
        results[name] = ablation_result

    path = os.path.join(OUT_DIR, "feature_ablation_report.txt")
    with open(path, "w") as f:
        f.write("Feature ablations -- classification AUC per split, target FPR 5%,\n"
                "message-level and trajectory GRU re-trained from scratch on each\n"
                "ablated feature subset (position-only and running aggregate are\n"
                "unaffected: position-only doesn't use these features at all, and\n"
                "running aggregate is a transform of message-level's own output, so\n"
                "its numbers would just mirror message-level's row above).\n\n")
        for ablation_name, scorer_results in results.items():
            f.write(f"=== {ablation_name} ===\n")
            for scorer_name, split_aucs in scorer_results.items():
                vals = "  ".join(f"{s}={split_aucs[s]:.3f}" for s in SPLITS)
                f.write(f"  {scorer_name}: {vals}\n")
            f.write("\n")
    print(open(path).read())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
