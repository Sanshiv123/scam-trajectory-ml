"""GRU training and causal prefix controls, independent of CLI and report writing."""

import numpy as np
from data.labels import labels
from features.emotion_features import turn_features
from models.trajectory_model import TrajectoryModel


def shuffle_prefix(f, t, seed):
    # Latest turn stays fixed. Permute only already observed past turns;
    # recompute deltas after shuffling, so features cannot reveal old order.
    rng = np.random.default_rng(seed + 7919 * t)
    idx = np.r_[rng.permutation(t), t]
    raw = f[: t + 1][idx]
    return turn_features(raw[:, :7], raw[:, 14])


def fit_gru(
    train, feats, seed, epochs, columns=None, early=False, shuffled=False, hidden_dim=64, lr=1e-2
):
    seqs, ys = [], []
    cols = list(range(15)) if columns is None else columns
    for c in train:
        f, y = feats[c.conv_id], labels(c, early)
        if shuffled:
            for t in range(len(f)):
                seqs.append(shuffle_prefix(f, t, seed)[:, cols])
                ys.append(np.array([0.0] * t + [y[t]]))
        else:
            seqs.append(f[:, cols])
            ys.append(y)
    model = TrajectoryModel(
        input_dim=len(cols), epochs=epochs, seed=seed, hidden_dim=hidden_dim, lr=lr
    ).fit(seqs, ys, last_only=shuffled)
    return model


def score_gru(model, data, feats, cols=None, shuffled=False, seed=0):
    cols = list(range(15)) if cols is None else cols
    if not shuffled:
        return dict(
            zip(
                [c.conv_id for c in data],
                model.running_scores([feats[c.conv_id][:, cols] for c in data]),
            )
        )
    result = {}
    for c in data:
        f = feats[c.conv_id]
        seqs = [shuffle_prefix(f, t, seed)[:, cols] for t in range(len(f))]
        result[c.conv_id] = np.array([s[-1] for s in model.running_scores(seqs)])
    return result
