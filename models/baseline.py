# Two of the proposal's six scorers, both from the same per-turn emotion
# features (features/emotion_features.py):
#   - position-only: PreScam's strongest timing baseline, trained on just
#     the turn index -- catches ask-position leakage.
#   - message-level (emotion): a logistic regression on a single turn's
#     emotion features, no context from earlier turns.
# Running-aggregate (the TA-suggested baseline) is a transform of
# message-level's output, not its own model -- the proposal specifies this
# scorer tracks "mean and max over turns so far". Max-so-far is always >=
# mean-so-far, so taking their max would just collapse to max-so-far; we
# average them instead, so both the overall tenor (mean) and any single
# alarming message (max) contribute.
import numpy as np
from sklearn.linear_model import LogisticRegression


def turn_labels(conversations):
    # weak per-turn label: 1 at/after t_ask in a manipulative conversation, else 0
    labels = []
    for c in conversations:
        for i in range(len(c.turns)):
            labels.append(int(c.label == 1 and i >= c.harmful_turn_idx))
    return labels


class PositionScorer:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, position_feats, labels):
        self.clf.fit(position_feats, labels)
        return self

    def predict_proba(self, position_feats):
        return self.clf.predict_proba(position_feats)[:, 1]


def position_features(n_turns: int) -> np.ndarray:
    idx = np.arange(n_turns)
    return np.stack([idx, idx / max(n_turns - 1, 1)], axis=1)


class MessageLevelEmotionScorer:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, turn_feats, labels):
        self.clf.fit(turn_feats, labels)
        return self

    def predict_proba(self, turn_feats):
        return self.clf.predict_proba(turn_feats)[:, 1]


def running_aggregate(turn_scores: np.ndarray) -> np.ndarray:
    # TA-suggested baseline: running mean and running max of message-level
    # scores so far, averaged together (see module docstring)
    running_mean = np.cumsum(turn_scores) / (np.arange(len(turn_scores)) + 1)
    running_max = np.maximum.accumulate(turn_scores)
    return (running_mean + running_max) / 2
