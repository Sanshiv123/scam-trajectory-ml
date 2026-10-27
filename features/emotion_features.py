# Per-turn emotion features, using the pretrained classifier named in the
# proposal (j-hartmann/emotion-english-distilroberta-base, 7 classes). Each
# turn's feature vector is: the emotion distribution (7), its change from
# the previous turn (7), and a speaker flag (1) -- 15 dims total, matching
# "Features and scorers" in the proposal.
import numpy as np

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

_clf = None


def _get_classifier():
    global _clf
    if _clf is None:
        from transformers import pipeline
        _clf = pipeline("text-classification",
                         model="j-hartmann/emotion-english-distilroberta-base",
                         top_k=None)
    return _clf


def emotion_distribution(texts: list) -> np.ndarray:
    """Batch-scores a list of turns, returns an (n, 7) array in EMOTION_LABELS order."""
    clf = _get_classifier()
    raw = clf(texts, batch_size=16, truncation=True)
    out = np.zeros((len(texts), len(EMOTION_LABELS)))
    label_idx = {l: i for i, l in enumerate(EMOTION_LABELS)}
    for i, scores in enumerate(raw):
        for item in scores:
            out[i, label_idx[item["label"]]] = item["score"]
    return out


def turn_features(emotion_dist: np.ndarray, speakers: list) -> np.ndarray:
    """emotion_dist: (n_turns, 7) from emotion_distribution(). Returns (n_turns, 15):
    [emotion dist (7), delta from previous turn (7), speaker flag (1)]."""
    delta = np.diff(emotion_dist, axis=0, prepend=emotion_dist[:1])
    speaker = np.array(speakers, dtype=float).reshape(-1, 1)
    return np.concatenate([emotion_dist, delta, speaker], axis=1)
