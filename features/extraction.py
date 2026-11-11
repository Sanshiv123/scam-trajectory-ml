"""Build per-conversation features from the shared frozen-encoder cache."""

from features.cache import encode
from features.emotion_features import turn_features


def extract_features(data, kind="emotion"):
    texts = [t for c in data for t in c.turns]
    vectors, info = encode(texts, kind)
    result, offset = {}, 0
    for c in data:
        raw = vectors[offset : offset + len(c.turns)]
        offset += len(c.turns)
        result[c.conv_id] = turn_features(raw, c.speakers) if kind == "emotion" else raw
    return result, info
