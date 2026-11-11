"""Offline shortcut diagnostics, separate from deployable turn-by-turn scorers."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def style_vector(c, length_only=False):
    import re

    text = " ".join(c.turns)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    if length_only:
        return [len(c.turns), len(words), len(text)]
    n = max(1, len(words))
    chars = max(1, len(text))
    function = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "i",
        "you",
        "we",
        "it",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
    }
    return [
        np.mean([len(w) for w in words]) if words else 0,
        len(words) / sentences,
        sum(w in function for w in words) / n,
        text.count("!") / chars,
        text.count("?") / chars,
        text.count(",") / chars,
        sum(ch.isupper() for ch in text) / chars,
        sum(ch.isdigit() for ch in text) / chars,
    ]


def shortcut_probes(synthetic, real, holdout):
    train = [c for c in synthetic if c.split == "train"]
    test = [c for c in synthetic if c.split == "test"]
    out = {}
    for length_only in (True, False):
        name = "length" if length_only else "style"
        for mix in (False, True):
            tr = train + ([c for c in real if c.split == "train"] if mix else [])
            clf = make_pipeline(
                StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
            )
            clf.fit([style_vector(c, length_only) for c in tr], [c.label for c in tr])
            evaluations = {
                "synthetic_test": test,
                "synthetic_scams_vs_enron": [c for c in test if c.label] + holdout,
            }
            out[f"{name}_{'mixed' if mix else 'synthetic'}"] = {}
            for s, cs in evaluations.items():
                probs = clf.predict_proba([style_vector(c, length_only) for c in cs])[:, 1]
                out[f"{name}_{'mixed' if mix else 'synthetic'}"][s] = dict(
                    auc=float(roc_auc_score([c.label for c in cs], probs)),
                    n=len(cs),
                    predictions=[
                        dict(id=c.conv_id, label=c.label, score=float(p)) for c, p in zip(cs, probs)
                    ],
                )
    # A source probe on legitimate text, separate from scam detection.
    tr = [c for c in synthetic + real if c.label == 0 and c.split == "train"]
    te = [c for c in synthetic + real if c.label == 0 and c.split in ("val", "val_real")]
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    clf.fit([style_vector(c) for c in tr], [int(c.generator == "human") for c in tr])
    probs = clf.predict_proba([style_vector(c) for c in te])[:, 1]
    out["legitimate_source_probe"] = dict(
        auc=float(roc_auc_score([int(c.generator == "human") for c in te], probs)),
        n=len(te),
        predictions=[
            dict(id=c.conv_id, label=int(c.generator == "human"), score=float(p))
            for c, p in zip(te, probs)
        ],
        caveat="Separates human/domain from template/domain; does not prove detector reliance on authorship.",
    )
    return out
