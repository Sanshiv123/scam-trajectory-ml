"""TF-IDF and frozen-embedding text references. No transformer fine-tuning here."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data.labels import labels


def fit_text(train, data, kind, embeddings=None):
    y = np.concatenate([labels(c) for c in train])
    if embeddings is None:
        # Prefix strings are causally constructed. Fit vocabulary on train only.
        def text(c):
            return (
                c.turns
                if kind == "message"
                else [" ".join(c.turns[: t + 1]) for t in range(len(c.turns))]
            )

        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), max_features=6000),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        model.fit([s for c in train for s in text(c)], y)
        scores = {c.conv_id: model.predict_proba(text(c))[:, 1] for c in data}
    else:

        def x(c):
            f = embeddings[c.conv_id]
            return (
                f if kind == "message" else np.cumsum(f, axis=0) / np.arange(1, len(f) + 1)[:, None]
            )

        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=1500)
        )
        model.fit(np.concatenate([x(c) for c in train]), y)
        scores = {c.conv_id: model.predict_proba(x(c))[:, 1] for c in data}
    return model, scores
