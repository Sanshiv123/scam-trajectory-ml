"""Backward-compatible CLI. Reusable implementations live in the component packages."""

from importlib import import_module
from experiments.cli import main

_COMPAT = {
    "labels": ("data.labels", "labels"),
    "features": ("features.extraction", "extract_features"),
    "shuffle_prefix": ("experiments.training", "shuffle_prefix"),
    "fit_gru": ("experiments.training", "fit_gru"),
    "score_gru": ("experiments.training", "score_gru"),
    "fit_text": ("models.text", "fit_text"),
    "style_vector": ("experiments.probes", "style_vector"),
    "shortcut_probes": ("experiments.probes", "shortcut_probes"),
    "dump": ("experiments.artifacts", "dump"),
    "save_experiment": ("experiments.artifacts", "save_experiment"),
    "write_manifest": ("experiments.artifacts", "write_manifest"),
}


def __getattr__(name):
    if name not in _COMPAT:
        raise AttributeError(name)
    module, member = _COMPAT[name]
    return getattr(import_module(module), member)


if __name__ == "__main__":
    main()
