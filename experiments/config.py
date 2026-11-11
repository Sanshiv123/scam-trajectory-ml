"""Validated experiment settings. Configuration has no ML or network side effects."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunConfig:
    mode: str = "investigation"
    seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 30
    threads: int = 4
    data_seed: int = 42
    train_pairs: int = 40
    val_pairs: int = 20
    test_pairs: int = 20
    ood_pairs: int = 24
    enron_train_val: int = 200
    esconv: int = 150
    annomi: int = 133
    secondary_enron: int = 200
    max_real_turns: int = 64
    hidden_dim: int = 64
    learning_rate: float = 0.01
    skip_text: bool = False
    prepare_only: bool = False

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.mode not in ("corrected-mvp", "investigation"):
            raise ValueError("mode must be corrected-mvp or investigation")
        if not self.seeds or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if 0 not in self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must include reference seed 0 without duplicates")
        for name in (
            "epochs",
            "threads",
            "train_pairs",
            "val_pairs",
            "test_pairs",
            "ood_pairs",
            "enron_train_val",
            "esconv",
            "annomi",
            "secondary_enron",
            "max_real_turns",
            "hidden_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.data_seed) is not int or self.data_seed < 0:
            raise ValueError("data_seed must be a nonnegative integer")
        if not isinstance(self.learning_rate, (int, float)) or not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        for name in ("skip_text", "prepare_only"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    def as_dict(self):
        return asdict(self)


def load_config(path: Path, overrides=None) -> RunConfig:
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("config must be a JSON object")
    settings.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = set(settings) - {f.name for f in fields(RunConfig)}
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    if "seeds" in settings:
        settings["seeds"] = tuple(settings["seeds"])
    return RunConfig(**settings)
