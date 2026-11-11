"""Assemble train/calibration/test pools from a validated configuration."""

from dataclasses import dataclass

from data.generate_data import build_dataset as legacy_dataset
from data.investigation_data import build_dataset, load_real, validate
from experiments.config import RunConfig


@dataclass
class DatasetBundle:
    synthetic: list
    real: list
    holdout: list
    audit: dict

    @property
    def conversations(self):
        return self.synthetic + self.real + self.holdout


def prepare_dataset(config: RunConfig) -> DatasetBundle:
    sizes = dict(seed=config.data_seed)
    if config.mode == "corrected-mvp":
        synthetic = legacy_dataset(
            **sizes,
            n_pairs_train=config.train_pairs,
            n_pairs_val=config.val_pairs,
            n_pairs_test=config.test_pairs,
            n_pairs_ood=config.ood_pairs,
        )
        for conversation in synthetic:
            conversation.pair_id = f"legacy-{conversation.conv_id // 2}"
        real, holdout, real_audit = [], [], {}
    else:
        synthetic = build_dataset(
            **sizes,
            train_pairs=config.train_pairs,
            val_pairs=config.val_pairs,
            test_pairs=config.test_pairs,
            ood_pairs=config.ood_pairs,
        )
        real, holdout, real_audit = load_real(
            seed=config.data_seed,
            enron_train_val=config.enron_train_val,
            esconv=config.esconv,
            annomi=config.annomi,
            secondary=config.secondary_enron,
        )
        real_audit["prefix_limit"] = dict(
            max_turns=config.max_real_turns,
            clipped=sum(len(c.turns) > config.max_real_turns for c in real + holdout),
        )
        for conversation in real + holdout:
            conversation.turns = conversation.turns[: config.max_real_turns]
            conversation.speakers = conversation.speakers[: config.max_real_turns]
    validation = validate(synthetic + real + holdout)
    validation.update(synthetic_n=len(synthetic), real_n=len(real) + len(holdout))
    return DatasetBundle(synthetic, real, holdout, dict(validation=validation, real=real_audit))
