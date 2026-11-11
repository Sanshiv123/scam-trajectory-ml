"""Regression checks for the refactor, using archived data/models without downloads."""

import ast
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import joblib
import numpy as np
import torch

from data.labels import labels
from data.schema import Conversation
from experiments.cli import main
from experiments.config import ROOT, RunConfig, load_config
from experiments.datasets import prepare_dataset
from experiments.run_store import new_run, output_path
from experiments.training import fit_gru, shuffle_prefix
from features.emotion_features import turn_features
from models.baseline import position_features
from models.trajectory_model import TrajectoryModel


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def test_configs_reject_unknown_and_invalid_values(self):
        for changes in (
            {"seeds": (1, 2)},
            {"seeds": (0, 0)},
            {"epochs": 0},
            {"learning_rate": float("nan")},
            {"data_seed": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(RunConfig(), **changes)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            p = Path(directory) / "bad.json"
            p.write_text('{"unknown_setting": 1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(p)

    def test_run_names_cannot_escape_output_root(self):
        for name in ("../outside", "a/b", "", "x" * 65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                output_path(RunConfig(), name)
        self.assertEqual(output_path(RunConfig(), "trial-1").parent, ROOT / "outputs" / "runs")

    def test_run_lifecycle_preserves_existing_and_failed_runs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            path = Path(directory) / "run"
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with new_run(path, RunConfig()):
                    (path / "evidence.txt").write_text("keep me")
                    raise RuntimeError("fixture failure")
            self.assertEqual(json.loads((path / "run_state.json").read_text())["stage"], "failed")
            with self.assertRaises(FileExistsError):
                with new_run(path, RunConfig()):
                    self.fail("Must not enter an existing directory")
            self.assertEqual((path / "evidence.txt").read_text(), "keep me")

    def test_dry_run_has_no_model_import_or_write(self):
        code = """
import sys
from experiments.cli import main
main(['--config','configs/investigation.json','--run-name','dry-fixture','--dry-run'])
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertFalse(Path(config["output"]).exists())

    def test_presets_reproduce_both_archived_datasets(self):
        for mode in ("corrected-mvp", "investigation"):
            config = load_config(ROOT / "configs" / f"{mode}.json")
            bundle = prepare_dataset(config)
            value = [asdict(c) for c in bundle.conversations]
            digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            archived = json.loads((ROOT / "outputs" / mode / "manifest.json").read_text())
            self.assertEqual(digest, archived["dataset_hash"], mode)

    def test_legacy_schema_and_entry_points_still_resolve(self):
        from data.generate_data import Conversation as OldConversation
        from run_experiments import labels as old_labels, shuffle_prefix as old_shuffle
        from eval.summarize import summarize as old_summary
        from reporting.summary import summarize

        self.assertIs(OldConversation, Conversation)
        self.assertIs(old_labels, labels)
        self.assertIs(old_shuffle, shuffle_prefix)
        self.assertIs(old_summary, summarize)

    def test_saved_models_keep_their_predictions(self):
        torch.set_num_threads(2)
        folder = ROOT / "outputs" / "corrected-mvp"
        dataset = json.loads((folder / "dataset.json").read_text())[:12]
        frozen = np.load(folder / "frozen_emotion_features.npz")
        expected = np.load(folder / "synthetic/gru/seed-0/predictions.npz")
        model = TrajectoryModel()
        model.net.load_state_dict(
            torch.load(folder / "synthetic/gru/seed-0/model.pt", weights_only=True)
        )
        position = joblib.load(folder / "synthetic/position/seed-0/model.joblib")
        pos_expected = np.load(folder / "synthetic/position/seed-0/predictions.npz")
        for c in dataset:
            key = str(c["conv_id"])
            np.testing.assert_allclose(model.running_score(frozen[key]), expected[key], atol=2e-6)
            np.testing.assert_allclose(
                position.predict_proba(position_features(len(c["turns"]))), pos_expected[key]
            )

    def test_gru_training_matches_archived_implementation(self):
        with zipfile.ZipFile(ROOT / "outputs/corrected-mvp/source.zip") as archive:
            source = archive.read("run_experiments.py").decode("utf-8-sig")
        names = {"labels", "shuffle_prefix", "fit_gru"}
        functions = [
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        namespace = dict(np=np, TrajectoryModel=TrajectoryModel, turn_features=turn_features)
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), "archived_runner", "exec"),
            namespace,
        )
        config = replace(
            load_config(ROOT / "configs/corrected-mvp.json"),
            train_pairs=1,
            val_pairs=1,
            test_pairs=1,
            ood_pairs=1,
        )
        train = [c for c in prepare_dataset(config).conversations if c.split == "train"]
        rng = np.random.default_rng(8)
        feats = {c.conv_id: rng.random((len(c.turns), 15)) for c in train}
        for options in ({}, {"shuffled": True}):
            old = namespace["fit_gru"](train, feats, 0, 2, **options)
            new = fit_gru(train, feats, 0, 2, **options)
            for key, value in old.net.state_dict().items():
                np.testing.assert_allclose(
                    value.numpy(), new.net.state_dict()[key].numpy(), atol=1e-7
                )

    def test_small_suite_wires_training_storage_reports_and_verification(self):
        # Synthetic fixture features exercise the orchestration only. Temporary
        # outputs are never presented as pretrained-model experimental evidence.
        def fixture_encoder(texts, kind):
            self.assertEqual(kind, "emotion")
            vector = np.array([0.05, 0.05, 0.05, 0.05, 0.70, 0.05, 0.05], dtype=np.float32)
            return np.tile(vector, (len(texts), 1)), {
                "model": "TEST_FIXTURE",
                "revision": "test-only",
            }

        from experiments.runner import run_suite

        config = replace(
            load_config(ROOT / "configs/corrected-mvp.json"),
            train_pairs=1,
            val_pairs=1,
            test_pairs=1,
            ood_pairs=1,
            epochs=1,
            seeds=(0,),
            threads=2,
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            output = Path(directory) / "fixture-run"
            with patch("features.extraction.encode", side_effect=fixture_encoder):
                with new_run(output, config), patch("sys.stdout", new_callable=io.StringIO):
                    run_suite(config, output)
            self.assertTrue((output / "COMPLETE").exists())
            self.assertEqual(
                json.loads((output / "run_state.json").read_text())["stage"], "completed"
            )
            self.assertIn("Corrected original MVP", (output / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
