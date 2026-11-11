import unittest
import numpy as np
import torch
from types import SimpleNamespace

from data.investigation_data import build_dataset, load_real, validate
from eval.evaluate import calibrate_threshold, at_fpr
from features.emotion_features import turn_features
from models.baseline import position_features
from models.trajectory_model import TrajectoryModel
from run_experiments import shuffle_prefix, labels


class IntegrityTests(unittest.TestCase):
    def test_calibration_controls_ties_and_small_samples(self):
        for n in (1, 12, 60, 101):
            convs = [SimpleNamespace(conv_id=i) for i in range(n)]
            for values in (np.linspace(0, 1, n), np.ones(n), np.arange(n)//3,
                           np.ones(n,dtype=np.float32),np.linspace(0,1,n,dtype=np.float32)):
                scores = {i:np.array([values[i]]) for i in range(n)}
                for target in (0., .01, .05, .10):
                    threshold = calibrate_threshold(convs, scores, target)
                    self.assertLessEqual(int(np.count_nonzero(values>=threshold)),int(np.floor(target*n)))

    def test_no_calibration_fallback(self):
        with self.assertRaises(ValueError): calibrate_threshold([], {})

    def test_ask_turn_is_not_early(self):
        c=SimpleNamespace(conv_id=0,harmful_turn_idx=2)
        self.assertEqual(at_fpr([c],.5,{0:np.array([0.,0.,1.])})["early_recall"],0.)
        self.assertEqual(at_fpr([c],.5,{0:np.array([0.,1.,1.])})["early_recall"],1.)

    def test_position_has_no_future_length(self):
        np.testing.assert_array_equal(position_features(10),position_features(20)[:10])

    def test_dataset_structure_and_holdouts(self):
        data=build_dataset();validate(data)
        byid={c.conv_id:c for c in data}
        train=[c for c in data if c.split=="train"]
        self.assertNotIn("payroll_redirect",{c.pattern for c in train})
        self.assertNotIn("qwen",{c.generator for c in train})
        for c in data:
            if c.label:
                twin=byid[c.conv_id+1]
                self.assertEqual(c.turns[:2],twin.turns[:2])
                self.assertEqual(c.speakers,twin.speakers)
                self.assertIn(1,c.speakers[:c.harmful_turn_idx])
                self.assertEqual(c.phases[c.harmful_turn_idx],"ask")
                if c.split=="test_slow": self.assertGreaterEqual(c.harmful_turn_idx/(len(c.turns)-1),.7)
        self.assertGreater(len({c.harmful_turn_idx for c in data if c.label and c.split!="test_slow"}),3)
        self.assertEqual([c.turns for c in data],[c.turns for c in build_dataset()])

    def test_real_holdout_disjoint_and_ids_unique(self):
        mixed,holdout,_=load_real();validate(mixed+holdout)
        self.assertFalse({c.conv_id for c in mixed}&{c.conv_id for c in holdout})
        self.assertEqual(len(holdout),200)

    def test_prefix_scores_are_causal(self):
        torch.set_num_threads(2)
        model=TrajectoryModel()
        f=np.random.default_rng(1).random((12,15))
        np.testing.assert_allclose(model.running_score(f)[:6],model.running_score(f[:6]),atol=1e-6)

    def test_training_and_batched_scoring_with_prefix_loss(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(4)
        seqs=[rng.random((3,15)),rng.random((5,15))]
        ys=[np.array([0,0,1.]),np.array([0,0,0,0,0.])]
        model=TrajectoryModel(epochs=2).fit(seqs,ys,last_only=True)
        batched=model.running_scores(seqs)
        self.assertEqual([len(v) for v in batched],[3,5])
        for seq,pred in zip(seqs,batched):
            self.assertTrue(np.isfinite(pred).all())
            np.testing.assert_allclose(pred,model.running_score(seq),atol=1e-6)

    def test_shuffle_cannot_see_future_or_retain_old_deltas(self):
        raw=np.random.default_rng(0).random((12,7))
        f=turn_features(raw,list(range(12)))
        a=shuffle_prefix(f,5,0)
        f[6:]=9999
        np.testing.assert_array_equal(a,shuffle_prefix(f,5,0))
        np.testing.assert_allclose(a[:,7:14],np.diff(a[:,:7],axis=0,prepend=a[:1,:7]))
        np.testing.assert_array_equal(a[-1,:7],raw[5])

    def test_soft_labels_never_change_test_annotation(self):
        c=next(c for c in build_dataset() if c.label)
        old=c.harmful_turn_idx
        y=labels(c,early=True)
        self.assertTrue(np.any(y[:old]==.5));self.assertEqual(c.harmful_turn_idx,old)


if __name__=="__main__":unittest.main()
