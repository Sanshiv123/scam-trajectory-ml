import unittest

from eval import preask_divergence_audit as audit


def _conv(conv_id, pair_id, pattern, turns, phases, harmful_turn_idx):
    return dict(
        conv_id=conv_id, pair_id=pair_id, pattern=pattern,
        turns=turns, phases=phases, harmful_turn_idx=harmful_turn_idx,
    )


class PreaskDivergenceAuditTests(unittest.TestCase):
    """Covers eval/preask_divergence_audit.py, added for the final submission
    to answer the Nov 16 checkpoint feedback. Uses an inline fixture (via
    monkeypatched load()) instead of outputs/investigation/*.json, so this
    does not depend on, or get invalidated by, any future re-run."""

    def setUp(self):
        # p0: scam and twin share phase sequence but diverge in pre-ask
        # content at the authority turn -- a content difference, not a
        # structural leak. p1: scam and twin are pre-ask-identical.
        self.data = [
            _conv(0, "p0", "bec",
                  turns=["open", "please don't tell anyone", "ask"],
                  phases=["open", "authority", "ask"], harmful_turn_idx=2),
            _conv(1, "p0", "bec",
                  turns=["open", "I'll loop in Alex", "ask"],
                  phases=["open", "authority", "ask"], harmful_turn_idx=2),
            _conv(2, "p1", "bec",
                  turns=["open", "same line", "ask"],
                  phases=["open", "authority", "ask"], harmful_turn_idx=2),
            _conv(3, "p1", "bec",
                  turns=["open", "same line", "ask"],
                  phases=["open", "authority", "ask"], harmful_turn_idx=2),
        ]
        self.inventory = [
            dict(id=0, split="test", kind="late_or_missed", max_score=0.4),
            dict(id=2, split="test", kind="late_or_missed", max_score=0.3),
        ]
        self._orig_load = audit.load
        audit.load = lambda: (self.data, self.inventory)

    def tearDown(self):
        audit.load = self._orig_load

    def test_finds_matched_twin_by_pair_id_not_conv_id(self):
        rows = audit.run()
        self.assertEqual({r["conv_id"] for r in rows}, {0, 2})

    def test_detects_pre_ask_content_difference_at_matched_phase(self):
        row = next(r for r in audit.run() if r["conv_id"] == 0)
        self.assertEqual(row["n_differing_turns_pre_ask"], 1)
        self.assertEqual(row["differing_phases"], ["authority"])
        self.assertFalse(row["phase_sequence_differs"])

    def test_zero_difference_when_twin_is_identical_pre_ask(self):
        row = next(r for r in audit.run() if r["conv_id"] == 2)
        self.assertEqual(row["n_differing_turns_pre_ask"], 0)

    def test_diff_never_reaches_the_ask_turn(self):
        for r in audit.run():
            self.assertTrue(all(i < r["ask_idx"] for i in r["differing_turn_indices"]))

    def test_summarize_aggregates_counts_and_phases(self):
        summary = audit.summarize(audit.run())
        self.assertEqual(summary["n_missed"], 2)
        self.assertEqual(summary["n_with_zero_preask_difference"], 1)
        self.assertEqual(summary["n_with_content_difference_pre_ask"], 1)
        self.assertEqual(summary["differing_phase_counts"], {"authority": 1})


if __name__ == "__main__":
    unittest.main()
