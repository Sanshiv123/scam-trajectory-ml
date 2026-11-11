# Investigation rebuild

Base: original MVP commit 67cddb6 on main. The other checkpoint branch is not merged.

1. Correct the original MVP evaluation and rerun its original synthetic data.
2. Build v2 matched templates with real recipient interaction, wider ask timing,
   identical twin openings and identical speaker schedules. This remains a
   controlled template benchmark, not a realistic or human-validated scam corpus.
3. Run synthetic-only and real-negative-mixed training on identical test sets.
   Calibrate on synthetic validation twins in both conditions; report calibration,
   synthetic test, and independent real-Enron false-positive rates explicitly.
4. Run long/slow paired test variants, feature ablations, a causal shuffled-prefix
   control, and a pre-ask training-label experiment using training patterns only.
5. Rebuild length/style shortcut probes, TF-IDF and frozen DistilBERT text
   references. Frozen DistilBERT is not the proposal's fine-tuned ceiling.
6. Save datasets, configuration, source and model hashes, dependency lock,
   per-turn predictions, per-conversation metrics, model weights and plots.
   Repeat trainable neural models across seeds and report uncertainty.
7. Generate the checkpoint narrative from actual saved results. Historic
   numbers are not targets and will not be reused as experimental evidence.

No test-pattern labels enter training or hyperparameter selection. No human
audit, real-LLM generation, restyled Enron, or fine-tuned text result is claimed
without completing that work. LLM generation and a manual realism audit remain
separate limitations; a model download does not generate new conversations.
