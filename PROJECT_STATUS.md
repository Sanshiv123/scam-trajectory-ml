# Current project assessment

The current detector does not demonstrate useful early detection on the revised
benchmark. The investigation is still viable, but the dataset's realism and
surface shortcuts limit the strength of both positive and negative conclusions.

## What the evidence supports

- The corrected original MVP runs end to end and has reproducible outputs.
- On the revised standard test, the emotion GRU averages AUC 0.557 and 0% early
  recall at the target 5% operating point across three seeds. Observed FPR is 8.9%.
- Delayed-ask early recall is 51.7%, but legitimate-twin FPR is 82.2%. That is a
  timing/calibration-transfer failure, not useful robustness.
- The soft pre-ask label experiment did not improve standard-test early recall.
- TF-IDF and frozen-text models can classify the templates extremely well.
  A surface-style probe also gets synthetic-test AUC 1.000. This is strong reason
  to audit the template benchmark; it does not establish real scam recognition.

These findings do not show that emotion can never help social-engineering
detection. They describe these features, training choices and datasets. The
revised generator, pairing constraints and training set changed together, so the
drop from the original MVP cannot be attributed to one specific correction.

## Relation to the MVP feedback

The professor called the prototype, baselines and separate earliness metric a
good start. The feedback identified three issues rather than declaring the
project a failure:

1. Validation FPR could exceed its own target. Conservative calibration is now
   tested for ties and float32 scores; achieved validation and test FPRs are saved.
   Test-distribution drift is a different problem and is still present.
2. Pre-ask training labels deserved investigation. That experiment is now run
   without training on held-out patterns; it did not rescue the current model.
3. The generator holdout changes template style, not the actual LLM. The code and
   report retain that limitation and make no real-LLM generalization claim.

## What remains weak

The most serious issue is the benchmark. It is built from small fixed wording
banks and has no completed human realism audit or independent real-scam test.
The long test primarily changes setup length and ask timing; it does not validate
subtle or adaptive evasion. Human-written source text also differs in domain and
style from the synthetic examples. Architecture cleanup cannot resolve these
scientific limitations.

## Recommended next iteration

1. Audit matched scam/legitimate examples and model errors using the existing
   worksheet and error inventory. Verify the ask, legitimate process, realism and
   any label-correlated wording/formatting. Record human judgments rather than
   treating template construction as validation.
2. Repair the strongest surface shortcuts and add paraphrase/scenario diversity.
   If real LLM generation is used, generate both classes symmetrically and repeat
   the shortcut checks; LLM output alone is not a realism guarantee.
3. Test length-matched calibration on validation examples and evaluate on fresh,
   untouched delayed-ask scenarios. Do not tune thresholds on the current test
   results and then report them as held-out performance.
4. Keep the next model comparison small: isolated-message, aggregate, GRU and a
   text reference, with a stated hypothesis and unchanged matched test sets.

The investigation checkpoint can honestly describe failed hypotheses and the
experiments that exposed them. For the final report, stronger dataset validation
and a focused follow-up would add more value than adding model families.

Evidence: `outputs/investigation/FINDINGS.md`, `summary.csv`, `shortcut_probes.json`,
`paired_bootstrap.json`, the per-model predictions and the reviewed PDF. These
completed artifacts predate the structural refactor and remain unchanged.
