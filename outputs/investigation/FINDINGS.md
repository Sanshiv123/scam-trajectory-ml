# Interpretation of the rebuilt results

The emotion GRU achieves mean standard-test AUC 0.557 and early recall 0.0% at the target 5% operating point, with observed FPR 8.9%. This run does not demonstrate useful early warning on the standard revised benchmark. Its performance should not be equated with the corrected original-data rerun because the dataset, pairing constraints and training set changed together.

On the long/delayed-ask test, GRU early recall rises to 51.7%, but legitimate-twin FPR rises to 82.2%. This is a calibration/timing-transfer failure, not a robustness success. Adding real negatives gives standard-test AUC 0.505 and long-test FPR 51.1%; it does not establish a fix for early detection.

Removing deltas changes standard AUC to 0.523; removing the speaker flag gives 0.571. Shuffled prefixes give AUC 0.527, while soft pre-ask labels give 0.559 and early recall 0.0%. Small classification differences do not rescue the absent early-warning benefit. Seed-0 paired intervals below quantify conditional differences, not causal proof of a real-world emotion signal.

TF-IDF prefixes achieve standard AUC 1.000 and early recall 81.7%, but FPR on the held-out pattern is 100.0%. High transcript AUC and badly shifted alert thresholds can coexist. The template benchmark remains vulnerable to surface shortcuts; text-model superiority here does not establish realistic manipulation detection.
