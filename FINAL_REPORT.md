# Emotional Trajectory Detection of Social Engineering — Final Project Report

**Team 6:** Sanchusri Kavitha Babu, Rahul Ajith Kumar, Sisvanthkumar Sathivadivel
**Code:** https://github.com/Sanshiv123/scam-trajectory-ml/tree/sanchu/final-submission

This report covers the project from the original MVP through this final
submission: the MVP, feedback on it, the rebuilt checkpoint, feedback on that,
and the new analysis this submission adds in response. Every number below
comes from saved artifacts in the repository (`outputs/investigation/`,
`outputs/corrected-mvp/`), not a re-typed or historical figure.

---

## 1. The question

Can an ordered sequence of per-turn emotion features detect a harmful request
earlier than a single-message detector, at a comparable false-alarm rate? We
also tested whether any advantage transfers to an unseen escalation pattern, an
unseen phrasing style, a longer delayed-ask conversation, and real (non-synthetic)
legitimate text.

## 2. Project arc

| Stage | Date | What happened |
|---|---|---|
| MVP + evaluation pipeline | Oct 28 | First working pipeline: 4 scorers (position, message-level, running-aggregate, trajectory GRU), synthetic matched-twin data, AT@FPR earliness metric. |
| Instructor feedback on MVP | — | Called the prototype, baselines and earliness metric a good start, but flagged three concrete issues: (1) validation FPR could exceed its own target; (2) pre-ask training labels deserved investigation; (3) the "generator" holdout changes template style, not a real LLM. |
| Investigation & iteration checkpoint (rebuilt) | Nov 11 | Full rebuild addressing all three points: conservative tie-aware calibration with reported actual FPR, a soft pre-ask-label experiment, explicit "style not LLM" labeling throughout. Added real-data mixing, a delayed-ask stress test, 9 feature/causal ablations, shortcut probes, and text-model references (TF-IDF, frozen DistilBERT), all averaged over 3 training seeds. |
| Instructor feedback on checkpoint | Nov 16 | Confirmed the delayed-ask "improvement" is a calibration failure (false alarms rise even more than recall does), confirmed early recall is still 0% after the label experiment, and asked for a specific audit: compare missed scams against their legitimate twins before the request and check whether any difference goes beyond template wording. |
| This final report | Nov 17 | Directly answers that audit request (Section 5) and draws the project's final conclusions. |

## 3. What we tried

- **Four core scorers** on per-turn emotion features (7-dim distribution, 7-dim
  turn-to-turn change, 1-dim speaker flag): position-only, isolated-message
  logistic regression, a running mean/max aggregate, and a 1-layer 64-unit GRU.
- **Real-data mixing**: 410 real Enron/ESConv/AnnoMI conversations mixed into
  training, 73 held out for validation, 200 separate Enron threads held out
  purely for an independent false-positive check never used in training or
  calibration.
- **Nine ablations**: GRU and logistic variants with/without delta features,
  with/without the speaker flag, emotion-distribution-only.
- **A causal shuffled-prefix control**: shuffle only the turns already seen,
  keep the current turn fixed, recompute deltas — isolates whether the model
  uses turn *order* or just turn *content*.
- **A soft pre-ask label experiment**: give urgency/authority turns in
  *training* patterns a softer positive label, to see whether that teaches the
  model to generalize escalation itself rather than memorizing the trained
  pattern. Held-out pattern labels were never touched.
- **A delayed-ask stress test**: the same scam patterns, stretched to 20-32
  turns with the ask pushed to 70-95% of the way through, twins matched on the
  same length/speaker schedule/position.
- **Shortcut probes**: can conversation length alone, or surface writing style
  alone, or raw text (TF-IDF / frozen DistilBERT embeddings) separate scam from
  legitimate without reading escalation at all?
- **This report's new addition**: a direct per-turn audit of every scam the
  headline model missed, against its matched twin, to see whether real content
  differences exist before the ask turn (Section 5).

## 4. What we found

**Standard test** (in-distribution, target 5% FPR, mean over 3 seeds):

| Scorer | AUC | Early recall @5% | Observed FPR @5% |
|---|---:|---:|---:|
| Position-only | 0.500 | 0.0% | 0.0% |
| Isolated-message | 0.448 | 0.0% | 0.0% |
| Running aggregate | 0.484 | 1.7% | 3.3% |
| Trajectory GRU | 0.557 | 0.0% | 8.9% |

**GRU generalization:**

| Split | AUC | Early recall @5% | Observed FPR @5% |
|---|---:|---:|---:|
| Held-out pattern | 0.513 | 0.0% | 9.7% |
| Held-out style | 0.527 | 0.0% | 2.8% |
| Held-out both | 0.503 | 0.0% | 2.8% |
| Delayed-ask ("slow") | 0.595 | 51.7% | 82.2% |

The GRU is the only scorer consistently above chance on classification
(AUC 0.50-0.60 across every split), but it produces **0% early recall on the
standard test and every held-out split** — it cannot tell a scam from its
twin *before* the ask turn at a usable false-alarm rate, even though it can
tell them apart by the end of the conversation.

The delayed-ask result looked promising at first glance (51.7% early recall)
but the same threshold produces an 82.2% false-positive rate on that split's
*legitimate* twins — worse than flipping a coin. This is a calibration/timing
transfer failure: a threshold tuned on short conversations does not transfer
to long ones, not a genuine robustness gain.

The soft pre-ask label experiment did not rescue early recall (still 0.0% on
standard test). The shuffled-prefix causal control scores close to the real
model (AUC 0.527 vs 0.557), meaning the GRU is not obviously exploiting
turn *order* beyond what single-turn content already gives it.

Shortcut probes were the most actionable finding: conversation length alone
reaches AUC 0.62, and raw surface style/text reaches **AUC 1.000** on the
synthetic test. TF-IDF and frozen-DistilBERT text references also hit AUC
1.000. This means the synthetic benchmark can be solved by surface features alone
— length, raw text, writing style — with no reading of escalation involved.
Any result on it, positive or negative, depends on checking this first. The
real-Enron false-positive check stayed low for the GRU (0.3%)
and position-only (0.0%), but the frozen-DistilBERT prefix reference reached
36-37.5% — a scorer that looks perfect on synthetic text can still misfire
badly on real text.

## 5. New analysis: auditing the missed scams (answering the Nov 16 feedback)

The checkpoint feedback asked specifically: for scams the model missed,
compare them against their legitimate twins *before* the request, and check
whether any differences go beyond template wording.

We built `eval/preask_divergence_audit.py`, which pulls every conversation the
headline model (`synthetic/gru`) missed on the standard test split
(`outputs/investigation/error_inventory.json`, kind `late_or_missed`, 60
conversations), finds its matched twin by `pair_id`, and diffs their turns
strictly before the ask.

**Result: every single one of the 60 missed scams (100%) has at least one
pre-ask turn that differs from its twin in actual manipulative content, not
just wording** — specifically the "authority" phase turn in all 60 cases (and
the "urgency" phase turn in 15 of them). For example, in conversation 364 the
scam's authority turn is *"Please don't loop in anyone else on this, I want to
keep it contained until it's announced"* against the twin's *"I'll have Alex
countersign once the PO is in the system"* — not a paraphrase of the same
idea, but the opposite move (discouraging oversight vs. requesting it). The
phase schedule itself never differs between scam and twin (same number and
order of phases in every case), so this is a controlled, matched content
difference, not a structural leak.

The 0% early recall does not mean there is no signal before the ask. There
is, and it is concentrated in one phase: authority. The current pipeline
(general-purpose emotion features fed into a small GRU) is not using that
signal early enough, at a false-alarm rate we can afford. The next step is a
feature or model that reads authority/urgency language directly, instead of
general emotion, before concluding the hypothesis itself is wrong. This is
also the first next step `PROJECT_STATUS.md` already recommends (audit
matched scam/legitimate errors using the worksheet and error inventory), done
here as a script instead of left as an unfilled worksheet.

## 6. What worked / what didn't

**Worked:**

- Catching and fixing real evaluation bugs from the MVP (threshold ties,
  final-length leakage in the position baseline, ungrounded vocabulary in the
  opening-leakage probe) — the kind of correction that changes whether a
  result can be trusted at all, not just its magnitude.
- The real-data mixing experiment, which surfaced a concrete, previously
  invisible safety risk (position-only firing on real email at a high rate)
  that synthetic-only validation could not have caught.
- The causal shuffled-prefix control, which gives a real answer to "is the
  model using sequence information" rather than an assumption.
- Building reproducible, seed-averaged, immutable experiment runs with
  independent metric verification — removing "did we get the same numbers
  twice" as a source of doubt.
- This report's audit, which gives a concrete, falsifiable answer to exactly
  what the feedback asked, instead of a general restatement of the limitation.

**Didn't work:**

- The trajectory GRU does not demonstrate useful early warning on the
  standard benchmark (0% early recall at every realistic FPR target).
- The delayed-ask "improvement" is not real robustness — it's explained by a
  calibration/timing mismatch, and the feedback caught this correctly.
- The soft pre-ask label experiment did not change the standard-test result.
- The synthetic benchmark itself is compromised as an evaluation of "reading
  escalation" specifically, because trivial surface features (length, style,
  raw text) solve it almost perfectly — so a model that scored well on it
  would not have proven much either.

## 7. What we learned

- A 0% early-recall result is not proof there is no signal. Our audit found
  real content differences before the ask in 100% of missed cases — the
  pipeline just isn't using that signal in time.
- Two of the three most useful findings (the real-data FPR spike, the
  shortcut probes hitting AUC 1.000) are about the benchmark, not the model.
  A matched-twin generator needs its own shortcut audit before its numbers
  mean anything.
- Each round of feedback pointed at one checkable claim (calibration, label
  leakage, generalization scope), and acting on each produced this project's
  strongest results, including this report's audit.
- A clearly explained negative result is worth more here than an unexplained
  higher number, and every claim in this repo traces to a saved artifact
  (`PROJECT_STATUS.md`, `FINDINGS.md`, the run manifests).

## 8. Limitations

This remains a controlled template benchmark: four fixed BEC/vishing patterns,
four text-substitution "styles" standing in for different LLMs (not actual
multi-model generation), no completed human realism audit (the 150-example
worksheet is prepared but unfilled), and no independent real-scam test set.
The delayed-ask test changes setup length and ask timing, not adaptive or
subtle evasion. Real conversations (Enron/ESConv/AnnoMI) are presumed
legitimate based on the provided source files; their original collection and
threading were not independently re-verified. Paired-bootstrap and multi-seed
uncertainty intervals do not capture calibration or training-distribution
uncertainty beyond what was explicitly measured.

## 9. Reproducing this report

```bash
pip install -e .
scam-trajectory --config configs/investigation.json --run-name <name>
python eval/preask_divergence_audit.py   # Section 5, new in this report
python build_report.py --run outputs/investigation
```

Full per-split, per-seed tables are in `outputs/investigation/REPORT.md` and
`summary.csv`; this report summarizes them rather than duplicating every row.
