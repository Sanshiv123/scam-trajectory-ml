# Emotional-Trajectory Detection of Social Engineering — MVP + Evaluation Pipeline

**Team 6:** Sanchusri Kavitha Babu, Rahul Ajith Kumar, Sisvanthkumar Sathivadivel

This is the MVP + Evaluation Pipeline checkpoint for our project. It's a
working end-to-end pipeline (data -> features -> scorers -> evaluation)
that we can run today, currently running fully on synthetic data: the
scam/matched-twin conversations are still template-generated placeholders
(we don't have an API key for a real LLM in this environment), and the
legitimate side is synthetic too for this run. Numbers in
`outputs/report.txt` demonstrate that the pipeline runs end to end and
produces something interpretable, not final results.

We also built and validated a real-data path for the legitimate side
(real Enron email threads + real ESConv/AnnoMI conversations, plus a
secondary false-positive check against real Enron text) -- see
`data/external_data.py` and `data/external/*.json`. It's not active in
this run (we decided to keep this checkpoint synthetic-only), but the
code and the downloaded data are in the repo and ready to switch back on;
see "Built but not used right now" below.

## What's in scope for this checkpoint

From our proposal's full design, we're keeping four pieces for now and
deferring the rest to later checkpoints:

**Built now:**
- The 4 real escalation patterns from our proposal (executive wire
  request, vendor bank-detail change, IT helpdesk vishing, payroll
  redirect), each paired with a matched legitimate "twin" conversation
  (same roles, same amount, same deadline, resolved through a normal
  process instead of manipulation) -- this removes "LLM-written vs.
  normal text" as a shortcut.
- Real per-turn emotion features from the pretrained classifier named in
  our proposal (`j-hartmann/emotion-english-distilroberta-base`).
- Four of our six scorers: position-only, message-level (emotion),
  running aggregate (the TA-suggested baseline -- running mean and
  running max of message-level scores so far, averaged together), and
  the trajectory model (1-layer GRU, 64 hidden units) -- our main model.
- The held-out splits: one pattern (payroll redirect) and one generator
  voice held out entirely from training, so we can check generalization.
- The opening-turn leakage shortcut check (can a simple classifier tell
  scam from twin using only the first 2 turns?).

**Built but not used right now:**
- A real-data path for the legitimate side: real Enron email threads
  (parsed from the public Enron corpus into actual back-and-forth
  threads, deduplicated, fraud-investigation threads filtered out) and
  real ESConv/AnnoMI conversations (`data/external_data.py`,
  `data/external/*.json` -- ~1,500 Enron threads, ~1,300 ESConv, 133
  AnnoMI, already downloaded and parsed).
- A secondary FPR check against real, held-out Enron threads
  (`eval/evaluate.py::secondary_fpr_check`), to test whether a threshold
  calibrated on synthetic validation twins still holds its false-positive
  rate on real text.
- We built and ran this end to end earlier and it surfaced a real
  finding (position-only badly over-fires on real text), but decided to
  keep this checkpoint fully synthetic so the numbers below are a clean
  baseline. None of this code was deleted -- `run_mvp.py` just doesn't
  call it right now.

**Deferred to later checkpoints:**
- Real LLM-generated scam/matched-twin conversations -- still
  template-generated, since this environment has no API key for an
  external model. `data/llm_backend.py` has the wiring ready.
- The ~300 Enron threads reserved for the authorship-restyling shortcut
  check (`data/external/enron_authorship_source.json`) are pulled and
  ready, but not yet restyled by an LLM, for the same reason.
- The two text-based scorers (message-level and trajectory on raw text,
  fine-tuned DistilBERT) that serve as our reference ceiling.
- The rest of the shortcut-check suite (authorship probe, position +
  correlation-with-length check, human realism audit).
- The slow-burn evaluation (gradual-escalation conversations).
- Feature ablations.

## Files

| File | What it does |
|---|---|
| `data/generate_data.py` | Builds the synthetic scam/twin conversation pairs across all splits (train/val/test + 3 held-out splits) |
| `data/llm_backend.py` | Stubbed wiring for a real LLM backend (OpenAI/Anthropic) to replace the template generator -- not called yet, no API key in this environment |
| `data/external_data.py` | Loads the real Enron/ESConv/AnnoMI data from `data/external/*.json` into the same Conversation format -- built and validated, not called by `run_mvp.py` in this run |
| `features/emotion_features.py` | Runs the pretrained emotion classifier and builds the 15-dim per-turn feature vector (7-class emotion distribution + delta from previous turn + speaker flag) |
| `models/baseline.py` | Position-only and message-level (emotion) scorers, plus the running-aggregate transform |
| `models/trajectory_model.py` | The GRU trajectory model |
| `eval/evaluate.py` | AT@FPR earliness metric, classification metrics, opening-leakage check, secondary FPR check |
| `run_mvp.py` | Runs everything end to end and writes `outputs/report.txt` + two plots |

## Running it

```bash
pip install -r requirements.txt   # scikit-learn, numpy, matplotlib, torch, transformers
python3 run_mvp.py
```

The first run downloads the emotion classifier's weights (needs internet
access once; cached after that). Produces `outputs/report.txt` and two
plots in `outputs/`.

## What we found

**The three baselines (position-only, message-level, running aggregate)
sit at or near chance and never fire early.** AUC 0.36-0.65 across all
four splits, and 0% early_recall for all three, on every split, at both
FPR targets we calibrated to (5%, 10%).

**The trajectory (GRU) model clearly separates scam from legitimate, and
now shows some real earliness in-distribution.** AUC 0.75-0.91 across
all four splits, beating every baseline everywhere. On the
in-distribution test split specifically, early_recall is 9% at the
5%-FPR threshold and 27% at the 10%-FPR threshold (median lead of 1 turn
when it fires early). That's new since we added class-balanced loss
(a `pos_weight` term, see `models/trajectory_model.py`) to the GRU's
training objective -- before that change, every scorer including the
GRU showed 0% early_recall everywhere.

**But that earliness doesn't generalize yet.** On all three held-out
splits (unseen pattern, unseen generator voice, both), the GRU's
early_recall is still 0%, even though its classification AUC on those
same splits is high (0.75-0.91). So it can tell a held-out scam
conversation apart from a legitimate one by the end of the conversation,
but it hasn't learned to flag a *novel* escalation pattern or generator
voice before the harmful ask -- only the pattern/voices it was trained
on. Most likely cause: our per-turn training label is only 1 at/after
the ask turn, so the model never gets a training signal that earlier
urgency/authority turns in an unfamiliar pattern belong to the same kind
of escalation -- worth trying next, by widening how we label the pre-ask
turns during training.

The opening-turn leakage check comes out near chance (0.47), which is a
good sign: the generation process isn't leaking the scam/twin label into
the rapport turns.

## What's still not real data

1. **Everything in this run.** `data/generate_data.py` fills in template
   line banks (`PATTERNS` dict) with simple text transforms standing in
   for different LLMs (`GENERATORS`), because this environment has no API
   key for an external model. `data/llm_backend.py` has example wiring
   ready for OpenAI/Anthropic calls -- the line banks become prompts
   instead of literal text once a key is available. The legitimate side
   is template-generated too in this run, even though a real-data path
   exists and works (see "Built but not used right now" above) -- we
   kept this checkpoint synthetic-only for a clean baseline.
2. **The authorship-restyling shortcut check.** We pulled 300 real Enron
   threads specifically to be rewritten by an LLM for this check
   (`data/external/enron_authorship_source.json`), but haven't generated
   the restyled versions yet, same reason as above.
3. **The two text-based scorers and the DistilBERT reference ceiling** --
   not implemented yet.
4. **The rest of the shortcut checks (authorship probe, full position +
   correlation analysis, human realism audit) and the slow-burn split.**
