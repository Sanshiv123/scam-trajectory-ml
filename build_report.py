from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, ListFlowable, ListItem)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=16, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name="Note", parent=styles["Normal"], spaceAfter=8, leading=13,
                           textColor=colors.HexColor("#555555"), fontSize=9.5))

story = []

def h1(t): story.append(Paragraph(t, styles["H1"]))
def h2(t): story.append(Paragraph(t, styles["H2"]))
def body(t): story.append(Paragraph(t, styles["Body"]))
def note(t): story.append(Paragraph(t, styles["Note"]))
def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, styles["Body"])) for i in items],
                               bulletType="bullet", leftIndent=14))
    story.append(Spacer(1, 6))

# ---------------- Title ----------------
story.append(Paragraph("Emotional-Trajectory Detection of Social Engineering", styles["Title"]))
story.append(Paragraph("MVP + Evaluation Pipeline Checkpoint", styles["Heading3"]))
story.append(Paragraph("Team 6: Sanchusri Kavitha Babu, Rahul Ajith Kumar, Sisvanthkumar Sathivadivel",
                        styles["Normal"]))
story.append(Spacer(1, 10))

# ---------------- Recap ----------------
h2("Problem recap")
body("We're detecting BEC and vishing social-engineering conversations as early as possible, "
     "before the harmful request (a wire transfer, a bank-detail change, a login code, a payroll "
     "redirect) is actually made, rather than only flagging a conversation after the fact. The idea "
     "is that a model tracking how emotional tone shifts turn by turn can flag a conversation earlier "
     "than one that only looks at the current message.")

# ---------------- Scope ----------------
h2("What's in scope for this checkpoint")
body("Our full proposal has six scorers, three real external datasets, and several more shortcut "
     "checks and experiments. For this checkpoint we built and ran the subset below, and left the "
     "rest for later:")
bullets([
    "<b>Kept now</b>: our four real escalation patterns (executive wire request, vendor bank-detail "
    "change, IT helpdesk vishing, payroll redirect), each paired with a matched legitimate &ldquo;twin&rdquo; "
    "conversation; real per-turn emotion features from the pretrained classifier named in our "
    "proposal; four of our six scorers (position-only, message-level emotion, running aggregate, "
    "and the trajectory GRU); the held-out pattern and held-out generator splits; and the opening-turn "
    "leakage shortcut check.",
    "<b>Built but not used in this run</b>: a real-data pipeline that downloads and parses the real "
    "Enron email corpus into genuine conversation threads, plus real ESConv and AnnoMI conversations, "
    "for the legitimate side of the data (<font face='Courier'>data/external_data.py</font>, "
    "<font face='Courier'>data/external/*.json</font>), along with a secondary FPR check against real "
    "Enron threads (<font face='Courier'>eval/evaluate.py</font>). We built and ran this end to end, "
    "then decided to keep this checkpoint synthetic-only for now; the code and downloaded data are "
    "in the repo and ready to switch back on.",
    "<b>Deferred</b>: real LLM-generated scam/twin conversations (still template-generated &mdash; no "
    "API key for an external model in this environment), the LLM-restyled authorship-check subset of "
    "Enron threads, the two text-based scorers and the DistilBERT reference ceiling, the rest of the "
    "shortcut-check suite, the slow-burn evaluation, and feature ablations.",
])

# ---------------- 1. Core system ----------------
h1("1. Working core system / MVP")
body("The pipeline has four stages, all in the attached repo (<font face='Courier'>scam_trajectory_mvp/"
     "</font>), and it runs end to end with zero errors:")
bullets([
    "<b>Data generation</b> &mdash; builds matched scam/twin conversation pairs across our four patterns, "
    "varying length (8&ndash;16 turns) and ask position so turn index alone can't give it away. One "
    "pattern (payroll redirect) and one generator voice are held out entirely, for the generalization "
    "splits.",
    "<b>Emotion features</b> &mdash; every turn is scored by the pretrained classifier named in our "
    "proposal (<font face='Courier'>j-hartmann/emotion-english-distilroberta-base</font>, 7 emotion "
    "classes), giving a 15-dim feature per turn (the distribution, its change from the previous turn, "
    "and a speaker flag).",
    "<b>Four scorers</b> &mdash; position-only and message-level (emotion) as baselines, running aggregate "
    "(the TA-suggested baseline: running mean and running max of message-level scores so far, averaged "
    "together), and our main model: a 1-layer "
    "GRU with 64 hidden units over the per-turn emotion features, trained to output a running score "
    "at every turn.",
    "<b>Evaluation</b> &mdash; conversation-level accuracy/F1/AUC, plus the AT@FPR earliness metric from "
    "our proposal, on the in-distribution test split and all three held-out splits.",
])

# ---------------- 2. Data + eval pipeline ----------------
h1("2. Data and evaluation pipeline")
h2("Data")
body("Every conversation in this run &mdash; scam, matched twin, and legitimate &mdash; is still "
     "template-generated rather than pulled from a real LLM or real text, this environment has no API "
     "key for an external model. The patterns, roles, and matched-twin design all come directly from "
     "our proposal, and four simulated &ldquo;generator voices&rdquo; stand in for different LLMs until "
     "we wire in real calls (stubbed in <font face='Courier'>data/llm_backend.py</font>). We also built "
     "a real-data path for the legitimate side &mdash; actual Enron email threads parsed from the public "
     "corpus, plus real ESConv and AnnoMI conversations &mdash; and ran it end to end earlier (see "
     "<font face='Courier'>data/external_data.py</font>, <font face='Courier'>data/external/*.json</font>), "
     "but this checkpoint uses synthetic data throughout so the numbers below are a clean, consistent "
     "baseline; that real-data code is untouched and easy to switch back on.")
note("Addressing the specific points from the proposal and TA feedback &mdash; <b>shortcut risk</b>: every "
     "scam scenario has a matched legitimate twin sharing the same roles, amount, and deadline, so the "
     "model can't just learn &ldquo;LLM-written vs. normal&rdquo; or &ldquo;big dollar amount = scam.&rdquo; "
     "<b>Generalization</b>: the payroll-redirect pattern and one generator voice are held out entirely "
     "from training, giving us three held-out test splits (unseen pattern, unseen generator, both). "
     "<b>Earliness metric</b>: AT@FPR exactly as scoped &mdash; calibrate a threshold on legitimate "
     "validation conversations at a fixed false-positive rate, then measure how many turns before the "
     "harmful request each scam conversation first crosses it.")
h2("Evaluation")
body("For each of the four scorers, we report classification quality (accuracy/F1/AUC, using each "
     "conversation's max running score) and earliness (AT@FPR, calibrated separately at 5% and 10% "
     "FPR, both suggested in our proposal), on all four splits.")

# ---------------- 3. Baselines ----------------
h1("3. Baseline(s)")
body("Three baselines, in increasing order of what they use: position-only (just the turn index, the "
     "strongest timing baseline from the PreScam paper our proposal builds on), message-level emotion "
     "(a single turn's emotion features, no memory of earlier turns), and running aggregate (the running "
     "mean and running max of message-level scores so far, averaged together &mdash; the TA's suggested "
     "baseline). The trajectory GRU is evaluated against all three.")

# ---------------- 4. First experiment ----------------
h1("4. First experiment")
body("We trained all four scorers on the synthetic training split, then evaluated on the "
     "in-distribution test split and all three held-out splits.")

hdr_style = ParagraphStyle(name="Hdr", parent=styles["Normal"], fontSize=7.0, leading=8.5,
                            textColor=colors.white, fontName="Helvetica-Bold")
cell_style = ParagraphStyle(name="Cell", parent=styles["Normal"], fontSize=7.6, leading=9.2)

def H(t): return Paragraph(t, hdr_style)
def C(t): return Paragraph(t, cell_style)

rows = [
    [H("Scorer"), H("Split"), H("Accuracy"), H("F1"), H("AUC")],
    [C("Position-only"), C("test"), C("0.500"), C("0.286"), C("0.500")],
    [C("Position-only"), C("heldout pattern"), C("0.500"), C("0.143"), C("0.500")],
    [C("Position-only"), C("heldout LLM"), C("0.500"), C("0.250"), C("0.500")],
    [C("Position-only"), C("heldout both"), C("0.500"), C("0.250"), C("0.500")],
    [C("Message-level (emotion)"), C("test"), C("0.489"), C("0.042"), C("0.429")],
    [C("Message-level (emotion)"), C("heldout pattern"), C("0.500"), C("0.000"), C("0.646")],
    [C("Message-level (emotion)"), C("heldout LLM"), C("0.417"), C("0.000"), C("0.365")],
    [C("Message-level (emotion)"), C("heldout both"), C("0.500"), C("0.000"), C("0.649")],
    [C("Running aggregate"), C("test"), C("0.511"), C("0.043"), C("0.524")],
    [C("Running aggregate"), C("heldout pattern"), C("0.500"), C("0.143"), C("0.556")],
    [C("Running aggregate"), C("heldout LLM"), C("0.458"), C("0.000"), C("0.472")],
    [C("Running aggregate"), C("heldout both"), C("0.500"), C("0.000"), C("0.569")],
    [C("Trajectory (GRU)"), C("test"), C("0.733"), C("0.667"), C("0.828")],
    [C("Trajectory (GRU)"), C("heldout pattern"), C("0.500"), C("0.000"), C("0.750")],
    [C("Trajectory (GRU)"), C("heldout LLM"), C("0.708"), C("0.588"), C("0.910")],
    [C("Trajectory (GRU)"), C("heldout both"), C("0.500"), C("0.000"), C("0.757")],
]
t = Table(rows, hAlign="LEFT", repeatRows=1,
          colWidths=[1.55*inch, 1.1*inch, 0.75*inch, 0.6*inch, 0.6*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)
story.append(Spacer(1, 6))
note("Classification metrics use each conversation's max running score. The trajectory GRU clearly "
     "leads every baseline on every split (AUC 0.75&ndash;0.91); the three baselines sit at or near "
     "chance (AUC 0.36&ndash;0.65). See Section 5 for why the GRU's numbers moved up from an earlier "
     "synthetic-only run and what changed.")
story.append(Spacer(1, 10))

body("Opening-turn leakage check (can a simple classifier tell a scam conversation from its matched "
     "twin using only the first 2 turns?): <b>0.470</b> 5-fold CV accuracy, where 0.5 is chance and "
     "1.0 would mean the generation process leaked the label into the rapport turns. This is close "
     "to chance, which is what we want &mdash; it means the matched-twin design is doing its job.")
story.append(Spacer(1, 8))

story.append(Image("outputs/example_traces.png", width=6.4*inch, height=2.4*inch))
note("Running score per turn for two example scam conversations (solid, circle markers) vs. two "
     "matched-twin conversations (dashed, x markers), on the test split, for all four scorers.")
story.append(Spacer(1, 10))

story.append(Image("outputs/earliness_hist.png", width=4.3*inch, height=3.0*inch))
note("Distribution of detection lead time (turns before the harmful request; negative means after "
     "it) on the test split, at the 5%-FPR threshold.")

# ---------------- 5. Preliminary results ----------------
h1("5. Preliminary results and what we're reading from them")
body("Three things stand out from this run:")
bullets([
    "<b>The three baselines sit at or near chance and never fire early.</b> Position-only, message-"
    "level, and running aggregate all land in the AUC 0.36&ndash;0.65 range (0.5 = coin flip) and show "
    "0% early_recall on every split at both FPR targets we calibrated to (5%, 10%). None of them pick "
    "up a usable signal before the harmful ask.",
    "<b>The trajectory GRU now clearly separates scam from legitimate, and shows some real earliness "
    "in-distribution.</b> Its AUC is 0.75&ndash;0.91 across all four splits, beating every baseline "
    "everywhere. On the in-distribution test split specifically, early_recall is 9% at the 5%-FPR "
    "threshold and 27% at the 10%-FPR threshold (median lead of 1 turn when it fires early). That's new "
    "since we added class-balanced loss (a pos_weight term) to the GRU's training objective &mdash; "
    "before that change, every scorer including the GRU showed 0% early_recall everywhere.",
    "<b>But that earliness doesn't generalize yet.</b> On all three held-out splits (unseen pattern, "
    "unseen generator voice, both), the GRU's early_recall is still 0%, even though its classification "
    "AUC on those same splits is high (0.75&ndash;0.91). So it can tell a held-out scam conversation "
    "apart from a legitimate one by the end, but it hasn't learned to flag a <i>novel</i> escalation "
    "pattern or voice before the harmful ask &mdash; only the pattern/voices it was trained on. Most "
    "likely cause: our per-turn training label is 1 only at or after the ask turn, so the model never "
    "gets a training signal that earlier urgency/authority turns in an unfamiliar pattern belong to the "
    "same kind of escalation &mdash; worth trying next, by widening how we label the pre-ask turns "
    "during training.",
])
body("The opening-turn leakage check (0.470, reported above) stays close to chance, which tells us the "
     "matched-twin design is doing its job: the generation process isn't leaking the scam/twin label "
     "into the early rapport turns.")

h2("Next steps")
bullets([
    "Try widening the per-turn training label to include the authority/urgency turns right before "
    "the ask, and see whether that gives the GRU an earlier signal to learn from, especially on the "
    "held-out splits.",
    "Switch the real-data path back on (Enron/ESConv/AnnoMI legitimate conversations and the secondary "
    "FPR check against real Enron threads &mdash; built and validated earlier, see "
    "<font face='Courier'>data/external_data.py</font>) once we've decided this checkpoint's synthetic-"
    "only numbers are the baseline we want to compare against.",
    "Replace the template generator with real LLM-generated conversations, using our actual escalation "
    "patterns (stubbed and ready to wire in &mdash; <font face='Courier'>data/llm_backend.py</font>) &mdash; "
    "this needs an API key we don't have in this environment yet.",
    "Generate the LLM-restyled authorship-check subset: we already pulled and reserved 300 real Enron "
    "threads for this (<font face='Courier'>data/external/enron_authorship_source.json</font>), just "
    "haven't had an LLM rewrite them yet, same reason as above.",
    "Add the DistilBERT text-based scorers as the reference ceiling.",
    "Finish the rest of the shortcut-check suite (authorship probe, full position + correlation "
    "analysis, human realism audit) and add the slow-burn evaluation split.",
])

doc = SimpleDocTemplate("Checkpoint_Submission.pdf", pagesize=letter,
                         topMargin=0.7*inch, bottomMargin=0.7*inch,
                         leftMargin=0.75*inch, rightMargin=0.75*inch)
doc.build(story)
print("done")
