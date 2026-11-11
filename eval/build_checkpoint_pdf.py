"""ReportLab checkpoint, driven exclusively by the completed rebuild artifacts."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",default="outputs/investigation")
    parser.add_argument("--out",default="output/pdf/Investigation_Iteration_Checkpoint_Rebuilt.pdf")
    args=parser.parse_args();root=Path(args.run)
    if not (root/"COMPLETE").exists():raise RuntimeError("Finish the experiment run before building the submission")
    load=lambda p:json.loads(p.read_text(encoding="utf-8"))
    manifest=load(root/"manifest.json");audit=load(root/"dataset_audit.json")
    dataset=load(root/"dataset.json")
    synthetic_n=sum(c["generator"]!="human" for c in dataset)
    metrics={}
    for p in root.glob("**/seed-*/metrics.json"):
        name=str(p.parent.parent.relative_to(root)).replace("\\","/")
        metrics.setdefault(name,[]).append(load(p))
    def value(name,split,key,target="0.05"):
        section,field=key
        return mean(m[target][split][section][field] for m in metrics[name])
    auc=lambda name,split="test":f"{value(name,split,('classification','auc')):.3f}"
    early=lambda name,split="test",target="0.05":f"{value(name,split,('earliness','early_recall'),target):.1%}"
    fpr=lambda name,split="test",target="0.05":f"{value(name,split,('false_positives','fpr'),target):.1%}"
    findings=[
        f"The emotion GRU achieves mean standard-test AUC {auc('synthetic/gru')} and early recall {early('synthetic/gru')} at the target 5% operating point, with observed FPR {fpr('synthetic/gru')}. This run does not demonstrate useful early warning on the standard revised benchmark. Its performance should not be equated with the corrected original-data rerun because the dataset, pairing constraints and training set changed together.",
        f"On the long/delayed-ask test, GRU early recall rises to {early('synthetic/gru','test_slow')}, but legitimate-twin FPR rises to {fpr('synthetic/gru','test_slow')}. This is a calibration/timing-transfer failure, not a robustness success. Adding real negatives gives standard-test AUC {auc('mixed/gru')} and long-test FPR {fpr('mixed/gru','test_slow')}; it does not establish a fix for early detection.",
        f"Removing deltas changes standard AUC to {auc('ablation/no-delta')}; removing the speaker flag gives {auc('ablation/no-speaker')}. Shuffled prefixes give AUC {auc('investigation/shuffled-prefix')}, while soft pre-ask labels give {auc('investigation/preask-label')} and early recall {early('investigation/preask-label')}. Small classification differences do not rescue the absent early-warning benefit. Seed-0 paired intervals below quantify conditional differences, not causal proof of a real-world emotion signal.",
        f"TF-IDF prefixes achieve standard AUC {auc('synthetic/tfidf-prefix')} and early recall {early('synthetic/tfidf-prefix')}, but FPR on the held-out pattern is {fpr('synthetic/tfidf-prefix','test_heldout_pattern')}. High transcript AUC and badly shifted alert thresholds can coexist. The template benchmark remains vulnerable to surface shortcuts; text-model superiority here does not establish realistic manipulation detection."]
    (root/"FINDINGS.md").write_text("# Interpretation of the rebuilt results\n\n"+"\n\n".join(findings)+"\n",encoding="utf-8")
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyTextReport",fontName="Helvetica",fontSize=9.3,leading=12.4,spaceAfter=7))
    styles.add(ParagraphStyle(name="SmallReport",fontName="Helvetica",fontSize=8,leading=10,spaceAfter=5))
    for name in ("Title","Heading1","Heading2"):styles[name].textColor=colors.black
    styles["Title"].fontSize=19;styles["Title"].leading=23
    styles["Heading1"].fontSize=14;styles["Heading1"].leading=17
    styles["Heading2"].fontSize=11.5;styles["Heading2"].leading=14
    story=[]
    def p(text,style="BodyTextReport"):story.append(Paragraph(text,styles[style]))
    def heading(text):p(text,"Heading2")
    def table(headers,rows,widths):
        data=[[Paragraph(escape(str(x)),styles["SmallReport"]) for x in row] for row in [headers]+rows]
        t=Table(data,colWidths=[w*inch for w in widths],repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8edf1")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#b9c1c8")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        story.extend([t,Spacer(1,8)])
    p("Emotional Trajectory Detection of Social Engineering","Title")
    p("Investigation and Iteration Checkpoint | November 11, 2026","Heading2")
    p("Team 6: Rahul Ajith Kumar, Sanchusri Kavitha Babu, Sisvanthkumar Sathivadivel","SmallReport")
    p("Rebuilt from the original MVP on branch <b>rahul/investigation-rebuild</b>. This report replaces unsupported historical claims with newly saved experiments; it does not assume the earlier reported numbers are correct.")
    heading("Question and experimental setup")
    p("Can an ordered sequence of emotion features detect a harmful request earlier than an isolated-message detector at a comparable false-alarm rate? We also test transfer to unseen template patterns/styles, longer delayed-ask conversations, and real legitimate text.")
    p(f"The revised benchmark has <b>{synthetic_n}</b> synthetic conversations, including matched scam/legitimate twins with identical opening turns, length and alternating speakers. Scenario identities are disjoint across training, validation and held-out tests; delayed-ask variants reuse only standard-test scenarios. Four generator labels are text-substitution styles, not actual LLM outputs.")
    p("Frozen emotion distributions (7), changes (7), and speaker flag (1) feed a 1-layer, 64-unit GRU. The isolated-message baseline excludes changes from the previous turn. Neural models use three seeds and 30 fixed epochs. Thresholds are calibrated only on legitimate synthetic validation twins; test labels never enter training or calibration.")
    heading("Corrections to the MVP")
    p("We corrected threshold ties/off-by-one behavior, removed final-length leakage from the position baseline, fitted the opening probe vocabulary within grouped folds, and removed hard-coded findings. The corrected original-data rerun is preserved separately under outputs/corrected-mvp; the revised-data investigation is not a like-for-like replication of October's dataset.")
    heading("Standard test results")
    names=[("Position","synthetic/position"),("Message","synthetic/message"),("Aggregate","synthetic/aggregate"),("GRU","synthetic/gru")]
    table(["Scorer","AUC","Early @1%","Early @5%","Early @10%","FPR @5%"],
        [[label,auc(n),early(n,target="0.01"),early(n),early(n,target="0.1"),fpr(n)] for label,n in names],
        [1.05,.7,1.05,1.05,1.1,1.05])
    p("@ percentages identify calibration targets. FPR is the observed share of legitimate test conversations alerted at any turn; it need not equal the target. Table entries are means over available seeds. AUC measures classification over the whole conversation, not earliness.","SmallReport")
    heading("GRU generalization")
    table(["Test split","AUC","Early @5%","Early @10%","FPR @5%"],
        [[s.replace("test_",""),auc("synthetic/gru",s),early("synthetic/gru",s),early("synthetic/gru",s,"0.1"),fpr("synthetic/gru",s)] for s in ["test_heldout_pattern","test_heldout_style","test_heldout_both","test_slow"]],
        [2.05,.8,1.05,1.05,1.05])
    story.append(PageBreak())
    p("Ablations and timing investigation","Heading1")
    p("We retrained each GRU feature subset on the same training data for seeds 0, 1 and 2. The comparison below reports both classification and earliness; a higher AUC alone does not establish better early warning.")
    variants=[("Full","synthetic/gru"),("No emotion delta","ablation/no-delta"),("No speaker","ablation/no-speaker"),("Emotion only","ablation/emotion-only"),("Delta + speaker","ablation/delta-speaker"),("Shuffled prefix","investigation/shuffled-prefix"),("Soft pre-ask labels","investigation/preask-label")]
    table(["Variant","Test AUC","Unseen-pattern AUC","Slow AUC","Test early @5%"],
        [[label,auc(n),auc(n,"test_heldout_pattern"),auc(n,"test_slow"),early(n)] for label,n in variants],
        [1.6,.85,1.3,.9,1.35])
    p("The shuffled-prefix control retains the current turn, permutes only already observed history, and recomputes deltas. Its training examples are shuffled prefixes with last-turn loss. This checks an ordering hypothesis without showing the model future test turns. Soft pre-ask labels mark only training urgency/authority turns as 0.5; ask annotations and held-out-pattern exclusion are unchanged.")
    boot=load(root/"paired_bootstrap.json")
    rows=[]
    for name in ("ablation/no-delta","investigation/shuffled-prefix","investigation/preask-label"):
        r=boot[name]["test"]
        fmt=lambda x:f"[{x[0]:+.3f}, {x[1]:+.3f}]"
        rows.append([name.split("/")[-1],f"{r['auc_difference']:+.3f}",fmt(r['auc_difference_ci95']),f"{r['early_recall_difference']:+.3f}",fmt(r['early_recall_difference_ci95'])])
    heading("Paired uncertainty on the standard test")
    table(["Candidate","AUC delta","95% interval","Early delta","95% interval"],rows,[1.35,.8,1.4,.9,1.55])
    p("Differences are candidate minus full GRU, conditional on seed-0 models and fixed validation thresholds. Intervals use 400 matched-scenario bootstrap samples; training and calibration uncertainty are not included. Multi-seed ranges are recorded in REPORT.md. These exploratory comparisons have no multiplicity adjustment.","SmallReport")
    story.append(Image(str(root/"lead_and_trace.png"),width=6.0*inch,height=2.18*inch))
    p("Positive lead means an alert before the ask. Never-alerted scams are counted in the legend rather than silently removed. The right panel shows one matched test pair, not an aggregate performance claim.","SmallReport")
    story.append(PageBreak())
    p("Real negatives and shortcut diagnostics","Heading1")
    p("We compared synthetic-only training with training that adds real presumed-legitimate Enron, ESConv and AnnoMI examples. Of 483 real train/validation examples, 410 enter training and 73 are diagnostic validation only. Another 200 Enron conversations are excluded from all training and calibration. Calibration remains on synthetic twins in both conditions.")
    table(["Training / scorer","Enron FPR @5%","Enron FPR @10%"],
        [[f"{cond} / {label}",fpr(f"{cond}/{name}","secondary_enron"),fpr(f"{cond}/{name}","secondary_enron","0.1")] for cond in ("synthetic","mixed") for label,name in [("position","position"),("message","message"),("aggregate","aggregate"),("GRU","gru")]],
        [3.0,1.5,1.5])
    p("Wilson intervals and per-model counts are saved in metrics.json. Real transcripts are capped at their first 64 observed turns for CPU training; the number clipped is recorded in dataset_audit.json. Exact duplicate transcripts were filtered before splitting, but related-thread and semantic overlap are not ruled out.","SmallReport")
    heading("Surface-style and length probes")
    probes=load(root/"shortcut_probes.json")
    rows=[]
    for name in ("length_synthetic","length_mixed","style_synthetic","style_mixed"):
        rows.append([name,f"{probes[name]['synthetic_test']['auc']:.3f}",f"{probes[name]['synthetic_scams_vs_enron']['auc']:.3f}"])
    table(["Probe / training","Synthetic-test AUC","Scams vs Enron AUC"],rows,[2.2,1.9,1.9])
    p(f"A separate legitimate-source probe gets AUC {probes['legitimate_source_probe']['auc']:.3f} on held-out validation examples. Human/domain versus template/domain separability warns about confounding; it does not prove that the GRU relies on authorship. Full-transcript length features are used only for offline probes.")
    heading("Text reference models")
    rows=[]
    for name in ("tfidf-message","tfidf-prefix","distilbert-frozen-message","distilbert-frozen-prefix"):
        n=f"synthetic/{name}"
        rows.append([name,auc(n),auc(n,"test_heldout_pattern"),early(n)])
    table(["Reference","Test AUC","Unseen-pattern AUC","Test early @5%"],rows,[2.5,.9,1.35,1.25])
    p("DistilBERT is a frozen token-mean encoder followed by logistic regression; the prefix variant averages observed turn embeddings. It is not the proposed fine-tuned transcript ceiling. TF-IDF vocabulary is fitted on training text only. Finite template wording remains a major shortcut risk, even for held-out styles.","SmallReport")
    story.append(PageBreak())
    p("Failure analysis and next iteration","Heading1")
    errors=load(root/"error_inventory.json");counts=Counter((r["split"],r["kind"]) for r in errors)
    p("We saved every seed-0 GRU false alarm and late/missed scam with its ID, score and alert/ask position. This inventory supports manual review; it is not a completed human explanation of the errors.")
    table(["Evaluation set","False alarms","Late / missed scams"],
        [[s,counts[s,"false_positive"],counts[s,"late_or_missed"]] for s in ["test","test_heldout_pattern","test_heldout_style","test_heldout_both","test_slow","secondary_enron"]],
        [3.0,1.5,1.5])
    heading("What changed and what failed")
    p(findings[0]);p(findings[1])
    heading("Limitations and remaining checks")
    p("The long test adds neutral setup and delays the ask to 70-95% of 20-32 turns. It is a timing/length stress test, not validated low-intensity or adaptive evasion. There is no independent real-scam test set, no real-LLM generated conversation set, and no LLM-restyled Enron experiment in this run.")
    p("The original Enron preparation script is unavailable; provided JSON labels/threading have not been independently verified. Exact deduplication cannot exclude related-thread overlap. Small held-out and calibration samples limit precision, especially at a 1% target.")
    p("A 150-example human audit worksheet is saved as human_audit_PENDING.csv. Realism ratings, independent ask verification, inter-rater agreement and manual failure-cluster analysis are still pending. We do not claim that calmly worded templates are realistic because of their tone alone.")
    heading("Next experiments")
    p("Complete the human audit and inspect false alarms and late/missed scams. Prioritize removing surface-style shortcuts and testing length-matched validation calibration on fresh delayed-ask scenarios. Add new paraphrases, scenario families and independent real scam transcripts if available; a fine-tuned DistilBERT ceiling remains separate work.")
    heading("Reproduction and evidence")
    p("Run python run_mvp.py for the corrected original MVP and python run_experiments.py --mode investigation for this checkpoint. Each completed run saves dataset/configuration and hashes, pretrained model revisions, dependency versions, per-turn predictions, model weights, metrics at 1%/5%/10% targets, error inventory and plots. Completed outputs are protected against accidental overwriting.")
    p("Evidence: outputs/investigation/REPORT.md, summary.csv, paired_bootstrap.json, shortcut_probes.json, manifest.json and experiment subdirectories. The historical 40%/56% Enron rates, 0.78/0.48 slow-burn ablation and 1.000 DistilBERT score are not reused as findings.","SmallReport")
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(.7*inch,.42*inch,"Team 6 | Investigation and Iteration | Rebuilt results")
        canvas.drawRightString(7.8*inch,.42*inch,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=letter,leftMargin=.7*inch,rightMargin=.7*inch,
        topMargin=.55*inch,bottomMargin=.65*inch,title="Investigation and Iteration Checkpoint",author="Team 6").build(story,onFirstPage=footer,onLaterPages=footer)
    provenance=dict(builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        metrics_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("**/seed-*/metrics.json")},
        pdf_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(output.resolve())


if __name__=="__main__":main()
