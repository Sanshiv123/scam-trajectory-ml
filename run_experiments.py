"""Reproducible checkpoint experiments; every reported score has saved predictions."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data.generate_data import build_dataset as legacy_dataset
from data.investigation_data import build_dataset, load_real, validate
from eval.evaluate import full_report, opening_leakage_check
from features.cache import encode
from features.emotion_features import turn_features
from models.baseline import PositionScorer, MessageLevelEmotionScorer, position_features, running_aggregate
from models.trajectory_model import TrajectoryModel

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def labels(c, early=False):
    y = np.array([int(c.label and t >= c.harmful_turn_idx) for t in range(len(c.turns))], dtype=float)
    if early and c.label:
        for i, phase in enumerate(c.phases):
            if i < c.harmful_turn_idx and phase in ("urgency", "authority"): y[i] = .5
    return y


def features(data, kind="emotion"):
    texts = [t for c in data for t in c.turns]
    vectors, info = encode(texts, kind)
    result, offset = {}, 0
    for c in data:
        raw = vectors[offset:offset+len(c.turns)]; offset += len(c.turns)
        result[c.conv_id] = turn_features(raw, c.speakers) if kind == "emotion" else raw
    return result, info


def shuffle_prefix(f, t, seed):
    # Latest turn stays fixed. Permute only already observed past turns;
    # recompute deltas after shuffling, so features cannot reveal old order.
    rng = np.random.default_rng(seed + 7919*t)
    idx = np.r_[rng.permutation(t), t]
    raw = f[:t+1][idx]
    return turn_features(raw[:, :7], raw[:, 14])


def fit_gru(train, feats, seed, epochs, columns=None, early=False, shuffled=False):
    seqs, ys = [], []
    cols = list(range(15)) if columns is None else columns
    for c in train:
        f, y = feats[c.conv_id], labels(c, early)
        if shuffled:
            for t in range(len(f)):
                seqs.append(shuffle_prefix(f, t, seed)[:, cols]); ys.append(np.array([0.]*t+[y[t]]))
        else:
            seqs.append(f[:, cols]); ys.append(y)
    model = TrajectoryModel(input_dim=len(cols), epochs=epochs, seed=seed).fit(seqs, ys, last_only=shuffled)
    return model


def score_gru(model, data, feats, cols=None, shuffled=False, seed=0):
    cols = list(range(15)) if cols is None else cols
    if not shuffled:
        return dict(zip([c.conv_id for c in data], model.running_scores([feats[c.conv_id][:, cols] for c in data])))
    result = {}
    for c in data:
        f = feats[c.conv_id]
        seqs = [shuffle_prefix(f, t, seed)[:, cols] for t in range(len(f))]
        result[c.conv_id] = np.array([s[-1] for s in model.running_scores(seqs)])
    return result


def fit_text(train, data, kind, embeddings=None):
    y = np.concatenate([labels(c) for c in train])
    if embeddings is None:
        # Prefix strings are causally constructed. Fit vocabulary on train only.
        def text(c):
            return c.turns if kind == "message" else [" ".join(c.turns[:t+1]) for t in range(len(c.turns))]
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=6000),
                              LogisticRegression(class_weight="balanced", max_iter=1000))
        model.fit([s for c in train for s in text(c)], y)
        scores = {c.conv_id: model.predict_proba(text(c))[:, 1] for c in data}
    else:
        def x(c):
            f = embeddings[c.conv_id]
            return f if kind == "message" else np.cumsum(f, axis=0) / np.arange(1,len(f)+1)[:,None]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight="balanced", max_iter=1500))
        model.fit(np.concatenate([x(c) for c in train]), y)
        scores = {c.conv_id: model.predict_proba(x(c))[:, 1] for c in data}
    return model, scores


def style_vector(c, length_only=False):
    import re
    text = " ".join(c.turns)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    if length_only: return [len(c.turns), len(words), len(text)]
    n = max(1,len(words)); chars=max(1,len(text))
    function = {"the","a","an","and","or","but","i","you","we","it","of","to","in","on","for","with"}
    return [np.mean([len(w) for w in words]) if words else 0, len(words)/sentences,
            sum(w in function for w in words)/n, text.count("!")/chars,
            text.count("?")/chars, text.count(",")/chars, sum(ch.isupper() for ch in text)/chars,
            sum(ch.isdigit() for ch in text)/chars]


def shortcut_probes(synthetic, real, holdout):
    train = [c for c in synthetic if c.split == "train"]
    test = [c for c in synthetic if c.split == "test"]
    out = {}
    for length_only in (True, False):
        name = "length" if length_only else "style"
        for mix in (False, True):
            tr = train + ([c for c in real if c.split == "train"] if mix else [])
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
            clf.fit([style_vector(c,length_only) for c in tr],[c.label for c in tr])
            evaluations = {"synthetic_test": test,
                           "synthetic_scams_vs_enron": [c for c in test if c.label]+holdout}
            out[f"{name}_{'mixed' if mix else 'synthetic'}"] = {}
            for s,cs in evaluations.items():
                probs=clf.predict_proba([style_vector(c,length_only) for c in cs])[:,1]
                out[f"{name}_{'mixed' if mix else 'synthetic'}"][s]=dict(
                    auc=float(roc_auc_score([c.label for c in cs],probs)),n=len(cs),
                    predictions=[dict(id=c.conv_id,label=c.label,score=float(p)) for c,p in zip(cs,probs)])
    # A source probe on legitimate text, separate from scam detection.
    tr = [c for c in synthetic+real if c.label == 0 and c.split == "train"]
    te = [c for c in synthetic+real if c.label == 0 and c.split in ("val","val_real")]
    clf = make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=1000))
    clf.fit([style_vector(c) for c in tr],[int(c.generator=="human") for c in tr])
    probs=clf.predict_proba([style_vector(c) for c in te])[:,1]
    out["legitimate_source_probe"] = dict(auc=float(roc_auc_score([int(c.generator=="human") for c in te],probs)),n=len(te),
        predictions=[dict(id=c.conv_id,label=int(c.generator=="human"),score=float(p)) for c,p in zip(te,probs)],
        caveat="Separates human/domain from template/domain; does not prove detector reliance on authorship.")
    return out


def save_experiment(out, name, seed, data, feats, scores, weights=None, fitted=None):
    folder = out / name / f"seed-{seed}"; folder.mkdir(parents=True,exist_ok=True)
    by = {s:[c for c in data if c.split==s] for s in sorted({c.split for c in data})}
    # val_real is reported as a diagnostic only, never mixed into calibration.
    reports = {}
    for alpha in (.01,.05,.10):
        report, _ = full_report(by,feats,lambda c,f:scores[c.conv_id],alpha)
        reports[str(alpha)] = report
    dump(folder / "metrics.json", reports)
    np.savez_compressed(folder / "predictions.npz", **{str(k):v for k,v in scores.items()})
    if weights is not None: torch.save(weights,folder / "model.pt")
    if fitted is not None:
        import joblib
        joblib.dump(fitted,folder / "model.joblib")
    # No raw external email text copied into the human-readable report.
    with (folder / "conversations.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["id","pair","scenario","split","label","ask","max_score","alert","lead"]);writer.writeheader()
        th=next(iter(reports["0.05"].values()))["earliness"]["threshold"]
        for c in data:
            crossed=np.flatnonzero(scores[c.conv_id]>=th)
            alert=int(crossed[0]) if len(crossed) else None
            writer.writerow(dict(id=c.conv_id,pair=c.pair_id or f"legacy-{c.conv_id//2}",scenario=c.scenario_id,split=c.split,label=c.label,ask=c.harmful_turn_idx,
                max_score=float(scores[c.conv_id].max()),alert=alert,lead=c.harmful_turn_idx-alert if c.label and alert is not None else None))
    print(f"saved {name} seed {seed}",flush=True)
    return reports


def write_manifest(out,args,data,models):
    files = [p for p in ROOT.rglob("*.py") if not any(part in (".venv",".git",".cache","outputs","output","tmp") for part in p.relative_to(ROOT).parts)]
    import zipfile
    with zipfile.ZipFile(out/"source.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for p in files: archive.write(p,str(p.relative_to(ROOT)))
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),config=vars(args),
        base="67cddb6",git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        git_branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
        python=sys.version,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        external_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"data/external").glob("*.json")},
        dataset_hash=hashlib.sha256(json.dumps([asdict(c) for c in data],sort_keys=True).encode()).hexdigest(),models=models,
        calibration="Synthetic legitimate validation only; conservative observed FPR, no population guarantee",
        limitations=["Controlled templates, no real scam test set", "No completed human audit or LLM restyling", "Frozen DistilBERT references are not fine-tuned models"])
    dump(out/"manifest.json",manifest)
    lock=subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
    (out/"requirements-lock.txt").write_text(lock,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["corrected-mvp","investigation"],default="corrected-mvp")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--skip-text",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    if 0 not in args.seeds or args.epochs<1 or args.threads<1:
        parser.error("Include reference seed 0 and use positive epochs/threads")
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    out=ROOT/"outputs"/args.mode;out.mkdir(parents=True,exist_ok=True)
    if (out/"COMPLETE").exists(): raise RuntimeError("Completed run exists; archive it before rerunning to avoid overwriting evidence")
    synthetic=legacy_dataset() if args.mode=="corrected-mvp" else build_dataset()
    if args.mode=="corrected-mvp":
        for c in synthetic: c.pair_id=f"legacy-{c.conv_id//2}"
        real,holdout,audit=[],[],{}
    else:
        real,holdout,audit=load_real()
        # Limit real inputs to observed first 64 turns for bounded CPU training.
        audit["prefix_limit"] = dict(max_turns=64, clipped=sum(len(c.turns)>64 for c in real+holdout))
        for c in real+holdout:
            c.turns=c.turns[:64];c.speakers=c.speakers[:64]
    data=synthetic+real+holdout
    summary=validate(data)
    dump(out/"dataset.json",[asdict(c) for c in data]);dump(out/"dataset_audit.json",dict(validation=summary,real=audit))
    feats,emotion_info=features(data)
    np.savez_compressed(out/"frozen_emotion_features.npz",**{str(k):v for k,v in feats.items()})
    embeddings,text_info=(None,{}) if args.skip_text or args.mode=="corrected-mvp" else features(data,"text")
    write_manifest(out,args,data,dict(emotion=emotion_info,distilbert=text_info))
    if args.prepare_only: return
    train=[c for c in synthetic if c.split=="train"]
    leak=opening_leakage_check([c for c in synthetic if c.split in ("train","val")])
    dump(out/"opening_probe.json",leak)
    byid={c.conv_id:c for c in synthetic}
    diagnostics={}
    for split in sorted({c.split for c in synthetic}):
        diffs=[]
        for c in synthetic:
            if c.label and c.split==split:
                twin=byid[c.conv_id+1]
                # Direct pairwise separation in frozen emotions before the ask.
                a=feats[c.conv_id][:c.harmful_turn_idx,:7]
                b=feats[twin.conv_id][:c.harmful_turn_idx,:7]
                diffs.append(float(np.abs(a-b).sum(1).mean()))
        diagnostics[split]=dict(n_pairs=len(diffs),mean_preask_emotion_L1=float(np.mean(diffs)),
            median_preask_emotion_L1=float(np.median(diffs)))
    dump(out/"preask_emotion_diagnostics.json",diagnostics)
    if real: dump(out/"shortcut_probes.json",shortcut_probes(synthetic,real,holdout))
    for condition in (["synthetic"] if args.mode=="corrected-mvp" else ["synthetic","mixed"]):
        tr=train+([c for c in real if c.split=="train"] if condition=="mixed" else [])
        y=np.concatenate([labels(c) for c in tr])
        pm=PositionScorer().fit(np.concatenate([position_features(len(c.turns)) for c in tr]),y)
        ps={c.conv_id:pm.predict_proba(position_features(len(c.turns))) for c in data}
        save_experiment(out,f"{condition}/position",0,data,feats,ps,fitted=pm)
        # The true isolated-message baseline excludes the previous-turn delta.
        msg_cols=list(range(7))+[14]
        mm=MessageLevelEmotionScorer().fit(np.concatenate([feats[c.conv_id][:,msg_cols] for c in tr]),y)
        ms={c.conv_id:mm.predict_proba(feats[c.conv_id][:,msg_cols]) for c in data}
        save_experiment(out,f"{condition}/message",0,data,feats,ms,fitted=mm)
        save_experiment(out,f"{condition}/aggregate",0,data,feats,{k:running_aggregate(v) for k,v in ms.items()},fitted=mm)
        for seed in args.seeds:
            model=fit_gru(tr,feats,seed,args.epochs)
            save_experiment(out,f"{condition}/gru",seed,data,feats,score_gru(model,data,feats),model.net.state_dict())
        if args.mode=="investigation":
            for kind in ("message","prefix"):
                model,scores=fit_text(tr,data,kind)
                import joblib
                folder=out/f"{condition}/tfidf-{kind}";folder.mkdir(parents=True,exist_ok=True)
                joblib.dump(model,folder/"model.joblib")
                save_experiment(out,f"{condition}/tfidf-{kind}",0,data,feats,scores)
                if embeddings is not None:
                    model,scores=fit_text(tr,data,kind,embeddings)
                    folder=out/f"{condition}/distilbert-frozen-{kind}";folder.mkdir(parents=True,exist_ok=True)
                    joblib.dump(model,folder/"model.joblib")
                    save_experiment(out,f"{condition}/distilbert-frozen-{kind}",0,data,feats,scores)
    if args.mode=="investigation":
        subsets={"no-delta":list(range(7))+[14],"no-speaker":list(range(14)),
                 "emotion-only":list(range(7)),"delta-speaker":list(range(7,15))}
        for name,cols in dict(full=list(range(15)),**subsets).items():
            # Delta-bearing logistic variants have adjacent-turn context;
            # they are not the isolated-message baseline in the main comparison.
            model=MessageLevelEmotionScorer().fit(np.concatenate([feats[c.conv_id][:,cols] for c in train]),np.concatenate([labels(c) for c in train]))
            scores={c.conv_id:model.predict_proba(feats[c.conv_id][:,cols]) for c in data}
            save_experiment(out,f"ablation/logistic-{name}",0,data,feats,scores,fitted=model)
        for seed in args.seeds:
            for name,cols in subsets.items():
                model=fit_gru(train,feats,seed,args.epochs,columns=cols)
                save_experiment(out,f"ablation/{name}",seed,data,feats,score_gru(model,data,feats,cols),model.net.state_dict())
            for name,options in [("preask-label",dict(early=True)),("shuffled-prefix",dict(shuffled=True))]:
                # Only synthetic training patterns; held-out labels never enter fit.
                model=fit_gru(train,feats,seed,args.epochs,**options)
                scores=score_gru(model,data,feats,shuffled=options.get("shuffled",False),seed=seed)
                save_experiment(out,f"investigation/{name}",seed,data,feats,scores,model.net.state_dict())
    from eval.summarize import summarize
    summarize(out)
    (out/"COMPLETE").write_text(datetime.now(timezone.utc).isoformat(),encoding="utf-8")
    print(f"COMPLETE: {out}",flush=True)


if __name__=="__main__": main()
