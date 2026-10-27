# Evaluation, following the proposal's "Evaluation Pipeline" section:
# AT@FPR earliness (alert turns before t_ask, at a fixed false-positive
# rate on legitimate conversations), reported as early recall / median lead
# / late-or-never rate / mean lead as a fraction of pre-ask turns. Plus one
# of the shortcut checks: opening-turn leakage.
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def conversation_running_scores(conversations, turn_feats_by_id, running_score_fn):
    # running_score_fn(conversation, its turn features) -> per-turn running score array
    return {c.conv_id: running_score_fn(c, turn_feats_by_id[c.conv_id]) for c in conversations}


def calibrate_threshold(legit_conversations, running_scores_by_id, target_fpr=0.05):
    # smallest threshold so at most target_fpr of these legitimate conversations
    # ever cross it (i.e. max running score >= threshold)
    max_scores = np.array([running_scores_by_id[c.conv_id].max() for c in legit_conversations])
    if len(max_scores) == 0:
        return 0.5
    k = int(np.ceil((1 - target_fpr) * len(max_scores))) - 1
    k = min(max(k, 0), len(max_scores) - 1)
    return float(np.sort(max_scores)[k])


def conversation_metrics(conversations, running_scores_by_id, threshold=0.5):
    y_true = [c.label for c in conversations]
    scores = [running_scores_by_id[c.conv_id].max() for c in conversations]
    y_pred = [int(s >= threshold) for s in scores]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(set(y_true)) > 1:
        out["auc"] = roc_auc_score(y_true, scores)
    return out


def at_fpr(scam_conversations, threshold, running_scores_by_id):
    # AT@FPR_alpha = t_ask - min{t : s_t >= threshold}, per scam conversation
    leads = []
    for c in scam_conversations:
        scores = running_scores_by_id[c.conv_id]
        crossed = np.where(scores >= threshold)[0]
        if len(crossed) == 0:
            leads.append(None)
        else:
            leads.append(c.harmful_turn_idx - int(crossed[0]))

    early = [l for l in leads if l is not None and l > 0]
    late_or_never = sum(1 for l in leads if l is None or l <= 0)
    lead_fractions = [leads[i] / max(c.harmful_turn_idx, 1)
                       for i, c in enumerate(scam_conversations) if leads[i] is not None and leads[i] > 0]
    return {
        "threshold": threshold,
        "early_recall": len(early) / len(scam_conversations) if scam_conversations else 0.0,
        "median_lead_turns": float(np.median(early)) if early else None,
        "late_or_never_rate": late_or_never / len(scam_conversations) if scam_conversations else 0.0,
        "mean_lead_fraction": float(np.mean(lead_fractions)) if lead_fractions else None,
        "per_conversation_leads": leads,
    }


SPLITS = ["test", "test_heldout_pattern", "test_heldout_llm", "test_heldout_both"]


def full_report(conversations_by_split, turn_feats_by_id, running_score_fn, target_fpr=0.05):
    report = {}
    all_convs = [c for split in conversations_by_split.values() for c in split]
    running_scores_by_id = conversation_running_scores(all_convs, turn_feats_by_id, running_score_fn)

    val_legit = [c for c in conversations_by_split["val"] if c.label == 0]
    threshold = calibrate_threshold(val_legit, running_scores_by_id, target_fpr)

    for split_name in SPLITS:
        convs = conversations_by_split[split_name]
        scams = [c for c in convs if c.label == 1]
        earliness = at_fpr(scams, threshold, running_scores_by_id)
        earliness["target_fpr"] = target_fpr
        report[split_name] = {
            "n": len(convs),
            "classification": conversation_metrics(convs, running_scores_by_id, threshold=threshold),
            "earliness": earliness,
        }
    return report, running_scores_by_id


def secondary_fpr_check(real_legit_conversations, threshold, running_scores_by_id):
    """Secondary FPR check (proposal's term for the real-Enron-threads role):
    at the threshold already calibrated on our synthetic validation twins,
    what fraction of real human business-email threads (never used in
    training or calibration) still trigger a false alarm? Answers whether
    the threshold generalizes to real text, not just our synthetic style."""
    if not real_legit_conversations:
        return {"n": 0, "fpr": None}
    scores = [running_scores_by_id[c.conv_id].max() for c in real_legit_conversations]
    fpr = sum(1 for s in scores if s >= threshold) / len(scores)
    return {"n": len(real_legit_conversations), "fpr": fpr}


def opening_leakage_check(conversations, n_turns=2, cv=5):
    """Shortcut check: can a TF-IDF classifier tell scam from matched twin
    using only the first n_turns? If it scores well above chance, generation
    leaked the label into the opening (rapport) turns, which should look the
    same for a scam and its twin."""
    texts = [" ".join(c.turns[:n_turns]) for c in conversations]
    labels = [c.label for c in conversations]
    if len(set(labels)) < 2:
        return {"cv_accuracy": None}
    X = TfidfVectorizer(max_features=500, ngram_range=(1, 2)).fit_transform(texts)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, labels, cv=min(cv, min(np.bincount(labels))))
    return {"cv_accuracy": float(scores.mean())}
