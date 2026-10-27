# Loads the three real external datasets from our proposal's data table and
# turns them into Conversation objects (data/generate_data.py), so they can
# sit alongside the synthetic scam/twin pairs in the same pipeline:
#
#   Enron threads, 4+ messages, fraud-filtered   -- legitimate, human,
#     train + secondary FPR check (does the calibrated threshold still
#     hold its false-positive rate on real business email, not just our
#     synthetic matched twins?)
#   ESConv / AnnoMI                               -- legitimate, human,
#     train diversity / emotional-language negatives (real emotionally-
#     charged conversations that are NOT scams, so the model can't learn
#     "any strong emotion = manipulative")
#
# The Enron threads were pulled from the real Enron email corpus
# (cs.cmu.edu/~enron) and reconstructed into actual back-and-forth threads
# by parsing RFC822 headers and grouping by normalized subject + shared
# participants (see the data-prep script used to build data/external/
# *.json -- not included here since it's a one-time ~3GB download/parse,
# not something run_mvp.py re-does on every run). ESConv and AnnoMI are
# pulled as-is from their public HuggingFace releases.
#
# NOT yet done: the proposal's "~300 Enron threads restyled by the same
# LLMs" authorship-shortcut subset. That needs an LLM to actually rewrite
# the 300 source threads in data/external/enron_authorship_source.json --
# we have the source threads ready, but haven't generated the restyled
# versions yet (see README).
import json
import os
from data.generate_data import Conversation

_DIR = os.path.join(os.path.dirname(__file__), "external")


def _speaker_sequence(turns, first_speaker_key):
    # binary encoding consistent with the synthetic data: whoever sent the
    # first turn is speaker 0, everyone else is speaker 1
    seq = []
    for t in turns:
        seq.append(0 if t["speaker"] == first_speaker_key else 1)
    return seq


def load_enron(path=None, label_split="train"):
    path = path or os.path.join(_DIR, "enron_main.json")
    with open(path) as f:
        threads = json.load(f)
    convs = []
    for i, t in enumerate(threads):
        turns = [turn["text"] for turn in t["turns"]]
        first_sender = t["turns"][0]["speaker"]
        speakers = _speaker_sequence(t["turns"], first_sender)
        convs.append(Conversation(
            conv_id=None, turns=turns, speakers=speakers, label=0,
            pattern="enron_real", generator="human", harmful_turn_idx=None,
            split=label_split, is_twin=False,
        ))
    return convs


def load_esconv(path=None, label_split="train"):
    path = path or os.path.join(_DIR, "esconv.json")
    with open(path) as f:
        convs_raw = json.load(f)
    convs = []
    for rec in convs_raw:
        turns = [t["text"] for t in rec["turns"]]
        speakers = _speaker_sequence(rec["turns"], "sys")
        convs.append(Conversation(
            conv_id=None, turns=turns, speakers=speakers, label=0,
            pattern="esconv_real", generator="human", harmful_turn_idx=None,
            split=label_split, is_twin=False,
        ))
    return convs


def load_annomi(path=None, label_split="train"):
    path = path or os.path.join(_DIR, "annomi.json")
    with open(path) as f:
        convs_raw = json.load(f)
    convs = []
    for rec in convs_raw:
        turns = [t["text"] for t in rec["turns"]]
        speakers = _speaker_sequence(rec["turns"], "sys")
        convs.append(Conversation(
            conv_id=None, turns=turns, speakers=speakers, label=0,
            pattern="annomi_real", generator="human", harmful_turn_idx=None,
            split=label_split, is_twin=False,
        ))
    return convs


def load_all_real_legitimate(rng, train_frac=0.85, n_secondary_fpr=300,
                              n_enron=None, n_esconv=None, n_annomi=None):
    """Loads Enron + ESConv + AnnoMI. Enron is split into two
    non-overlapping groups up front: most of it goes into train/val
    (same as ESConv/AnnoMI), and a separate slice (n_secondary_fpr
    threads, never used in training or calibration) is held out purely
    for the secondary FPR check: at the threshold calibrated on our
    synthetic validation twins, what fraction of real human business
    threads still trigger a false alarm? ESConv/AnnoMI have no held-out
    slice -- they're here for training diversity, not for an FPR check.
    Returns (train_and_val_convs, secondary_fpr_convs). Optional n_* caps
    how many of each to use (None = use all available)."""
    enron = load_enron()
    rng.shuffle(enron)
    if n_enron is not None:
        enron = enron[:n_enron]
    secondary_fpr = enron[:n_secondary_fpr]
    for c in secondary_fpr:
        c.split = "secondary_fpr"
    enron_for_train = enron[n_secondary_fpr:]

    esconv = load_esconv()
    annomi = load_annomi()
    if n_esconv is not None:
        esconv = rng.sample(esconv, min(n_esconv, len(esconv)))
    if n_annomi is not None:
        annomi = rng.sample(annomi, min(n_annomi, len(annomi)))

    all_real = enron_for_train + esconv + annomi
    rng.shuffle(all_real)
    n_train = int(len(all_real) * train_frac)
    for c in all_real[:n_train]:
        c.split = "train"
    for c in all_real[n_train:]:
        c.split = "val"
    return all_real, secondary_fpr
