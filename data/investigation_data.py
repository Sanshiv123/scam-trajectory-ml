"""Version 2 controlled template benchmark. No generated text is a real LLM output.

Matched twins share opening, turn count, timing, speakers and scenario variables.
Slow variants reuse test scenarios only; no test scenario enters training.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from data.generate_data import (Conversation, PATTERNS, GENERATORS, NAMES,
                                AMOUNTS, DEADLINES, VICTIM_REPLIES,
                                HELD_OUT_PATTERN, HELD_OUT_GENERATOR)

ROOT = Path(__file__).resolve().parent / "external"
FILLER = ["I am checking the record on my side.", "Thanks for checking the details.",
          "Please let me know once you have reviewed the information.",
          "We can keep the update in this conversation for reference."]


def make_pair(cid, pattern, style, split, rng, slow=False, scenario_id=None):
    p = PATTERNS[pattern]
    name, name2 = rng.sample(NAMES, 2)
    fmt = dict(name=name, name2=name2, amount=rng.choice(AMOUNTS),
               deadline=rng.choice(DEADLINES))
    length = rng.randint(20, 32) if slow else rng.randint(10, 18)
    possible = list(range(4, length-1, 2))
    if slow:
        possible = [t for t in possible if .70 <= t / (length-1) <= .95]
    ask = rng.choice(possible)
    transform = GENERATORS[style]
    pair = f"v2-{cid // 2}"
    scam, twin, phases = [], [], []
    for i in range(length):
        if i % 2:
            # Exact same neutral recipient replies in each twin: no speaker shortcut.
            a = b = transform(rng.choice(VICTIM_REPLIES))
            phase = "reply"
        elif i == 0:
            a = b = transform(rng.choice(p["rapport"]).format(**fmt))
            phase = "rapport"
        elif i == ask:
            a, b = (transform(p[k].format(**fmt)) for k in ("ask", "request"))
            phase = "ask"
        elif i > ask:
            a = b = transform(rng.choice(FILLER))
            phase = "followup"
        elif i == ask-2:
            a = transform(rng.choice(p["authority"]).format(**fmt))
            b = transform(rng.choice(p["verify"]).format(**fmt))
            phase = "authority"
        elif i >= ask-4 and i >= 4:
            a = transform(rng.choice(p["urgency"]).format(**fmt))
            # Keep business urgency in legitimate twins as well.
            b = transform(rng.choice(p["urgency"]).format(**fmt))
            phase = "urgency"
        else:
            bank = p["trust"] if i == 2 else FILLER
            a = b = transform(rng.choice(bank).format(**fmt))
            phase = "trust"
        scam.append(a); twin.append(b); phases.append(phase)
    common = dict(speakers=[i % 2 for i in range(length)], pattern=pattern,
                  generator=style, split=split, pair_id=pair, phases=phases,
                  scenario_id=scenario_id or hashlib.sha256(json.dumps([pattern,fmt],sort_keys=True).encode()).hexdigest()[:24])
    return (Conversation(cid, scam, label=1, harmful_turn_idx=ask, **common),
            Conversation(cid+1, twin, label=0, is_twin=True, **common))


def build_dataset(seed=42, train_pairs=40, val_pairs=20, test_pairs=20, ood_pairs=24):
    rng = random.Random(seed)
    patterns = [p for p in PATTERNS if p != HELD_OUT_PATTERN]
    styles = [s for s in GENERATORS if s != HELD_OUT_GENERATOR]
    data, cid, used_scenarios = [], 0, set()
    for split, count in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs),
                         ("test_heldout_pattern", ood_pairs),
                         ("test_heldout_style", ood_pairs), ("test_heldout_both", ood_pairs)]:
        for pat in (patterns if split in ("train", "val", "test") else [None]):
            for _ in range(count):
                p = pat or (HELD_OUT_PATTERN if split in ("test_heldout_pattern", "test_heldout_both")
                            else rng.choice(patterns))
                s = HELD_OUT_GENERATOR if split in ("test_heldout_style", "test_heldout_both") else rng.choice(styles)
                while True:
                    state = rng.getstate()
                    normal = make_pair(cid, p, s, split, rng)
                    if normal[0].scenario_id not in used_scenarios: break
                used_scenarios.add(normal[0].scenario_id)
                data.extend(normal); cid += 2
                if split == "test":
                    slow_rng = random.Random(); slow_rng.setstate(state)
                    data.extend(make_pair(cid, p, s, "test_slow", slow_rng, slow=True,
                                          scenario_id=normal[0].scenario_id)); cid += 2
    return data


def load_real(seed=42, enron_train_val=200, esconv=150, annomi=133, secondary=200):
    """Deduplicate full transcripts globally before assigning disjoint pools.

    Real Enron cleanliness/threading is inherited from provided JSON, not
    independently established. These are presumed-legitimate negatives.
    """
    rng = random.Random(seed)
    seen, pools, audit = set(), {}, {}
    # Reserve authorship source transcripts even though restyling is not run.
    reserved = json.loads((ROOT / "enron_authorship_source.json").read_text(encoding="utf-8"))
    for rec in reserved:
        text = "\n".join(t["text"].strip() for t in rec["turns"])
        seen.add(hashlib.sha256(text.encode()).hexdigest())
    for source, filename in [("enron", "enron_main.json"), ("esconv", "esconv.json"), ("annomi", "annomi.json")]:
        records = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        pool, dropped = [], 0
        for rec in records:
            turns = rec.get("turns", [])
            if len(turns) < 2 or not all(t.get("text", "").strip() for t in turns):
                dropped += 1; continue
            text = "\n".join(t["text"].strip() for t in turns)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                dropped += 1; continue
            seen.add(digest)
            first = turns[0]["speaker"]
            pool.append(Conversation(conv_id=f"real-{source}-{digest[:20]}",
                turns=[t["text"] for t in turns], speakers=[int(t["speaker"] != first) for t in turns],
                label=0, generator="human", pattern=f"{source}_real", pair_id=f"real-{digest}",
                scenario_id=f"real-{digest}"))
        rng.shuffle(pool); pools[source] = pool
        audit[source] = dict(input=len(records), filtered_or_duplicate=dropped, eligible=len(pool))
    if len(pools["enron"]) < secondary + enron_train_val:
        raise ValueError("Insufficient unique Enron examples")
    holdout = pools["enron"][:secondary]
    for c in holdout: c.split = "secondary_enron"
    mixed = pools["enron"][secondary:secondary+enron_train_val] + pools["esconv"][:esconv] + pools["annomi"][:annomi]
    rng.shuffle(mixed)
    for i, c in enumerate(mixed): c.split = "train" if i < int(.85*len(mixed)) else "val_real"
    return mixed, holdout, audit


def validate(data):
    assert len({c.conv_id for c in data}) == len(data), "Duplicate IDs"
    groups, scenarios = {}, {}
    for c in data:
        assert len(c.turns) == len(c.speakers) > 0
        assert c.label in (0, 1)
        if c.label: assert 0 <= c.harmful_turn_idx < len(c.turns)
        if c.pair_id: groups.setdefault(c.pair_id, []).append(c)
        if c.scenario_id and c.generator != "human":
            # Normal / slow variants may share one test scenario, never train.
            family = "test" if c.split == "test_slow" else c.split
            old = scenarios.setdefault(c.scenario_id, family)
            assert old == family, "Scenario crosses train/validation/test"
    for pair in groups.values():
        assert len({c.split for c in pair}) == 1, "Pair crosses split"
        if len(pair) == 2 and not pair[0].pair_id.startswith("legacy-"):
            a, b = sorted(pair, key=lambda c: c.label, reverse=True)
            assert a.turns[:2] == b.turns[:2]
            assert a.speakers == b.speakers and len(a.turns) == len(b.turns)
    return dict(n=len(data), splits=dict(Counter(c.split for c in data)),
                templates=True, human_realism_audit_completed=False)
