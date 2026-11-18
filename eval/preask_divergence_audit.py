# Responds directly to the Nov 16 checkpoint feedback: "compare missed scams
# with their legitimate twins before the request and check whether any
# differences go beyond template wording." Uses only already-saved artifacts
# (outputs/investigation/dataset.json, error_inventory.json) -- no retraining,
# no new model, no change to any existing experiment or result.
#
# For every "late_or_missed" scam on the in-distribution test split (the
# headline synthetic/gru model, per reporting/summary.py), finds its matched
# legitimate twin via pair_id and diffs their turns strictly before the ask
# turn. A "template wording" difference would just be paraphrase; this checks
# whether the differing turns instead carry different manipulation *content*
# (urgency/authority phrasing) at matched phase positions.
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "investigation")


def load():
    with open(os.path.join(DATA_DIR, "dataset.json")) as f:
        data = json.load(f)
    with open(os.path.join(DATA_DIR, "error_inventory.json")) as f:
        inv = json.load(f)
    return data, inv


def run(split="test", kind="late_or_missed"):
    data, inv = load()
    by_id = {c["conv_id"]: c for c in data}
    by_pair = {}
    for c in data:
        by_pair.setdefault(c["pair_id"], []).append(c)

    def twin_of(conv_id):
        pair = by_pair[by_id[conv_id]["pair_id"]]
        return next(c for c in pair if c["conv_id"] != conv_id)

    targets = [e for e in inv if e["kind"] == kind and e["split"] == split]
    rows = []
    for e in targets:
        scam = by_id[e["id"]]
        twin = twin_of(e["id"])
        ask = scam["harmful_turn_idx"]
        pre_scam, pre_twin = scam["turns"][:ask], twin["turns"][:ask]
        ph_scam, ph_twin = scam["phases"][:ask], twin["phases"][:ask]
        diffs = [i for i in range(min(len(pre_scam), len(pre_twin))) if pre_scam[i] != pre_twin[i]]
        phase_diffs = [i for i in range(min(len(ph_scam), len(ph_twin))) if ph_scam[i] != ph_twin[i]]
        rows.append(dict(
            conv_id=e["id"], pattern=scam["pattern"], ask_idx=ask,
            n_pre_ask_turns=len(pre_scam), n_differing_turns_pre_ask=len(diffs),
            differing_turn_indices=diffs, differing_phases=[ph_scam[i] for i in diffs],
            phase_sequence_differs=bool(phase_diffs), max_score=e["max_score"],
        ))
    return rows


def summarize(rows):
    n = len(rows)
    n_zero_diff = sum(1 for r in rows if r["n_differing_turns_pre_ask"] == 0)
    from collections import Counter
    phase_counts = Counter(p for r in rows for p in r["differing_phases"])
    return dict(
        n_missed=n,
        n_with_zero_preask_difference=n_zero_diff,
        n_with_content_difference_pre_ask=n - n_zero_diff,
        mean_differing_turns=sum(r["n_differing_turns_pre_ask"] for r in rows) / n if n else None,
        differing_phase_counts=dict(phase_counts),
    )


if __name__ == "__main__":
    rows = run()
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    out_path = os.path.join(DATA_DIR, "preask_divergence_audit.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"Wrote {out_path}")
