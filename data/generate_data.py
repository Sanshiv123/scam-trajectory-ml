# Synthetic placeholder conversation generator, built to match the proposal's
# actual design: 4 BEC/vishing patterns, each scam scenario paired with a
# matched legitimate twin (same roles, same amount, same deadline, resolved
# through a normal process instead of manipulation). Conversations vary in
# length (8-16 turns) and the ask position is randomized, so turn index alone
# can't predict it. One pattern (payroll_redirect) and one generator voice are
# held out entirely, for the held-out-pattern / held-out-LLM splits.
#
# This still uses text templates rather than real LLM calls (no API access
# in this environment) -- see llm_backend.py for where to wire in real
# generation. Everything downstream only depends on the Conversation objects
# this file returns.
import random
from dataclasses import dataclass, field

AMOUNTS = ["$18,400", "$42,900", "$67,250", "$94,000", "$123,500", "$156,800"]
DEADLINES = ["by end of day", "before the wire window closes at 3pm", "by tomorrow morning",
             "before the end of the week", "within the hour"]

# ---------------------------------------------------------------------------
# Each pattern defines: the two roles, and phase-tagged line banks for the
# scam path (rapport -> trust -> urgency -> authority -> ask) and the twin
# path (rapport -> trust -> verify -> request). {amount}/{deadline}/{name}
# are filled in per scenario so twin and scam share the same stakes.
# ---------------------------------------------------------------------------
PATTERNS = {
    "executive_wire_request": {
        "roles": ("CFO", "Finance team"),
        "rapport": [
            "Hi {name}, quick one for you today.",
            "{name}, glad I caught you, need your help on something time-sensitive.",
        ],
        "trust": [
            "I'm travelling for the acquisition talks so I can't call, doing this over chat.",
            "This is the account I mentioned in the leadership sync last week.",
        ],
        "urgency": [
            "This has to go out {deadline}, the counterparty is waiting on confirmation.",
            "I know it's short notice but this window closes fast.",
        ],
        "authority": [
            "I'll take responsibility for this one, just need it processed directly.",
            "Please don't loop in anyone else on this, I want to keep it contained until it's announced.",
        ],
        "ask": "Wire {amount} to the account I'm sending through now, {deadline}.",
        "verify": [
            "Let's get this through the usual process so it's properly logged.",
            "I'll have {name2} countersign once the PO is in the system.",
        ],
        "request": "Can you process {amount} to the vendor on file {deadline}? Go ahead and have {name2} verify it first.",
    },
    "vendor_bank_detail_change": {
        "roles": ("Vendor contact", "Accounts payable"),
        "rapport": [
            "Hi {name}, this is accounts at our end reaching out about an upcoming invoice.",
            "Hope you're doing well, {name} -- following up on the payment due soon.",
        ],
        "trust": [
            "We've worked with your team for a while now so wanted to flag this directly.",
            "Our finance department updated our banking provider recently.",
        ],
        "urgency": [
            "We need the new details on file {deadline} or the next payment will bounce.",
            "This is a bit urgent since the invoice is already due.",
        ],
        "authority": [
            "Please update it on your end without a separate confirmation call, our bank is strict about timing.",
            "I'd rather not go through the usual verification this once, given the deadline.",
        ],
        "ask": "Please update our account on file to the new bank details {deadline} for the {amount} invoice.",
        "verify": [
            "Happy to send this over, though please confirm through your standard vendor verification process.",
            "We can also hop on a call with your AP team if that's easier.",
        ],
        "request": "Could you update our bank details on file for the {amount} invoice {deadline}? Feel free to verify with {name2} first.",
    },
    "it_helpdesk_vishing": {
        "roles": ("IT helpdesk", "Employee"),
        "rapport": [
            "Hi {name}, this is IT support, we got a ticket about unusual activity on your account.",
            "Hey {name}, quick security check on our end, won't take long.",
        ],
        "trust": [
            "I can see your employee ID and recent login location on our system here.",
            "This matches the pattern we've been seeing with a phishing wave this week.",
        ],
        "urgency": [
            "We need to lock this down {deadline} before it spreads to other accounts.",
            "If we don't act now your account could get suspended automatically.",
        ],
        "authority": [
            "I'm going to walk you through this directly, no need to open a separate ticket.",
            "Please don't mention this to anyone else on the floor yet, we're still containing it.",
        ],
        "ask": "I need your current login code to verify and lock down the account {deadline}.",
        "verify": [
            "I'll go ahead and log a ticket so this is tracked properly on our end.",
            "You can also call the helpdesk extension directly to confirm this is legitimate.",
        ],
        "request": "Can you confirm your employee ID so I can open a ticket for the account review {deadline}? {name2} can also verify me if needed.",
    },
    "payroll_redirect": {
        "roles": ("Payroll admin", "HR/Payroll staff"),
        "rapport": [
            "Hi {name}, following up about updating my direct deposit info.",
            "Hi {name}, hoping you can help with a quick payroll change.",
        ],
        "trust": [
            "I just switched banks so need this updated before the next cycle.",
            "HR said I could reach out directly for this instead of the portal.",
        ],
        "urgency": [
            "This needs to be in {deadline} so it applies to the next pay run.",
            "I know payroll already ran the prep, so timing matters here.",
        ],
        "authority": [
            "Let's just handle it over chat so it's faster than the portal process.",
            "No need to flag this to my manager, it's just a routine update.",
        ],
        "ask": "Please update my direct deposit account to the new routing and account number {deadline}, covering the {amount} pay cycle.",
        "verify": [
            "I'll send the standard direct-deposit change form so this is properly on file.",
            "I can also confirm with {name2} once the form comes through.",
        ],
        "request": "Could you update my direct deposit for the {amount} pay cycle {deadline}? I'll fill out the standard form and {name2} can confirm it.",
    },
}

HELD_OUT_PATTERN = "payroll_redirect"

VICTIM_REPLIES = [
    "Okay, let me take a look.",
    "Sure, one moment.",
    "Got it, thanks for letting me know.",
    "I see, give me a second.",
    "Understood, checking now.",
    "Alright, on it.",
]

# simulated "generator LLMs" -- different phrasing styles, standing in for
# different models until real LLM calls are wired in (see llm_backend.py)
GENERATORS = {
    "gpt": lambda s: s,
    "claude": lambda s: s.replace("Hi ", "Hello ").replace("Okay", "Alright"),
    "llama": lambda s: s.replace("I'm", "I am").replace("can't", "cannot"),
    "qwen": lambda s: s + " Thanks.",
}
HELD_OUT_GENERATOR = "qwen"

NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Riley", "Casey", "Jamie"]


@dataclass
class Conversation:
    conv_id: int
    turns: list = field(default_factory=list)       # list of turn strings
    speakers: list = field(default_factory=list)     # 0 = requester/attacker, 1 = target/recipient
    label: int = 0                                   # 1 = manipulative (scam), 0 = legitimate (incl. twin)
    pattern: str = None
    generator: str = "gpt"
    harmful_turn_idx: int = None                     # t_ask, manipulative only
    split: str = "train"
    is_twin: bool = False


def _pick(bank, rng, n):
    return [rng.choice(bank) for _ in range(n)]


def _render(lines, fmt, generator):
    return [GENERATORS[generator](line.format(**fmt)) for line in lines]


def make_pair(conv_id_start, pattern_name, generator, split, rng):
    """Builds one matched scam/twin pair sharing the same length, amount, and deadline."""
    p = PATTERNS[pattern_name]
    role1, role2 = p["roles"]
    name, name2 = rng.sample(NAMES, 2)
    amount, deadline = rng.choice(AMOUNTS), rng.choice(DEADLINES)
    fmt = {"name": name, "name2": name2, "amount": amount, "deadline": deadline}
    total_len = rng.randint(8, 16)

    # ---- scam path: rapport -> trust -> urgency -> authority -> ASK -> tail ----
    n_rapport, n_trust, n_urg, n_auth = 1, 1, rng.randint(1, 2), rng.randint(1, 2)
    scam_pre = (_render(_pick(p["rapport"], rng, n_rapport), fmt, generator)
                + _render(_pick(p["trust"], rng, n_trust), fmt, generator)
                + _render(_pick(p["urgency"], rng, n_urg), fmt, generator)
                + _render(_pick(p["authority"], rng, n_auth), fmt, generator))
    ask_line = _render([p["ask"]], fmt, generator)
    n_tail = max(0, total_len - len(scam_pre) - 1)
    scam_lines = scam_pre + ask_line
    harmful_idx = len(scam_lines) - 1
    for _ in range(n_tail):
        scam_lines.append(rng.choice(VICTIM_REPLIES))
    scam_turns, scam_speakers = [], []
    for i, line in enumerate(scam_lines):
        is_reply = i > harmful_idx or line in VICTIM_REPLIES
        scam_turns.append(line)
        scam_speakers.append(1 if is_reply else 0)

    # ---- twin path: rapport -> trust -> verify -> REQUEST -> tail, same total length ----
    twin_pre = (_render(_pick(p["rapport"], rng, n_rapport), fmt, generator)
                + _render(_pick(p["trust"], rng, n_trust), fmt, generator)
                + _render(_pick(p["verify"], rng, n_urg + n_auth), fmt, generator))
    request_line = _render([p["request"]], fmt, generator)
    twin_lines = twin_pre + request_line
    while len(twin_lines) < total_len:
        twin_lines.append(rng.choice(VICTIM_REPLIES))
    twin_lines = twin_lines[:total_len] if len(twin_lines) > total_len else twin_lines
    twin_turns, twin_speakers = [], []
    for i, line in enumerate(twin_lines):
        is_reply = line in VICTIM_REPLIES or i >= len(twin_pre) + 1
        twin_turns.append(line)
        twin_speakers.append(1 if is_reply else 0)

    scam = Conversation(conv_id_start, scam_turns, scam_speakers, 1, pattern_name,
                         generator, harmful_idx, split, is_twin=False)
    twin = Conversation(conv_id_start + 1, twin_turns, twin_speakers, 0, pattern_name,
                         generator, None, split, is_twin=True)
    return scam, twin


def build_dataset(n_pairs_train=25, n_pairs_val=20, n_pairs_test=15, n_pairs_ood=12, seed=42):
    rng = random.Random(seed)
    convs = []
    cid = 0

    in_dist_patterns = [p for p in PATTERNS if p != HELD_OUT_PATTERN]
    in_dist_generators = [g for g in GENERATORS if g != HELD_OUT_GENERATOR]

    for split, n in [("train", n_pairs_train), ("val", n_pairs_val), ("test", n_pairs_test)]:
        for pattern in in_dist_patterns:
            for _ in range(n):
                gen = rng.choice(in_dist_generators)
                scam, twin = make_pair(cid, pattern, gen, split, rng)
                convs += [scam, twin]
                cid += 2

    # held-out pattern: payroll_redirect, with in-distribution generators
    for _ in range(n_pairs_ood):
        gen = rng.choice(in_dist_generators)
        scam, twin = make_pair(cid, HELD_OUT_PATTERN, gen, "test_heldout_pattern", rng)
        convs += [scam, twin]
        cid += 2

    # held-out generator: in-distribution patterns, held-out generator voice
    for _ in range(n_pairs_ood):
        pattern = rng.choice(in_dist_patterns)
        scam, twin = make_pair(cid, pattern, HELD_OUT_GENERATOR, "test_heldout_llm", rng)
        convs += [scam, twin]
        cid += 2

    # both held out: payroll_redirect + held-out generator voice
    for _ in range(n_pairs_ood):
        scam, twin = make_pair(cid, HELD_OUT_PATTERN, HELD_OUT_GENERATOR, "test_heldout_both", rng)
        convs += [scam, twin]
        cid += 2

    rng.shuffle(convs)
    return convs


if __name__ == "__main__":
    data = build_dataset()
    from collections import Counter
    print(Counter((c.split, c.label) for c in data))
    print(data[0])
