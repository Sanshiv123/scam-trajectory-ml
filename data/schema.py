"""Common transcript schema shared by generators, loaders, models and evaluation."""

from dataclasses import dataclass, field


@dataclass
class Conversation:
    conv_id: int
    turns: list = field(default_factory=list)  # list of turn strings
    speakers: list = field(default_factory=list)  # 0 = requester/attacker, 1 = target/recipient
    label: int = 0  # 1 = manipulative (scam), 0 = legitimate (incl. twin)
    pattern: str = None
    generator: str = "gpt"
    harmful_turn_idx: int = None  # t_ask, manipulative only
    split: str = "train"
    is_twin: bool = False
    pair_id: str = None
    phases: list = field(default_factory=list)
    scenario_id: str = None
