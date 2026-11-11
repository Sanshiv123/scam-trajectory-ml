"""Weak-label policy shared by all training paths; never modifies ask annotations."""

import numpy as np


def labels(c, early=False):
    y = np.array(
        [int(c.label and t >= c.harmful_turn_idx) for t in range(len(c.turns))], dtype=float
    )
    if early and c.label:
        for i, phase in enumerate(c.phases):
            if i < c.harmful_turn_idx and phase in ("urgency", "authority"):
                y[i] = 0.5
    return y
