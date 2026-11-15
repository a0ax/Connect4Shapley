"""Named positions, each given as a move string (columns 1-7, first player moves first).

Every entry records what the exact solver says (`best`, 1-based columns; checked in
tests/test_positions.py) and what the experimental group explainer picks (`explainer`).
`kind` is "tactic" for one-move patterns the group explainer is built to see, and "hard"
for positions where the solver sees things the explainer cannot.

    from c4shap.positions import POSITIONS, get
    p = get("poisoned_square"); p.moves, p.title, p.note
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NamedPosition:
    name: str
    moves: str
    title: str
    note: str
    best: tuple[int, ...]        # solver-optimal columns (1-based)
    explainer: int               # column the group explainer recommends (1-based)
    kind: str = "tactic"

    @property
    def explainer_agrees(self) -> bool:
        return self.explainer in self.best


POSITIONS: list[NamedPosition] = [
    # ---- tactics the group explainer is designed around --------------------------------
    NamedPosition(
        "horizontal_three", "152537", "Three in a row: complete the four",
        "X holds a1 b1 c1. The group's only completion cell, d1, is playable, so the move that "
        "finishes the group wins on the spot. Playing there also kills O's e1 pair.",
        (4,), 4),
    NamedPosition(
        "vertical_three", "121212", "Vertical three",
        "X stacks a1 a2 a3 while O answers in column 2. Column 1 completes the vertical group. "
        "The runner-up, column 2, is just the block of O's own vertical three.",
        (1,), 1),
    NamedPosition(
        "gapped_three", "152547", "A gapped three counts too",
        "X has a1 b1 _ d1: not contiguous, but all three stones share one live window, so it is "
        "one group and c1 completes it.",
        (3,), 3),
    NamedPosition(
        "diagonal_three", "1344343225", "Rising diagonal three",
        "X holds b2 c3 and a supporting a1, and d4 is playable: the diagonal group a1-b2-c3 "
        "completes there.",
        (4,), 4),
    NamedPosition(
        "diagonal_three_falling", "4335665334", "Falling diagonal three",
        "Same idea along the other diagonal: the group's completion cell has just become playable.",
        (4,), 4),
    NamedPosition(
        "must_block", "152617", "Block the opponent's three",
        "O holds e1 f1 g1 and threatens d1. X has no win, so the best move is the one that "
        "removes the most opponent group value: column 4. Column 1 looks natural for X "
        "but leaves the threat unblocked.",
        (4,), 4),
    NamedPosition(
        "double_threat", "2737", "Open two becomes an unstoppable three",
        "X has b1 c1. Playing d1 makes b1-c1-d1 with both a1 and e1 playable: two completion "
        "cells, one move to block them. The explainer scores this as a double threat.",
        (4,), 4),
    NamedPosition(
        "poisoned_square", "336224143", "The tempting move that hands over the win",
        "Playing d3 would make a vertical three (+9 group value) but opens d4, the completion "
        "cell of O's diagonal a1-b2-c3. The explainer charges that move -1000 against O's "
        "diagonal group and recommends column 6 instead (also winning; the solver prefers column 2).",
        (2,), 6),
    NamedPosition(
        "forced_block_wins", "463644555", "A forced block that is also the only win",
        "O threatens b1, so X must block there. Every other move loses at once, yet the block "
        "itself leads to a win on ply 40. A heuristic that only 'blocks' finds the move; "
        "only the solver knows it wins.",
        (2,), 2, "hard"),
    # ---- positions where the solver sees what groups can't -----------------------------
    NamedPosition(
        "quiet_only_move", "7356536662322", "One quiet move wins; six lose",
        "No immediate tactics, yet only column 4 does not lose (a win by a single point of "
        "score, on the last ply). The explainer picks column 1, which the solver says loses.",
        (4,), 1, "hard"),
    NamedPosition(
        "late_squeeze", "7246715413571", "A win that only arrives on ply 40",
        "Only column 3 wins, and not until the board is nearly full, so the result hinges on who "
        "runs out of safe moves first. A one-move-ahead heuristic cannot see that; the solver "
        "proves it. The explainer picks column 5, which the solver says loses.",
        (3,), 5, "hard"),
    NamedPosition(
        "lone_draw", "3654461133122", "Only one move holds the draw",
        "Every move except column 5 loses within a few moves. The explainer finds it here "
        "(the threat it must stop is visible as a group), but the position is drawn: nothing "
        "in the group values says so.",
        (5,), 5, "hard"),
]

_BY_NAME = {p.name: p for p in POSITIONS}


def get(name: str) -> NamedPosition:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown position {name!r}; choose from {', '.join(_BY_NAME)}") from None
