"""Exact Shapley values for one decision of the network.

Players   groups of pieces: the columns (<= 7 players) or, for small positions, single pieces.
Coalition a set S of players. The board for S keeps S's pieces and removes everyone else's.
Value     v(S) = what the network says about that board: the probability it gives the explained
          move (target="policy") or its value head (target="value").
Shapley   phi_i = average, over all orders of adding players one at a time, of how much v
          rises when player i joins.  Computed exactly from all 2^n coalition values, so the
          values always add up:  sum(phi) = v(everyone) - v(nobody).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .board import COLS, ROWS, Board

MAX_PLAYERS = 16     # 2^16 = 65,536 network evaluations


@dataclass
class Explanation:
    board: Board
    target: str                              # "policy[col N]" or "value"
    labels: list[str]
    members: list[list[tuple[int, int]]]     # (row, col) cells of each player
    phi: np.ndarray                          # Shapley value per player
    base: float                              # v(nobody): the network on an empty board
    full: float                              # v(everybody): the network on the real board
    move: int | None = None                  # explained column (0-based) for the policy target

    def cell_map(self) -> np.ma.MaskedArray:
        out, covered = np.zeros((ROWS, COLS)), np.zeros((ROWS, COLS), bool)
        for p, cells in zip(self.phi, self.members):
            for r, c in cells:
                out[r, c], covered[r, c] = p, True
        return np.ma.masked_array(out, mask=~covered)

    def table(self) -> str:
        lines = [f"target: {self.target}   base={self.base:+.4f}  full={self.full:+.4f}"]
        lines += [f"  {self.labels[i]:>8}  {self.phi[i]:+.4f}" for i in np.argsort(-np.abs(self.phi))]
        lines.append(f"  sum(phi)={self.phi.sum():+.4f}  full-base={self.full - self.base:+.4f}")
        return "\n".join(lines)


def shapley_values(v: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley values. v[mask] = value of the coalition whose members are the set bits of mask."""
    masks = np.arange(1 << n)
    size = np.array([bin(m).count("1") for m in masks])
    weight = np.array([math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)])
    phi = np.zeros(n)
    for i in range(n):
        without = masks[(masks >> i) & 1 == 0]                      # coalitions i could join
        phi[i] = (weight[size[without]] * (v[without | (1 << i)] - v[without])).sum()
    return phi


def settle_columns(grids: np.ndarray) -> np.ndarray:
    """Let pieces fall to the bottom of their column so every board stays legal."""
    idx = np.argsort(grids != 0, axis=1, kind="stable")             # empties first, order kept
    return np.take_along_axis(grids, idx, axis=1)


def _players(grid: np.ndarray, kind: str):
    if kind == "columns":
        cols = [c for c in range(COLS) if (grid[:, c] != 0).any()]
        return ([[(int(r), c) for r in np.nonzero(grid[:, c])[0]] for c in cols],
                [f"col {c + 1}" for c in cols])
    if kind == "cells":
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(grid))]
        return [[x] for x in cells], [f"{'abcdefg'[c]}{ROWS - r}" for r, c in cells]
    raise ValueError(f"players must be 'columns' or 'cells', not {kind!r}")


def explain(model, board: Board, *, target: str = "policy", move: int | None = None,
            players: str = "columns") -> Explanation:
    """`move` is a 0-based column (default: the network's favourite on the real board)."""
    grid = board.grid
    members, labels = _players(grid, players)
    n = len(members)
    if n > MAX_PLAYERS:
        raise ValueError(f"{n} players would need 2^{n} network evaluations; use players='columns'")

    if target == "policy" and move is None:
        move = int(model.evaluate(grid[None], board.ply)[0][0].argmax())

    # one board per coalition: bit j of the mask says whether player j's pieces are kept
    masks = np.arange(1 << n)
    boards = np.repeat(grid[None], len(masks), axis=0)
    for j, cells in enumerate(members):
        gone = ((masks >> j) & 1) == 0
        for r, c in cells:
            boards[gone, r, c] = 0
    probs, values = model.evaluate(settle_columns(boards), board.ply)
    v = values if target == "value" else probs[:, move]

    desc = f"policy[col {move + 1}]" if target == "policy" else "value"
    return Explanation(board, desc, labels, members, shapley_values(v, n),
                       float(v[0]), float(v[-1]), move)
