"""Shapley explanation of one decision.

Players are groups of pieces (columns, or individual cells). A coalition S keeps its
pieces and "removes" everyone else's; v(S) is the network's output on the resulting
board (probability of the explained move, or the value head). `shap` computes the
Shapley values: exactly for <= 10 players, by permutation sampling otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap

from .board import COLS, ROWS, Board


@dataclass
class Explanation:
    board: Board
    target: str                                # "policy[col N]" or "value"
    labels: list[str]
    members: list[list[tuple[int, int]]]       # cells belonging to each player
    phi: np.ndarray                            # Shapley value per player
    base: float                                # v(empty coalition)
    full: float                                # v(all pieces)
    method: str                                # "exact" or "permutation"
    move: int | None = None                    # explained column (0-based) for policy target

    def cell_map(self) -> np.ma.MaskedArray:
        out, covered = np.zeros((ROWS, COLS)), np.zeros((ROWS, COLS), bool)
        for p, cells in zip(self.phi, self.members):
            for r, c in cells:
                out[r, c], covered[r, c] = p, True
        return np.ma.masked_array(out, mask=~covered)

    def table(self) -> str:
        order = np.argsort(-np.abs(self.phi))
        lines = [f"target: {self.target}   base={self.base:+.4f}  full={self.full:+.4f}  ({self.method})"]
        lines += [f"  {self.labels[i]:>8}  {self.phi[i]:+.4f}" for i in order]
        lines.append(f"  sum(phi)={self.phi.sum():+.4f}  full-base={self.full - self.base:+.4f}")
        return "\n".join(lines)


def settle_columns(grids: np.ndarray) -> np.ndarray:
    """Let pieces fall to the bottom of their column (keeps gravity-valid boards)."""
    idx = np.argsort(grids != 0, axis=1, kind="stable")  # empties first, order kept
    return np.take_along_axis(grids, idx, axis=1)


def _groups(grid: np.ndarray, kind: str):
    if kind == "cells":
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(grid))]
        return [[x] for x in cells], [f"{'abcdefg'[c]}{ROWS - r}" for r, c in cells]
    if kind == "columns":
        cols = [c for c in range(COLS) if (grid[:, c] != 0).any()]
        return ([[(int(r), c) for r in np.nonzero(grid[:, c])[0]] for c in cols],
                [f"col {c + 1}" for c in cols])
    raise ValueError(f"unknown groups {kind!r}")


def explain(model, board: Board, *, target: str = "policy", move: int | None = None,
            groups: str = "columns", settle: bool = True, nsamples: int = 4096,
            seed: int = 0) -> Explanation:
    """move is a 0-based column; defaults to the model's top move on the full board."""
    grid = board.grid
    members, labels = _groups(grid, groups)
    n = len(members)
    masks = np.zeros((n, ROWS, COLS), np.int32)
    for j, cells in enumerate(members):
        for r, c in cells:
            masks[j, r, c] = 1

    if target == "policy" and move is None:
        move = int(model.evaluate(grid[None], board.ply)[0][0].argmax())

    def fn(Z):
        absent = (np.asarray(Z) < 0.5).astype(np.int32)                  # (B, n)
        drop = np.einsum("bn,nrc->brc", absent, masks) > 0
        grids = np.where(drop, 0, grid[None]).astype(np.int8)
        if settle:
            grids = settle_columns(grids)
        probs, values = model.evaluate(grids, board.ply)
        return values if target == "value" else probs[:, move]

    masker = shap.maskers.Independent(np.zeros((1, n)), max_samples=1)  # absent = removed
    x = np.ones((1, n))
    if n <= 10:
        method, ex = "exact", shap.explainers.Exact(fn, masker)(x, silent=True)
    else:
        method = "permutation"
        ex = shap.explainers.Permutation(fn, masker, seed=seed)(x, max_evals=nsamples, silent=True)

    desc = f"policy[col {move + 1}]" if target == "policy" else "value"
    return Explanation(board, desc, labels, members, np.asarray(ex.values[0], float),
                       float(fn(np.zeros((1, n)))[0]), float(fn(x)[0]), method, move)
