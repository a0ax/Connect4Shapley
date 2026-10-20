"""Minimal Connect 4 board, always stored from the side-to-move's point of view."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROWS, COLS = 6, 7


@dataclass(frozen=True)
class Board:
    grid: np.ndarray  # (6, 7) int8, row 0 = top. +1 = side to move, -1 = opponent, 0 = empty
    ply: int          # number of moves played so far

    @classmethod
    def from_moves(cls, moves: str) -> "Board":
        """Build a position from a move string of columns 1-7, e.g. '4453'."""
        absolute = np.zeros((ROWS, COLS), np.int8)
        heights = [0] * COLS
        for i, ch in enumerate(moves):
            c = int(ch) - 1
            if not 0 <= c < COLS or heights[c] >= ROWS:
                raise ValueError(f"illegal move {ch!r} at ply {i + 1}")
            absolute[ROWS - 1 - heights[c], c] = 1 if i % 2 == 0 else -1
            heights[c] += 1
        grid = absolute if len(moves) % 2 == 0 else -absolute
        return cls(grid, len(moves))

    @property
    def legal_columns(self) -> np.ndarray:
        return np.flatnonzero(self.grid[0] == 0)

    def __str__(self) -> str:
        sym = {1: "X", -1: "O", 0: "."}  # X = side to move
        rows = [" ".join(sym[int(v)] for v in row) for row in self.grid]
        return "\n".join(rows + ["1 2 3 4 5 6 7"])
