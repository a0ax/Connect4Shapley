import itertools
import math

import numpy as np
import pytest

from c4shap import Board, explain, shapley_values
from c4shap.explain import settle_columns


class ColumnWeights:
    """value = sum(weight[col] * grid): every piece's Shapley value is known in closed form."""

    def __init__(self, w):
        self.w = w

    def evaluate(self, grids, ply):
        return np.full((len(grids), 7), 1 / 7), (grids * self.w).sum((1, 2)).astype(float)


W = np.random.default_rng(0).normal(size=7)


def test_board_perspective():
    b = Board.from_moves("4")
    assert b.grid[5, 3] == -1 and b.ply == 1          # opponent just moved; the mover sees -1


def test_bad_move_strings():
    with pytest.raises(ValueError, match="not a move string"):
        Board.from_moves("MOVES")
    with pytest.raises(ValueError, match="full"):
        Board.from_moves("1111111")


def test_settle():
    col = np.zeros((1, 6, 7), np.int8)
    col[0, 1, 0], col[0, 3, 0] = 1, -1
    assert list(settle_columns(col)[0, :, 0]) == [0, 0, 0, 0, 1, -1]


def brute_force_shapley(v, n):
    """The definition: average marginal contribution over every ordering of the players."""
    phi = np.zeros(n)
    for order in itertools.permutations(range(n)):
        mask = 0
        for i in order:
            phi[i] += v[mask | 1 << i] - v[mask]
            mask |= 1 << i
    return phi / math.factorial(n)


def test_matches_the_definition():
    for n in (1, 3, 5):
        v = np.random.default_rng(n).normal(size=1 << n)
        assert np.allclose(shapley_values(v, n), brute_force_shapley(v, n))


def test_axioms():
    n, rng = 4, np.random.default_rng(1)
    v = rng.normal(size=1 << n)
    assert shapley_values(v, n).sum() == pytest.approx(v[-1] - v[0])          # efficiency
    # dummy player: adding player 3 never changes anything -> value 0
    d = np.array([v[m & 0b0111] for m in range(1 << n)])
    assert shapley_values(d, n)[3] == pytest.approx(0)
    # symmetry: players 0 and 1 are interchangeable
    s = np.array([v[(m & 0b1100) | (bin(m & 0b11).count("1"))] for m in range(1 << n)])
    phi = shapley_values(s, n)
    assert phi[0] == pytest.approx(phi[1])


def test_columns_and_cells_on_an_additive_model():
    b = Board.from_moves("445312")
    by_col = explain(ColumnWeights(W), b, target="value", players="columns")
    for phi, label in zip(by_col.phi, by_col.labels):
        c = int(label.split()[1]) - 1
        assert phi == pytest.approx(W[c] * b.grid[:, c].sum())
    by_cell = explain(ColumnWeights(W), b, target="value", players="cells")
    for phi, [(r, c)] in zip(by_cell.phi, by_cell.members):
        assert phi == pytest.approx(W[c] * b.grid[r, c])
    for e in (by_col, by_cell):
        assert e.phi.sum() == pytest.approx(e.full - e.base)


def test_policy_target_defaults_to_the_networks_top_move():
    class Prefers4:
        def evaluate(self, grids, ply):
            p = np.full((len(grids), 7), 0.1)
            p[:, 3] = 0.4 + 0.1 * (grids != 0).sum((1, 2)) / 6
            return p / p.sum(1, keepdims=True), np.zeros(len(grids))

    e = explain(Prefers4(), Board.from_moves("4453"))
    assert e.move == 3 and e.target == "policy[col 4]"
    assert e.phi.sum() == pytest.approx(e.full - e.base)


def test_too_many_players():
    with pytest.raises(ValueError, match="columns"):
        explain(ColumnWeights(W), Board.from_moves("44533126451234567"), players="cells")
