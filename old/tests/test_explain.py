import importlib.util

import numpy as np
import pytest

from c4shap import Board, ToyModel, explain
from c4shap.explain import settle_columns


class AdditiveModel:
    """value = sum(W * grid): Shapley values are known in closed form."""

    def __init__(self, W):
        self.W = W

    def evaluate(self, grids, ply):
        return np.full((len(grids), 7), 1 / 7), (grids * self.W).sum((1, 2)).astype(float)


needs_shap = pytest.mark.skipif(importlib.util.find_spec("shap") is None, reason="shap not installed")

W = np.random.default_rng(0).normal(size=(6, 7))


def test_board_perspective():
    b = Board.from_moves("4")
    assert b.grid[5, 3] == -1 and b.ply == 1  # opponent just moved; mover sees -1


def test_settle():
    col = np.zeros((1, 6, 7), np.int8)
    col[0, 1, 0], col[0, 3, 0] = 1, -1
    out = settle_columns(col)
    assert list(out[0, :, 0]) == [0, 0, 0, 0, 1, -1]


@needs_shap
def test_cells_additive_exact():
    b = Board.from_moves("4453")
    e = explain(AdditiveModel(W), b, target="value", groups="cells", settle=False)
    assert e.method == "exact"
    for p, [(r, c)] in zip(e.phi, e.members):
        assert p == pytest.approx(b.grid[r, c] * W[r, c])


@needs_shap
def test_columns_additive_exact():
    b = Board.from_moves("4453")
    e = explain(AdditiveModel(W), b, target="value", groups="columns", settle=False)
    for p, cells in zip(e.phi, e.members):
        assert p == pytest.approx(sum(b.grid[r, c] * W[r, c] for r, c in cells))


@needs_shap
@pytest.mark.parametrize("groups,moves", [("columns", "4453312"), ("cells", "44533122116655")])
def test_efficiency(groups, moves):
    e = explain(ToyModel(), Board.from_moves(moves), groups=groups, nsamples=2000)
    assert e.phi.sum() == pytest.approx(e.full - e.base, abs=1e-6)


@needs_shap
def test_solver_model_shapley_is_efficient():
    """Shapley over columns of the perfect player's value sums to full - base."""
    from c4shap import SolverModel, ToyModel

    # sparse coalitions can't be solved quickly: cap the search and fall back to the toy model
    model = SolverModel(node_limit=2_000_000, fallback=ToyModel())
    e = explain(model, Board.from_moves("1212124"), target="value", groups="columns")
    assert e.phi.sum() == pytest.approx(e.full - e.base, abs=1e-6)
