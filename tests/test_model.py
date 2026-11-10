import numpy as np

from c4shap import Board
from c4shap.model import SolverModel, calibrate_encodings, make_battery, _split_outputs


def test_solver_model_plays_perfectly():
    m = SolverModel()
    b = Board.from_moves("152637")
    probs, values = m.evaluate(b.grid[None], b.ply)
    assert probs[0].argmax() == 3 and probs[0].sum() == 1 and values[0] == 1
    b = Board.from_moves("3654461133122")                  # only column 5 holds the draw
    probs, values = m.evaluate(b.grid[None], b.ply)
    assert probs[0].argmax() == 4 and values[0] == 0


def test_solver_model_handles_masked_board_with_a_four():
    g = np.zeros((6, 7), np.int8)
    g[5, :4] = 1                                           # mover already has four in a row
    probs, values = SolverModel().evaluate(g[None], 7)
    assert values[0] == 1 and abs(probs[0].sum() - 1) < 1e-6


def test_calibration_ranks_a_correct_encoding_first():
    good = SolverModel()
    blind = lambda g, ply: (np.full((len(g), 7), 1 / 7), np.zeros(len(g)))
    rows = calibrate_encodings({"blind": blind, "solver": good.evaluate}, make_battery(20))
    assert rows[0][0] == "solver" and rows[0][1] == 1.0 and rows[-1][1] < 0.5


class _T:                                                  # stands in for a torch tensor
    def __init__(self, a): self.a = np.asarray(a, np.float32)
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


def test_split_outputs_accepts_either_head_order():
    pol, val = np.zeros((3, 7)), np.ones((3, 1))
    for out in ((_T(pol), _T(val)), (_T(val), _T(pol)), {"value": _T(val), "policy": _T(pol)}):
        p, v = _split_outputs(out)
        assert p.shape == (3, 7) and v.shape == (3,)


def test_solver_model_limit_and_fallback():
    import pytest
    from c4shap.model import ToyModel
    from c4shap.solver import SearchLimit

    empty = np.zeros((1, 6, 7), np.int8)
    with pytest.raises(SearchLimit):
        SolverModel(node_limit=1000).evaluate(empty, 0)
    m = SolverModel(node_limit=1000, fallback=ToyModel())
    probs, _ = m.evaluate(empty, 0)
    assert m.fallbacks == 1 and abs(probs[0].sum() - 1) < 1e-6
