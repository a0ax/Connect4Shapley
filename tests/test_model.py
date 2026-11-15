import numpy as np
import pytest

from c4shap import Board
from c4shap.model import CHECKS, _split_outputs, encode, masked_softmax, pick_encoding


def test_check_positions_are_legal():
    for moves, col in CHECKS:
        assert 1 <= col <= 7 and Board.from_moves(moves).grid[0, col - 1] == 0


def test_masked_softmax_ignores_full_columns():
    g = np.zeros((1, 6, 7), np.int8)
    g[0, 0, 2] = 1
    p = masked_softmax(np.zeros((1, 7)), g)
    assert p[0, 2] == 0 and p.sum() == pytest.approx(1)


def test_encode():
    g = Board.from_moves("4").grid[None]                 # opponent stone at d1; mover = player 2
    x = encode(g, 1)
    assert x.shape == (1, 3, 6, 7) and x[0, 1, 5, 3] == 1 and x[0, 2].min() == 1   # theirs plane, side=ply%2
    assert encode(g, 1, flip=True)[0, 1, 0, 3] == 1       # rows flipped
    assert encode(g, 1, absolute=True)[0, 0, 5, 3] == 1   # absolute: player 1 plane
    assert encode(g, 1, planes=2).shape == (1, 2, 6, 7)


def test_pick_encoding_finds_the_one_that_works():
    target = dict(mine_first=False, side="1-ply%2", flip=True, absolute=False)
    answers = {Board.from_moves(m).grid.tobytes(): col - 1 for m, col in CHECKS}

    def fake_net(grids, ply, **opts):                    # plays right only under `target`
        p = np.full((1, 7), 1 / 7)
        if opts == target:
            p[0, answers[grids[0].tobytes()]] = 1
        return p

    best, report = pick_encoding(fake_net)
    assert best == target and report[0][0] == len(CHECKS) and report[1][0] < 3
    with pytest.raises(RuntimeError, match="no input encoding"):
        pick_encoding(lambda g, ply, **o: np.full((1, 7), 1 / 7) + np.eye(7)[0] * 1e-3)


class _T:                                                # stands in for a torch tensor
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
