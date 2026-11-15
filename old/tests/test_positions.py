import pytest

from c4shap.positions import POSITIONS, get
from c4shap.solver import Solver

SOLVER = Solver()


@pytest.mark.parametrize("pos", POSITIONS, ids=lambda p: p.name)
def test_solver_agrees_with_the_recorded_best_move(pos):
    assert tuple(c + 1 for c in SOLVER.best_moves(pos.moves)) == pos.best


def test_get():
    assert get("must_block").moves == "152617"
    with pytest.raises(KeyError):
        get("nope")


def test_hard_positions_include_real_explainer_failures():
    assert not get("quiet_only_move").explainer_agrees
    assert not get("late_squeeze").explainer_agrees
