"""Checks against the real network. Run with: C4SHAP_REAL=1 pytest -k real

Needs `pip install -e ".[az]"`, the `az` package and network access for the weights."""
import os

import pytest

from c4shap import AZModel, Board

pytestmark = pytest.mark.skipif(not os.environ.get("C4SHAP_REAL"), reason="needs real weights")


def test_encoding_calibrates_against_the_solver():
    m = AZModel()                      # raises if no encoding beats chance
    assert m.report and m.report[0][1] > 0.5


def test_takes_immediate_win():
    # X has three stacked in column 1 and it is X's move: playing column 1 wins at once.
    probs, value = AZModel().evaluate(Board.from_moves("121212").grid[None], 6)
    assert probs[0].argmax() == 0 and probs[0, 0] > 0.5
