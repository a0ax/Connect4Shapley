"""Encoding sanity check on the real network. Run with: C4SHAP_REAL=1 pytest -k real"""
import os

import pytest

from c4shap import AZModel, Board

pytestmark = pytest.mark.skipif(not os.environ.get("C4SHAP_REAL"), reason="needs real weights")


def test_takes_immediate_win():
    # X has three stacked in column 1 and it is X's move: playing column 1 wins at once.
    probs, value = AZModel().evaluate(Board.from_moves("121212").grid[None], 6)
    assert probs[0].argmax() == 0 and probs[0, 0] > 0.5
