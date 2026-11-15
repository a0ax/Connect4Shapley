"""Needs the real weights + `az` package: C4SHAP_REAL=1 pytest -k real"""
import os

import pytest

from c4shap import AZModel, Board

pytestmark = pytest.mark.skipif(not os.environ.get("C4SHAP_REAL"), reason="needs real weights")


def test_network_takes_the_win_under_the_chosen_encoding():
    m = AZModel()                       # raises if no encoding makes it play sensibly
    probs, _ = m.evaluate(Board.from_moves("121212").grid[None], 6)
    assert probs[0].argmax() == 0
