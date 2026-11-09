import random
from functools import lru_cache

import pytest

from c4shap.solver import Position, SearchLimit, Solver, score_to_text


def naive_value(seq: str) -> int:
    """Independent reference: plain grid minimax with the same score convention."""
    start = tuple(tuple(i % 2 for i, ch in enumerate(seq) if int(ch) - 1 == c) for c in range(7))

    def won(cols, player, c):
        r = len(cols[c]) - 1
        get = lambda cc, rr: cols[cc][rr] if 0 <= cc < 7 and 0 <= rr < len(cols[cc]) else None
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            n = 1
            for s in (1, -1):
                k = 1
                while get(c + s * k * dc, r + s * k * dr) == player:
                    n, k = n + 1, k + 1
            if n >= 4:
                return True
        return False

    @lru_cache(None)
    def val(state, moves):
        if moves == 42:
            return 0
        cs, best = [list(x) for x in state], None
        for c in range(7):
            if len(cs[c]) >= 6:
                continue
            cs[c].append(moves % 2)
            v = (43 - moves) // 2 if won(cs, moves % 2, c) else -val(tuple(map(tuple, cs)), moves + 1)
            cs[c].pop()
            best = v if best is None or v > best else best
        return best

    return val(start, len(seq))


def random_position(n, rng):
    while True:
        p, seq = Position(), ""
        for _ in range(n):
            legal = [c for c in range(7) if p.can_play(c)]
            c = rng.choice(legal)
            if p.is_winning_move(c):
                break
            seq, p = seq + str(c + 1), p.play(c)
        else:
            return seq


def test_immediate_win_score():
    assert Solver().solve("112233") == 18          # X plays d1 on its 4th stone


def test_matches_naive_minimax():
    rng = random.Random(1)
    for n in (30, 29, 28, 28, 27, 27):
        seq = random_position(n, rng)
        assert Solver(backend="python").solve(seq) == naive_value(seq), seq


def test_c_and_python_backends_agree():
    try:
        c = Solver(backend="c")
    except RuntimeError:
        pytest.skip("no C compiler")
    rng = random.Random(2)
    for n in (24, 20, 18, 16):
        seq = random_position(n, rng)
        assert c.solve(seq) == Solver(backend="python").solve(seq), seq


def test_weak_solve_has_same_sign():
    rng = random.Random(3)
    for n in (22, 18, 16):
        seq = random_position(n, rng)
        full, weak = Solver().solve(seq), Solver().solve(seq, weak=True)
        assert (full > 0) - (full < 0) == weak


def test_best_move_blocks_single_threat():
    assert Solver().best_moves("152617") == [3]    # O threatens d1: column 4 is the only save


def test_node_limit():
    with pytest.raises(SearchLimit):
        Solver(node_limit=10, backend="python").solve("4")


def test_illegal_and_finished_games():
    with pytest.raises(ValueError):
        Position.from_moves("1111111")             # column 1 overflows
    with pytest.raises(ValueError):
        Position.from_moves("12121212")            # X already won vertically on ply 7


def test_score_text():
    assert score_to_text(0, 20) == "draw with best play"
    assert "wins on ply 7" in score_to_text(18, 6)
    assert "loses" in score_to_text(-18, 6)
