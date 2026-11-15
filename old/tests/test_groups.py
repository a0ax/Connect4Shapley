import random

import numpy as np

from c4shap import Board
from c4shap.groups import explain_groups, find_groups
from c4shap.positions import POSITIONS, get
from c4shap.solver import Position, Solver


def roles(move):
    return {i.role for i in move.items}


def test_finds_contiguous_and_gapped_threes():
    g = find_groups(Board.from_moves(get("horizontal_three").moves).grid)
    mine3 = [x for x in g if x.color == 1 and x.size == 3]
    assert len(mine3) == 1 and mine3[0].direction == "-" and (5, 3) in mine3[0].completions
    gap = [x for x in find_groups(Board.from_moves(get("gapped_three").moves).grid)
           if x.color == 1 and x.size == 3]
    assert len(gap) == 1 and gap[0].completions == ((5, 2),)     # c1 completes a1 b1 _ d1


def test_tactic_positions_pick_the_expected_move():
    for p in POSITIONS:
        assert explain_groups(Board.from_moves(p.moves)).best.col + 1 == p.explainer, p.name


def test_completing_a_group_wins():
    for name in ("horizontal_three", "vertical_three", "gapped_three", "diagonal_three"):
        best = explain_groups(Board.from_moves(get(name).moves)).best
        assert "completes four" in roles(best), name
        assert best.score > 900


def test_must_block_charges_the_unblocked_move():
    e = explain_groups(Board.from_moves(get("must_block").moves))
    by_col = {m.col: m for m in e.moves}
    assert "blocks" in roles(by_col[3])
    assert "leaves unblocked" in roles(by_col[0]) and by_col[0].score < -900


def test_poisoned_square_is_flagged():
    e = explain_groups(Board.from_moves(get("poisoned_square").moves))
    bad = next(m for m in e.moves if m.col == 3)
    enabled = [i for i in bad.items if i.role == "enables opp win"]
    assert enabled and enabled[0].group.color == -1 and enabled[0].group.direction == "/"
    # the tempting vertical three is credited, and still not worth it
    assert any(i.role == "builds / reach" and i.amount > 9 for i in bad.items)
    assert bad.score < -900 and e.best.col != 3


def test_double_threat_is_recognised():
    best = explain_groups(Board.from_moves(get("double_threat").moves)).best
    assert best.col == 3 and "double threat" in roles(best)


def test_full_columns_are_skipped():
    e = explain_groups(Board.from_moves("111111222222"))
    assert {m.col for m in e.moves} == {2, 3, 4, 5, 6}


def test_agrees_with_solver_most_of_the_time():
    rng, solver, hit, n = random.Random(11), Solver(), 0, 0
    while n < 50:
        p, seq = Position(), ""
        for _ in range(rng.randint(14, 24)):
            c = rng.choice([c for c in range(7) if p.can_play(c)])
            if p.is_winning_move(c):
                break
            seq, p = seq + str(c + 1), p.play(c)
        else:
            if p.winning_cells() & p.playable():
                continue
            n += 1
            hit += explain_groups(Board.from_moves(seq)).best.col in solver.best_moves(p)
    assert hit / n >= 0.7        # ~85% measured over 12k positions; not a solver
