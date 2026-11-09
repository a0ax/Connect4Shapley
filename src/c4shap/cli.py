from __future__ import annotations

import argparse
import sys

from .board import Board


def _model(a):
    name = "toy" if a.demo else a.model
    if name == "toy":
        from .model import ToyModel
        return ToyModel()
    if name == "solver":
        from .model import SolverModel, ToyModel
        # sparse Shapley coalitions are unsolvable in practice: cap the search, fall back to the toy model
        return SolverModel(node_limit=5_000_000, fallback=ToyModel())
    from .model import AZModel
    return AZModel(a.device, calibrate=not a.no_calibrate)


def cmd_solve(argv):
    """c4shap solve MOVES: exact solver output for a position."""
    p = argparse.ArgumentParser(prog="c4shap solve", description="Exact (perfect-play) analysis")
    p.add_argument("moves")
    p.add_argument("--node-limit", type=float, default=None, help="give up after this many nodes")
    p.add_argument("--backend", choices=["auto", "c", "python"], default="auto")
    a = p.parse_args(argv)
    from .solver import Position, Solver, score_to_text

    board = Board.from_moves(a.moves)
    print(board, "\n")
    s = Solver(node_limit=None if a.node_limit is None else int(a.node_limit), backend=a.backend)
    pos = Position.from_moves(a.moves)
    scores = s.move_scores(pos)
    best = max(v for v in scores if v is not None)
    print(f"solver backend: {s.backend}   nodes: {s.nodes}")
    for c, v in enumerate(scores):
        if v is not None:
            print(f"  col {c + 1}: {v:+3d}{'   <- best' if v == best else ''}")
    print(f"result: {score_to_text(best, pos.moves)}")


def cmd_doctor(argv):
    """c4shap doctor: check the real network against the solver."""
    p = argparse.ArgumentParser(prog="c4shap doctor", description="Check the AlphaZero model end to end")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    from .model import AZModel

    m = AZModel(a.device, calibrate=True)
    print(f"input planes: {m.in_channels}   chosen encoding: {m.encoding}")
    print("top candidates (share of solver-optimal moves / value-sign agreement):")
    for name, mv, vs in m.report[:5]:
        print(f"  {mv:5.0%}  {vs:5.0%}   {name}")
    print("OK: the network plays measurably better than chance under that encoding.")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "solve":
        return cmd_solve(argv[1:])
    if argv and argv[0] == "doctor":
        return cmd_doctor(argv[1:])

    p = argparse.ArgumentParser(prog="c4shap", description="Explain a Connect 4 AI decision "
                                "(also: `c4shap solve MOVES`, `c4shap doctor`)")
    p.add_argument("moves", help="move string, columns 1-7 (e.g. 4453); the side to move is explained")
    p.add_argument("--model", choices=["az", "solver", "toy"], default="az",
                   help="az = AlphaZero net, solver = perfect player, toy = heuristic (default: az)")
    p.add_argument("--explainer", choices=["shapley", "groups"], default="shapley",
                   help="shapley = exact/sampled Shapley values of the model; "
                        "groups = experimental stone-group attribution (ignores --model)")
    p.add_argument("--groups", choices=["columns", "cells"], default="columns",
                   help="Shapley players (only with --explainer shapley)")
    p.add_argument("--target", choices=["policy", "value"], default="policy")
    p.add_argument("--move", type=int, help="column 1-7 to explain (default: the model's top move)")
    p.add_argument("--nsamples", type=int, default=4096, help="model-evaluation budget when sampling")
    p.add_argument("--no-settle", action="store_true", help="leave floating pieces when masking")
    p.add_argument("--demo", action="store_true", help="same as --model toy")
    p.add_argument("--no-calibrate", action="store_true", help="skip the AZModel encoding check")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="save a PNG of the explanation")
    a = p.parse_args(argv)

    board = Board.from_moves(a.moves)
    print(board, "\n")

    if a.explainer == "groups":
        from .groups import explain_groups
        exp = explain_groups(board)
        print(exp.table())
        if a.out:
            from .viz import plot_groups
            plot_groups(exp, a.out, title=f"group explainer: {a.moves}")
            print(f"saved {a.out}")
        return

    from .explain import explain
    model = _model(a)
    exp = explain(model, board, target=a.target, move=None if a.move is None else a.move - 1,
                  groups=a.groups, settle=not a.no_settle, nsamples=a.nsamples)
    print(exp.table())
    if getattr(model, "fallbacks", 0):
        print(f"note: {model.fallbacks} coalition boards were too sparse to solve exactly and used the toy heuristic")
    if a.out:
        from .viz import plot
        plot(exp, a.out)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
