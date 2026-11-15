import argparse
import sys

from .board import Board


def check(argv):
    """c4shap check: how is the network being fed, and does it take wins / blocks?"""
    p = argparse.ArgumentParser(prog="c4shap check")
    p.add_argument("--device", default="cpu")
    from .model import CHECKS, AZModel

    m = AZModel(p.parse_args(argv).device)
    print(f"input planes: {m.planes}\nchosen encoding: {m.options}")
    print(f"correct on {m.report[0][0]}/{len(CHECKS)} win/block test positions "
          f"(runner-up encoding: {m.report[1][0]}/{len(CHECKS)})")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check"]:
        return check(argv[1:])

    p = argparse.ArgumentParser(prog="c4shap", description="Exact Shapley explanation of the "
                                "AlphaZero Connect 4 network (also: `c4shap check`)")
    p.add_argument("moves", help="move string of columns 1-7, first player first, e.g. 152637. "
                                 "The side to move is the one explained.")
    p.add_argument("--target", choices=["policy", "value"], default="policy")
    p.add_argument("--move", type=int, help="column 1-7 whose probability to explain (default: the network's pick)")
    p.add_argument("--players", choices=["columns", "cells"], default="columns",
                   help="columns = 7 players at most (fast); cells = one per piece, up to 16 pieces")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="save a heatmap PNG")
    a = p.parse_args(argv)

    try:
        board = Board.from_moves(a.moves)
        from .explain import explain
        from .model import AZModel
        exp = explain(AZModel(a.device), board, target=a.target, players=a.players,
                      move=None if a.move is None else a.move - 1)
    except ValueError as e:
        p.error(str(e))
    print(board, "\n")
    print(exp.table())
    if a.out:
        from .viz import plot
        plot(exp, a.out)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
