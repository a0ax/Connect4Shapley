from __future__ import annotations

import argparse

from .board import Board
from .explain import explain


def main(argv=None):
    p = argparse.ArgumentParser(prog="c4shap", description="Shapley explanation of a Connect 4 AI decision")
    p.add_argument("moves", help="move string, columns 1-7 (e.g. 4453); the side to move is explained")
    p.add_argument("--groups", choices=["columns", "cells"], default="columns")
    p.add_argument("--target", choices=["policy", "value"], default="policy")
    p.add_argument("--move", type=int, help="column 1-7 to explain (default: the model's top move)")
    p.add_argument("--nsamples", type=int, default=4096, help="model-evaluation budget when sampling")
    p.add_argument("--no-settle", action="store_true", help="leave floating pieces when masking")
    p.add_argument("--demo", action="store_true", help="toy heuristic model; no weights needed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="save a PNG of the explanation")
    a = p.parse_args(argv)

    if a.demo:
        from .model import ToyModel
        model = ToyModel()
    else:
        from .model import AZModel
        model = AZModel(a.device)

    board = Board.from_moves(a.moves)
    print(board, "\n")
    exp = explain(model, board, target=a.target, move=None if a.move is None else a.move - 1,
                  groups=a.groups, settle=not a.no_settle, nsamples=a.nsamples)
    print(exp.table())
    if a.out:
        from .viz import plot
        plot(exp, a.out)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
