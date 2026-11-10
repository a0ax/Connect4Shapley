"""Model adapters. The explainer only needs `evaluate(grids, ply) -> (probs, values)`:

    grids:  (B, 6, 7) int8, mover-relative (+1 = side to move), row 0 = top
    ply:    move count of the ORIGINAL position (held fixed while pieces are masked)
    probs:  (B, 7) policy over legal columns (illegal = 0, rows sum to 1)
    values: (B,)   value for the side to move, in [-1, 1]
"""
from __future__ import annotations

import numpy as np

REPO_ID = "steven0226/alphazero-connect4"
REVISION = "0ba2361fe4044af9f6bfadfa89997b46191077c7"  # commit pinned by the project's Space


def masked_softmax(logits: np.ndarray, grids: np.ndarray) -> np.ndarray:
    logits = np.where(grids[:, 0, :] != 0, -np.inf, logits)
    logits = logits - logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


class ToyModel:
    """Centre-loving heuristic. Lets you run the whole pipeline without any weights."""

    W = np.array([0, 1, 2, 3, 2, 1, 0], np.float32)

    def evaluate(self, grids, ply):
        g = grids.astype(np.float32)
        values = np.tanh(0.15 * (g * self.W).sum((1, 2)))
        logits = self.W + 0.4 * (g == 1).sum(1)
        return masked_softmax(logits, grids), values


class SolverModel:
    """Perfect player backed by the exact solver (`c4shap.solver`).

    Same `evaluate(grids, ply)` interface as the other models, so it plugs into `explain`:
      probs   uniform over the *optimal* columns (win if possible, else draw, else the slowest loss)
      values  +1 / 0 / -1 for a forced win / draw / loss for the side to move
    `exact_scores` returns the full Pons-style score of every column.

    Positions with <= 4 stones can take minutes-hours to solve, so `node_limit` (default 3e8,
    ~10 s with the C backend) raises `SearchLimit` instead of hanging; call `solver.Solver()`
    directly if you want to wait it out.
    """

    requires_gravity = True      # masked boards must be settled (explain(..., settle=True))

    def __init__(self, node_limit: int | None = 300_000_000, backend: str = "auto", fallback=None):
        """fallback: another model used for boards the solver can't finish within `node_limit`
        (very sparse boards, as in Shapley coalitions). None = raise `SearchLimit`.
        `self.fallbacks` counts how many boards used it."""
        from .solver import Solver

        self.solver = Solver(node_limit=node_limit, backend=backend)
        self.fallback, self.fallbacks = fallback, 0
        self._cache: dict[tuple, tuple] = {}

    def _solve_grid(self, grid: np.ndarray, ply: int):
        from .solver import Position

        key = (grid.tobytes(), ply)
        if key not in self._cache:
            pos = Position.from_board(grid, int((grid != 0).sum()))  # masked boards: count stones
            self.solver.nodes = 0
            scores = self.solver.move_scores(pos)
            self._cache[key] = tuple(scores)
        return self._cache[key]

    def exact_scores(self, grid: np.ndarray, ply: int) -> list:
        return list(self._solve_grid(grid, ply))

    def evaluate(self, grids, ply):
        probs = np.zeros((len(grids), 7), np.float32)
        values = np.zeros(len(grids), np.float32)
        for i, g in enumerate(grids):
            g = np.asarray(g, np.int8)
            legal = (g[0] == 0)
            if not legal.any():                        # board full: nothing to play
                continue
            from .solver import Position, is_win

            pos = Position.from_board(g, int((g != 0).sum()))
            if is_win(pos.pos) or is_win(pos.pos ^ pos.mask):   # a masked coalition can contain a four
                probs[i] = legal / legal.sum()
                values[i] = 1.0 if is_win(pos.pos) else -1.0
                continue
            from .solver import SearchLimit

            try:
                scores = self._solve_grid(g, ply)
            except SearchLimit:
                if self.fallback is None:
                    raise
                self.fallbacks += 1
                p1, v1 = self.fallback.evaluate(g[None], ply)
                probs[i], values[i] = p1[0], v1[0]
                continue
            best = max(s for s in scores if s is not None)
            for c, s in enumerate(scores):
                if s == best:
                    probs[i, c] = 1.0
            probs[i] /= probs[i].sum()
            values[i] = np.sign(best)
        return probs, values


_BATTERY: dict = {}


def make_battery(n: int = 40, seed: int = 0):
    """Solver-labelled positions for checking a model: [(grid, ply, optimal_columns, sign)].

    Positions are 12-24 stones deep (the solver answers those in milliseconds), plus a few
    hand-made tactics: take the win, block the single threat."""
    if (n, seed) in _BATTERY:
        return _BATTERY[(n, seed)]
    import random

    from .board import Board
    from .solver import Position, Solver

    rng, solver, out = random.Random(seed), Solver(), []
    seqs = ["121212", "152637", "152617"]  # take a vertical win / take a row win / block the one threat
    while len(seqs) < n:
        k, p, seq = rng.randint(12, 24), Position(), ""
        ok = True
        for _ in range(k):
            legal = [c for c in range(7) if p.can_play(c)]
            if not legal:
                ok = False
                break
            c = rng.choice(legal)
            if p.is_winning_move(c):
                ok = False
                break
            seq, p = seq + str(c + 1), p.play(c)
        if ok:
            seqs.append(seq)
    for seq in seqs:
        try:
            pos = Position.from_moves(seq)
        except ValueError:
            continue
        scores = solver.move_scores(pos)
        best = max(v for v in scores if v is not None)
        b = Board.from_moves(seq)
        out.append((b.grid, b.ply, [c for c, v in enumerate(scores) if v == best],
                    int(np.sign(best))))
    _BATTERY[(n, seed)] = out
    return out


def calibrate_encodings(candidates: dict, battery) -> list[tuple[str, float, float]]:
    """Score each candidate `evaluate(grids, ply) -> (probs, values)` against the battery.

    Returns [(name, move_accuracy, value_sign_accuracy)] best first. move_accuracy is the share
    of positions where the argmax column is solver-optimal."""
    rows = []
    for name, ev in candidates.items():
        hit = sign = 0
        for grid, ply, best_cols, s in battery:
            try:
                probs, values = ev(grid[None], ply)
            except Exception:
                break
            hit += int(probs[0].argmax() in best_cols)
            sign += int(np.sign(values[0]) == s) if s != 0 else int(abs(values[0]) < 0.34)
        else:
            rows.append((name, hit / len(battery), sign / len(battery)))
    return sorted(rows, key=lambda r: (-r[1], -r[2]))


class AZModel:
    """steven0226/alphazero-connect4 (6 blocks x 96 filters, ~1.1M params, CPU-friendly).

    The network class and input encoding were never checked against the real weights (the
    old code had `VERIFY` comments in their place). Rather than trust a guess, `calibrate()`
    tries every plausible encoding (plane order, row orientation, side-to-move plane,
    absolute vs mover-relative colours) on solver-labelled positions and keeps the one that
    plays best. If none beats chance it raises with a report instead of silently explaining
    a mis-fed network. Run `c4shap doctor` to see the report.
    """

    def __init__(self, device: str = "cpu", calibrate: bool = True, encoding: str | None = None):
        import json

        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg_path = hf_hub_download(REPO_ID, "config.json", revision=REVISION)
        w_path = hf_hub_download(REPO_ID, "model.safetensors", revision=REVISION)
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.torch, self.device = torch, device
        self.net = _build_net(cfg)
        self.net.load_state_dict(load_file(w_path))
        self.net.to(device).eval()
        self.in_channels = _input_channels(self.net)
        self.encoding = encoding or "mine,theirs,side=ply%2"
        self.report: list[tuple[str, float, float]] = []
        if calibrate:
            self.calibrate()

    # -- encodings -------------------------------------------------------------------
    def _variants(self) -> dict:
        planes = {"mine": lambda g, ply: (g == 1), "theirs": lambda g, ply: (g == -1)}
        sides = {"side=ply%2": lambda g, ply: np.full(g.shape, ply % 2),
                 "side=1-ply%2": lambda g, ply: np.full(g.shape, 1 - ply % 2),
                 "side=ones": lambda g, ply: np.ones(g.shape),
                 "side=zeros": lambda g, ply: np.zeros(g.shape)}
        variants = {}
        for flip in (False, True):
            for order in (("mine", "theirs"), ("theirs", "mine")):
                for sname, sfn in sides.items():
                    for absolute in (False, True):
                        name = f"{','.join(order)},{sname}{',flip' if flip else ''}{',absolute' if absolute else ''}"

                        def enc(g, ply, order=order, sfn=sfn, flip=flip, absolute=absolute):
                            g = np.asarray(g)
                            if absolute and ply % 2 == 1:      # colours by player 1 / player 2, not by mover
                                g = -g
                            ps = [planes[o](g, ply).astype(np.float32) for o in order]
                            if self.in_channels == 3:
                                ps.append(sfn(g, ply).astype(np.float32))
                            x = np.stack(ps, 1)
                            return x[:, :, ::-1, :].copy() if flip else x
                        if self.in_channels in (2, 3):
                            if self.in_channels == 2 and not sname.endswith("ply%2"):
                                continue                       # side plane irrelevant with 2 planes
                            variants[name] = enc
        return variants

    def _run(self, x: np.ndarray):
        t = self.torch
        with t.no_grad():
            out = self.net(t.as_tensor(x, dtype=t.float32).to(self.device))
        return _split_outputs(out)

    def _evaluate_with(self, enc):
        def ev(grids, ply):
            probs, values = [], []
            for i in range(0, len(grids), 512):
                chunk = np.asarray(grids[i:i + 512])
                logits, v = self._run(enc(chunk, ply))
                probs.append(masked_softmax(logits.reshape(len(chunk), 7), chunk))
                values.append(v.reshape(len(chunk)))
            return np.concatenate(probs), np.concatenate(values)
        return ev

    def calibrate(self, n: int = 40, min_gain: float = 0.2):
        """Pick the encoding that agrees best with the solver; raise if none beats chance."""
        battery = make_battery(n)
        chance = float(np.mean([len(b[2]) / max(1, (b[0][0] == 0).sum()) for b in battery]))
        variants = self._variants()
        self.report = calibrate_encodings({k: self._evaluate_with(v) for k, v in variants.items()}, battery)
        if not self.report or self.report[0][1] < chance + min_gain:
            top = "; ".join(f"{n}: {m:.0%}" for n, m, _ in self.report[:3])
            raise RuntimeError(
                f"no input encoding made the network play better than chance ({chance:.0%}). "
                f"Best tried: {top}. The weights/architecture probably don't match what "
                f"c4shap.model._build_net builds; check src/az/model.py in the az repo.")
        self.encoding = self.report[0][0]
        self._enc = variants[self.encoding]

    def evaluate(self, grids, ply):
        enc = getattr(self, "_enc", None) or self._variants().get(self.encoding)
        if enc is None:
            raise RuntimeError(f"unknown encoding {self.encoding!r}")
        return self._evaluate_with(enc)(grids, ply)


def _input_channels(net) -> int:
    for m in net.modules():
        if m.__class__.__name__ == "Conv2d":
            return int(m.in_channels)
    return 3


def _split_outputs(out):
    """Normalise a network's output to (policy_logits (B,7 or B,k), value (B,)) as numpy."""
    items = list(out.values()) if isinstance(out, dict) else list(out)
    arrs = [np.asarray(a.detach().float().cpu().numpy()) for a in items]
    pol = next((a for a in arrs if a.ndim >= 2 and a.shape[-1] == 7), None)
    val = next((a for a in arrs if a is not pol), None)
    if pol is None or val is None:
        raise RuntimeError(f"unexpected network output shapes {[a.shape for a in arrs]}")
    return pol.reshape(len(pol), 7), val.reshape(len(val))


def _build_net(cfg: dict):
    """Instantiate the network from the `az` package (pip install git+https://github.com/kuotunyu/alphazero-connect4)."""
    import inspect

    from az import model as az_model

    torch_module = __import__("torch").nn.Module
    classes = [c for _, c in inspect.getmembers(az_model, inspect.isclass)
               if issubclass(c, torch_module) and c.__module__ == az_model.__name__]
    preferred = [c for c in classes if "policy" in c.__name__.lower() or "value" in c.__name__.lower()]
    errors = []
    for cls in preferred + [c for c in classes if c not in preferred]:
        try:
            params = inspect.signature(cls).parameters
            kwargs = cfg if any(p.kind is p.VAR_KEYWORD for p in params.values()) else \
                {k: v for k, v in cfg.items() if k in params}
            return cls(**kwargs)
        except Exception as e:                      # try the next candidate class
            errors.append(f"{cls.__name__}: {e}")
    raise RuntimeError("could not build the network from az.model; tried " + "; ".join(errors or ["no nn.Module classes"]))
