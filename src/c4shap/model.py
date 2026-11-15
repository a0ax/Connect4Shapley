"""The neural network: steven0226/alphazero-connect4 (policy head + value head).

Everything else in the package only needs this one method:

    evaluate(grids, ply) -> (probs, values)
        grids   (B, 6, 7) int8, +1 = side to move, -1 = opponent, row 0 = top
        ply     moves played in the ORIGINAL position (kept fixed while pieces are removed)
        probs   (B, 7)  probability of each column (full columns get 0)
        values  (B,)    value head, from the side to move's point of view
"""
from __future__ import annotations

import inspect
import itertools
import json

import numpy as np

from .board import Board

REPO_ID = "steven0226/alphazero-connect4"
REVISION = "0ba2361fe4044af9f6bfadfa89997b46191077c7"

# Positions with one obviously right move (take the win, or block the only threat).
# (move string, correct column 1-7). Used to find out how to feed the network.
CHECKS = [("121212", 1), ("212121", 2), ("152637", 4), ("152617", 4),
          ("123252", 2), ("1344343225", 4), ("4335665334", 4)]


def masked_softmax(logits: np.ndarray, grids: np.ndarray) -> np.ndarray:
    logits = np.where(grids[:, 0, :] != 0, -np.inf, logits)      # full column = illegal
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# --- how the board is turned into network input ------------------------------------------
# The exact input layout of the network isn't documented anywhere we can rely on, so these
# knobs are tried automatically (see `pick_encoding`) and the one that plays best is kept.
SIDE = {"ply%2": lambda ply: ply % 2, "1-ply%2": lambda ply: 1 - ply % 2,
        "ones": lambda ply: 1, "zeros": lambda ply: 0}


def encode(grids, ply, *, mine_first=True, side="ply%2", flip=False, absolute=False, planes=3):
    """(B, 6, 7) mover-relative grids -> (B, planes, 6, 7) float32 network input."""
    g = -grids if absolute and ply % 2 else grids        # absolute: colours = player 1 / player 2
    chans = [g == 1, g == -1] if mine_first else [g == -1, g == 1]
    if planes == 3:
        chans.append(np.full(g.shape, SIDE[side](ply)))
    x = np.stack(chans, 1).astype(np.float32)
    return np.ascontiguousarray(x[:, :, ::-1] if flip else x)


def pick_encoding(probs_fn, planes: int = 3, min_hits: int = 5):
    """Try every encoding; probs_fn(grids, ply, **options) -> (B, 7) probabilities.
    Returns (best options, report) where report = [(hits, options), ...] best first."""
    boards = [(Board.from_moves(m), col - 1) for m, col in CHECKS]
    report = []
    for mine_first, side, flip, absolute in itertools.product(
            (True, False), SIDE if planes == 3 else ("ply%2",), (False, True), (False, True)):
        opts = dict(mine_first=mine_first, side=side, flip=flip, absolute=absolute)
        hits = sum(int(probs_fn(b.grid[None], b.ply, **opts)[0].argmax() == col) for b, col in boards)
        report.append((hits, opts))
    report.sort(key=lambda r: -r[0])                      # stable: the natural default wins ties
    if report[0][0] < min_hits:
        raise RuntimeError(
            f"no input encoding made the network take the win / block the threat in at least "
            f"{min_hits} of {len(boards)} test positions (best: {report[0][0]}). "
            f"The weights probably don't match the architecture built in _build_net.")
    return report[0][1], report


class AZModel:
    def __init__(self, device: str = "cpu", check: bool = True):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        with open(hf_hub_download(REPO_ID, "config.json", revision=REVISION)) as f:
            cfg = json.load(f)
        self.torch, self.device = torch, device
        self.net = _build_net(cfg)
        self.net.load_state_dict(load_file(hf_hub_download(REPO_ID, "model.safetensors", revision=REVISION)))
        self.net.to(device).eval()
        self.planes = next((m.in_channels for m in self.net.modules() if type(m).__name__ == "Conv2d"), 3)
        self.options = dict(mine_first=True, side="ply%2", flip=False, absolute=False)
        self.report = []
        if check:
            self.options, self.report = pick_encoding(self._probs, self.planes)

    def _forward(self, grids, ply, **opts):
        t = self.torch
        with t.no_grad():
            out = self.net(t.as_tensor(encode(grids, ply, planes=self.planes, **opts)).to(self.device))
        return _split_outputs(out)

    def _probs(self, grids, ply, **opts):
        return masked_softmax(self._forward(grids, ply, **opts)[0], grids)

    def evaluate(self, grids, ply):
        probs, values = [], []
        for i in range(0, len(grids), 512):
            chunk = grids[i:i + 512]
            logits, v = self._forward(chunk, ply, **self.options)
            probs.append(masked_softmax(logits, chunk))
            values.append(v)
        return np.concatenate(probs), np.concatenate(values)


def _split_outputs(out):
    """Network output -> (policy logits (B, 7), value (B,)), whatever order the heads come in."""
    items = out.values() if isinstance(out, dict) else out
    arrs = [a.detach().float().cpu().numpy() for a in items]
    pol = next((a for a in arrs if a.ndim >= 2 and a.shape[-1] == 7), None)
    val = next((a for a in arrs if a is not pol), None)
    if pol is None or val is None:
        raise RuntimeError(f"unexpected network output shapes {[a.shape for a in arrs]}")
    return pol.reshape(len(pol), 7), val.reshape(len(val))


def _build_net(cfg: dict):
    """Instantiate the network class from the `az` package
    (pip install git+https://github.com/kuotunyu/alphazero-connect4)."""
    from az import model as az_model
    from torch import nn

    classes = [c for _, c in inspect.getmembers(az_model, inspect.isclass)
               if issubclass(c, nn.Module) and c.__module__ == az_model.__name__]
    classes.sort(key=lambda c: not ("policy" in c.__name__.lower() or "value" in c.__name__.lower()))
    errors = []
    for cls in classes:
        try:
            params = inspect.signature(cls).parameters
            takes_all = any(p.kind is p.VAR_KEYWORD for p in params.values())
            return cls(**(cfg if takes_all else {k: v for k, v in cfg.items() if k in params}))
        except Exception as e:
            errors.append(f"{cls.__name__}: {e}")
    raise RuntimeError("could not build the network from az.model; tried " + "; ".join(errors or ["nothing"]))
