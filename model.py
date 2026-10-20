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


class AZModel:
    """steven0226/alphazero-connect4 (6 blocks x 96 filters, ~1.1M params, CPU-friendly)."""

    def __init__(self, device: str = "cpu"):
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

    def _encode(self, grids: np.ndarray, ply: int):
        # VERIFY against src/az: plane order, row orientation, meaning of the 3rd plane.
        t = self.torch
        g = t.as_tensor(grids, dtype=t.float32)
        side = t.full_like(g, float(ply % 2))
        return t.stack([(g == 1).float(), (g == -1).float(), side], 1).to(self.device)

    def evaluate(self, grids, ply):
        t, probs, values = self.torch, [], []
        with t.no_grad():
            for i in range(0, len(grids), 512):
                chunk = grids[i:i + 512]
                out, v = self.net(self._encode(chunk, ply))  # VERIFY: (policy logits, value)
                logits = out.float().cpu().numpy().reshape(len(chunk), 7)
                probs.append(masked_softmax(logits, chunk))
                values.append(v.float().cpu().numpy().reshape(len(chunk)))
        return np.concatenate(probs), np.concatenate(values)


def _build_net(cfg: dict):
    # pip install git+https://github.com/kuotunyu/alphazero-connect4
    from az import model as az_model

    # VERIFY class name / constructor against src/az/model.py in that repo.
    return az_model.PolicyValueNet(**cfg)
