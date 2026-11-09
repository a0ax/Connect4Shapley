from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .board import COLS, ROWS
from .explain import Explanation


def plot(exp: Explanation, path: str | None = None):
    """Board with each piece's group coloured by its Shapley value (blue = pushes toward
    the explained output, red = pushes against). Gold = side to move, grey = opponent."""
    cmap = exp.cell_map()
    lim = max(float(abs(exp.phi).max()), 1e-9)
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    im = ax.imshow(cmap, cmap="RdBu", vmin=-lim, vmax=lim)
    for r in range(ROWS):
        for c in range(COLS):
            v = exp.board.grid[r, c]
            if v:
                ax.add_patch(Circle((c, r), 0.32, fc="gold" if v == 1 else "dimgray", ec="black"))
    if exp.members and len(exp.members[0]) == 1:  # cell-level: annotate each piece
        for p, [(r, c)] in zip(exp.phi, exp.members):
            ax.text(c, r, f"{p:+.2f}", ha="center", va="center", fontsize=8)
    else:                                          # column-level: annotate under the axis
        by_col = {cells[0][1]: p for p, cells in zip(exp.phi, exp.members)}
        ax.set_xticks(range(COLS), [f"{c + 1}\n{by_col[c]:+.3f}" if c in by_col else str(c + 1)
                                    for c in range(COLS)])
    if exp.move is not None:
        ax.annotate("▼", (exp.move, -0.75), ha="center", color="green", annotation_clip=False)
    ax.set_xlim(-0.5, COLS - 0.5), ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_title(f"{exp.target}: base {exp.base:+.3f} → full {exp.full:+.3f}  ({exp.method} Shapley)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Shapley value")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_groups(exp, path: str | None = None, title: str | None = None):
    """Board for a `GroupExplanation`: groups drawn as bars through their stones (green = mine,
    red = opponent's, width ~ value), empty completion cells ringed, and a bar chart of the
    move scores underneath. The recommended column is marked with a green triangle."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch

    from .groups import cell_name

    grid = exp.board.grid
    best = exp.best
    fig = plt.figure(figsize=(7.4, 9.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[6.2, 2.2, 1.6], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.add_patch(FancyBboxPatch((-0.5, -0.5), COLS, ROWS, boxstyle="round,pad=0,rounding_size=0.2",
                                fc="#1f4e9c", ec="none"))
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r, c]
            ax.add_patch(Circle((c, r), 0.4, fc="gold" if v == 1 else "dimgray" if v == -1 else "#0d2a5c",
                                ec="black" if v else "none", lw=1.2))
    vmax = max([g.value for g in exp.groups] + [1.0])
    for g in exp.groups:
        col = "#2ca02c" if g.color == 1 else "#d62728"
        pts = np.array([(c, r) for r, c in g.stones], float)
        w = 4 + 14 * g.value / vmax
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=w, alpha=0.55, solid_capstyle="round", zorder=3)
        for (r, c) in g.completions:
            ax.add_patch(Circle((c, r), 0.4, fc="none", ec=col, lw=2, ls="--", zorder=4))
    r, c = best.cell
    ax.add_patch(Circle((c, r), 0.46, fc="none", ec="#00ff66", lw=3.5, zorder=5))
    ax.annotate("▼", (best.col, -1.12), ha="center", va="center", color="#00cc55", fontsize=16, annotation_clip=False)
    ax.set_xlim(-0.55, COLS - 0.45), ax.set_ylim(ROWS - 0.45, -0.55)
    ax.set_aspect("equal")
    ax.set_xticks(range(COLS), [str(i + 1) for i in range(COLS)])
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title or "group explainer", pad=34, fontsize=12)

    ax2 = fig.add_subplot(gs[1])
    cols = [m.col for m in exp.moves]
    scores = [m.score for m in exp.moves]
    colors = ["#00cc55" if m.col == best.col else ("#d62728" if m.score < -WIN_THRESHOLD else "#8aa0c8")
              for m in exp.moves]
    ax2.bar(cols, scores, color=colors)
    ax2.set_yscale("symlog", linthresh=1)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.margins(y=0.3)
    ax2.set_xticks(range(COLS), [str(i + 1) for i in range(COLS)])
    ax2.set_ylabel("move score\n(symlog)")
    for m in exp.moves:
        ax2.text(m.col, m.score, f"{m.score:+.1f}", ha="center", va="bottom" if m.score >= 0 else "top", fontsize=8)

    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    lines = [f"why col {best.col + 1} ({cell_name(best.cell)}):"]
    for it in sorted(best.items, key=lambda i: -abs(i.amount))[:4]:
        tag = it.group.label if it.group else "open lines through the cell"
        lines.append(f"  {it.amount:+8.2f}  {it.role:<16} {tag}")
    ax3.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


WIN_THRESHOLD = 500.0
