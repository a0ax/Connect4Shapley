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
