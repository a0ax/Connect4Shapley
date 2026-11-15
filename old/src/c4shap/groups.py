"""EXPERIMENTAL group-based explainer.

Instead of asking a network which squares mattered (Shapley over columns / cells), this
explains a move by the *groups of same-colour stones* it builds on, blocks, or wakes up.

What is a group?
    All stones of one colour inside one still-live four-cell window (no enemy stone in it),
    in one direction. A contiguous three-in-a-row is the typical case; a gapped three such
    as ``X X . X`` is the same group type (it is just as dangerous). A group with k stones
    and 4-k empty cells is a threat to complete those cells.

How is a group valued?
    value = W[k] * reach, summed over the live windows that make up the group.
        W      = {1: 0, 2: 1, 3: 10}
        reach  = mean over the window's empty cells of 1 / (1 + gap), where gap is the number
                 of empty cells still below that cell in its column (0 = playable right now)

How is a move explained?
    contribution(group) = change in that group's value caused by the move (opponent groups
    count with the opposite sign: killing or blocking one is good for the mover), plus three
    tactical overlays that look exactly one reply ahead:
        completes     the move makes four                           +WIN
        leaves/enables the opponent can win on the next move        -WIN
        double threat the move creates two playable winning cells   +WIN/2
    The sum over groups is the move's score, so the attribution is exact (efficient) by
    construction, like a Shapley decomposition, but of a hand-built heuristic rather than
    of a network. It is NOT a solver: it sees one move ahead, so it can be fooled by
    zugzwang, parity and deep combinations. `c4shap.solver` is the ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import COLS, ROWS, Board

WIN = 1000.0
W = {1: 0.0, 2: 1.0, 3: 10.0}
LINE_EPS = 0.02          # tiny tie-break: number of live lines through the played cell

Cell = tuple[int, int]   # (row, col), row 0 = top
DIRS = {"-": (0, 1), "|": (1, 0), "\\": (1, 1), "/": (-1, 1)}


def _windows() -> list[tuple[str, tuple[Cell, ...]]]:
    out = []
    for name, (dr, dc) in DIRS.items():
        for r in range(ROWS):
            for c in range(COLS):
                cells = tuple((r + i * dr, c + i * dc) for i in range(4))
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    out.append((name, cells))
    return out


WINDOWS = _windows()   # 69 windows


def cell_name(cell: Cell) -> str:
    r, c = cell
    return f"{'abcdefg'[c]}{ROWS - r}"


def heights(grid: np.ndarray) -> list[int]:
    return [int((grid[:, c] != 0).sum()) for c in range(COLS)]


def landing_row(grid: np.ndarray, c: int) -> int | None:
    h = int((grid[:, c] != 0).sum())
    return None if h >= ROWS else ROWS - 1 - h


def _reach(cell: Cell, h: list[int]) -> float:
    r, c = cell
    gap = (ROWS - 1 - r) - h[c]
    return 1.0 / (1 + max(gap, 0))


def _window_value(grid: np.ndarray, cells, color: int, h: list[int]) -> tuple[float, int, bool]:
    """(value, k, live) of one window for `color`."""
    own = opp = 0
    empties = []
    for cell in cells:
        v = int(grid[cell])
        if v == color:
            own += 1
        elif v == 0:
            empties.append(cell)
        else:
            opp += 1
    if opp:
        return 0.0, own, False
    if own == 4:
        return WIN, 4, True
    if own < 2:
        return 0.0, own, True
    reach = sum(_reach(e, h) for e in empties) / len(empties)
    return W[own] * reach, own, True


def wins_at(grid: np.ndarray, color: int, cell: Cell) -> bool:
    """Would `color` make four by occupying `cell`?"""
    r, c = cell
    for dr, dc in DIRS.values():
        n = 1
        for s in (1, -1):
            rr, cc = r + s * dr, c + s * dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and grid[rr, cc] == color:
                n += 1
                rr, cc = rr + s * dr, cc + s * dc
        if n >= 4:
            return True
    return False


def playable_cells(grid: np.ndarray) -> list[Cell]:
    return [(r, c) for c in range(COLS) if (r := landing_row(grid, c)) is not None]


@dataclass(frozen=True)
class Group:
    color: int                 # +1 = side to move, -1 = opponent
    direction: str             # one of - | \ /
    stones: tuple[Cell, ...]
    completions: tuple[Cell, ...]   # empty cells (across the group's live windows)
    value: float               # standing value on the current board

    @property
    def size(self) -> int:
        return len(self.stones)

    @property
    def label(self) -> str:
        who = "mine" if self.color == 1 else "opp"
        cells = " ".join(cell_name(s) for s in sorted(self.stones, key=lambda x: (x[1], -x[0])))
        return f"{who} {self.direction} [{cells}]"


def find_groups(grid: np.ndarray, min_size: int = 2) -> list[Group]:
    """Every live group with at least `min_size` stones on the current board."""
    h = heights(grid)
    acc: dict[tuple, list] = {}
    for name, cells in WINDOWS:
        for color in (1, -1):
            value, k, live = _window_value(grid, cells, color, h)
            if not live or k < min_size or k == 4:
                continue
            stones = tuple(c for c in cells if grid[c] == color)
            empties = tuple(c for c in cells if grid[c] == 0)
            e = acc.setdefault((color, name, stones), [0.0, set()])
            e[0] += value
            e[1].update(empties)
    groups = [Group(k[0], k[1], k[2], tuple(sorted(v[1])), v[0]) for k, v in acc.items()]
    return sorted(groups, key=lambda g: -g.value)


@dataclass
class Item:
    group: Group | None        # None = the small "open lines" tie-break term
    role: str
    amount: float


@dataclass
class MoveAttribution:
    col: int                   # 0-based
    cell: Cell
    items: list[Item] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(i.amount for i in self.items)


@dataclass
class GroupExplanation:
    board: Board
    groups: list[Group]                 # standing groups (size >= 2)
    moves: list[MoveAttribution]        # one per legal column

    @property
    def best(self) -> MoveAttribution:
        return max(self.moves, key=lambda m: (m.score, -abs(m.col - 3)))

    def ranked(self) -> list[MoveAttribution]:
        return sorted(self.moves, key=lambda m: (-m.score, abs(m.col - 3)))

    def table(self, top: int = 3) -> str:
        lines = ["groups on the board (value = standing threat):"]
        if not self.groups:
            lines.append("  (none of size >= 2)")
        for g in self.groups[:8]:
            comp = ", ".join(cell_name(c) for c in g.completions)
            lines.append(f"  {g.label:<34} value {g.value:6.2f}   needs: {comp}")
        lines.append("")
        lines.append("move ranking:")
        for m in self.ranked()[:top]:
            lines.append(f"  col {m.col + 1} ({cell_name(m.cell)})  score {m.score:+9.2f}")
            for it in sorted(m.items, key=lambda i: -abs(i.amount)):
                tag = it.group.label if it.group else "open lines through cell"
                lines.append(f"      {it.amount:+9.2f}  {it.role:<18} {tag}")
        return "\n".join(lines)


def _add(acc: dict, key, role: str, amount: float):
    if abs(amount) < 1e-12:
        return
    cur = acc.get((key, role), 0.0)
    acc[(key, role)] = cur + amount


def _group_for(grid: np.ndarray, name: str, cells, color: int, h_hint=None) -> Group:
    stones = tuple(c for c in cells if grid[c] == color)
    empties = tuple(c for c in cells if grid[c] == 0)
    return Group(color, name, stones, empties, 0.0)


def explain_groups(board: Board) -> GroupExplanation:
    """Score every legal move by the groups it builds on, blocks or wakes up."""
    grid = board.grid
    h0 = heights(grid)
    out: list[MoveAttribution] = []

    for c in range(COLS):
        r = landing_row(grid, c)
        if r is None:
            continue
        cell = (r, c)
        after = grid.copy()
        after[cell] = 1
        h1 = list(h0)
        h1[c] += 1
        acc: dict = {}
        info: dict = {}                     # key -> Group for display

        # --- 1. value changes of every group (all windows), exact per group ---------
        for name, cells in WINDOWS:
            for color in (1, -1):
                v0, k0, live0 = _window_value(grid, cells, color, h0)
                v1, k1, live1 = _window_value(after, cells, color, h1)
                if k0 < 2 and k1 < 2:
                    continue
                if abs(v1 - v0) < 1e-12:
                    continue
                if k0 < 2:
                    continue                # a pair born from a single stone is handled below
                stones = tuple(x for x in cells if grid[x] == color)
                key = (color, name, stones)
                info.setdefault(key, _group_for(grid, name, cells, color))
                delta = v1 - v0
                if color == 1:
                    if k1 < 4:              # k1 == 4 is scored by the "completes four" overlay
                        _add(acc, key, "builds / reach", delta)
                else:
                    role = "blocks" if not live1 else "opens cell for"
                    _add(acc, key, role, -delta)

        # --- pairs created from single stones: k0 == 1 -> k1 == 2 for the mover ------
        for name, cells in WINDOWS:
            v0, k0, live0 = _window_value(grid, cells, 1, h0)
            v1, k1, live1 = _window_value(after, cells, 1, h1)
            if live0 and k0 == 1 and k1 == 2 and cell in cells:
                stones = tuple(x for x in cells if grid[x] == 1)
                key = (1, name, stones)
                info.setdefault(key, _group_for(grid, name, cells, 1))
                _add(acc, key, "builds / reach", v1 - v0)

        # --- 2. tactical overlays (one reply ahead) -------------------------------
        completed = [(n, cs) for n, cs in WINDOWS if cell in cs and all(
            after[x] == 1 for x in cs)]
        if completed:
            for n, cs in completed:
                before = tuple(x for x in cs if x != cell)
                key = (1, n, before)
                info.setdefault(key, Group(1, n, before, (cell,), 0.0))
                _add(acc, key, "completes four", WIN / len(completed))
        else:
            opp_wins = [x for x in playable_cells(after) if wins_at(after, -1, x)]
            if opp_wins:
                existed = {x for x in playable_cells(grid) if wins_at(grid, -1, x)}
                for x in opp_wins:
                    role = "leaves unblocked" if x in existed else "enables opp win"
                    _attribute_threat(acc, info, after, x, -1, role, -WIN / len(opp_wins))
            else:
                my_wins = [x for x in playable_cells(after) if wins_at(after, 1, x)]
                if len(my_wins) >= 2:
                    for x in my_wins:
                        _attribute_threat(acc, info, after, x, 1, "double threat", (WIN / 2) / len(my_wins))

        items = [Item(info[k], role, amt) for (k, role), amt in acc.items()]
        lines = sum(1 for _, cs in WINDOWS if cell in cs and all(grid[x] != -1 for x in cs))
        items.append(Item(None, "open lines", LINE_EPS * lines))
        out.append(MoveAttribution(c, cell, items))

    return GroupExplanation(board, find_groups(grid), out)


def _attribute_threat(acc, info, after, x: Cell, color: int, role: str, amount: float):
    """Split `amount` over the groups (in the *current* board) that make cell x a win for `color`."""
    wins = [(n, cs) for n, cs in WINDOWS if x in cs and all(
        after[y] == color or y == x for y in cs)]
    share = amount / max(len(wins), 1)
    for n, cs in wins:
        stones = tuple(y for y in cs if y != x)
        key = (color, n, stones)
        info.setdefault(key, Group(color, n, stones, (x,), 0.0))
        _add(acc, key, role, share)
