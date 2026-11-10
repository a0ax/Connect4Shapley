"""Perfect Connect 4 solver: exact game-theoretic values for the standard 7x6 board.

Bitboard negamax with alpha-beta pruning, a transposition table, forced-move pruning
and threat-based move ordering (the approach behind Pascal Pons' solver).

Score convention (same as Pons): for the side to move,

    score > 0   win;  score = 22 - (stones the mover will have played when the game ends)
    score = 0   draw
    score < 0   loss; symmetric

so a bigger |score| means a faster win / slower loss. `Solver.solve(moves)` takes the same
move string as `Board.from_moves` ('4453' = columns 1-7).

This is pure Python. It is fast once a position has roughly 14+ stones; from the first
few moves it can take minutes because the search tree is huge. `node_limit` lets you
bail out instead of waiting (raises `SearchLimit`).
"""
from __future__ import annotations

import numpy as np

WIDTH, HEIGHT = 7, 6
H1 = HEIGHT + 1                      # one sentinel bit on top of each column
MAX_SCORE = (WIDTH * HEIGHT + 1) // 2  # 21; scores range over [-18, 18]

# Move order: centre first.
ORDER = (3, 2, 4, 1, 5, 0, 6)


def _bottom_mask() -> int:
    m = 0
    for c in range(WIDTH):
        m |= 1 << (c * H1)
    return m


BOTTOM = _bottom_mask()
BOARD_MASK = BOTTOM * ((1 << HEIGHT) - 1)


def col_mask(c: int) -> int:
    return ((1 << HEIGHT) - 1) << (c * H1)


def top_mask(c: int) -> int:
    return 1 << (HEIGHT - 1 + c * H1)


def bottom_mask_col(c: int) -> int:
    return 1 << (c * H1)


COL_MASKS = tuple(col_mask(c) for c in range(WIDTH))
TOP_MASKS = tuple(top_mask(c) for c in range(WIDTH))
BOT_MASKS = tuple(bottom_mask_col(c) for c in range(WIDTH))


def popcount(x: int) -> int:
    return x.bit_count()


def _div2(x: int) -> int:
    """x / 2 truncating toward zero (C semantics; the score bounds rely on it)."""
    return x // 2 if x >= 0 else -((-x) // 2)


def is_win(pos: int) -> bool:
    """Does the stone set `pos` contain four in a row?"""
    m = pos & (pos >> H1)                 # horizontal
    if m & (m >> (2 * H1)):
        return True
    m = pos & (pos >> HEIGHT)             # diagonal \
    if m & (m >> (2 * HEIGHT)):
        return True
    m = pos & (pos >> (HEIGHT + 2))       # diagonal /
    if m & (m >> (2 * (HEIGHT + 2))):
        return True
    m = pos & (pos >> 1)                  # vertical
    return bool(m & (m >> 2))


def winning_cells(pos: int, mask: int) -> int:
    """Bitmask of empty cells where `pos` would complete four (ignores playability)."""
    r = (pos << 1) & (pos << 2) & (pos << 3)           # vertical
    p = (pos << H1) & (pos << (2 * H1))                # horizontal
    r |= p & (pos << (3 * H1))
    r |= p & (pos >> H1)
    p = (pos >> H1) & (pos >> (2 * H1))
    r |= p & (pos << H1)
    r |= p & (pos >> (3 * H1))
    p = (pos << HEIGHT) & (pos << (2 * HEIGHT))        # diagonal \
    r |= p & (pos << (3 * HEIGHT))
    r |= p & (pos >> HEIGHT)
    p = (pos >> HEIGHT) & (pos >> (2 * HEIGHT))
    r |= p & (pos << HEIGHT)
    r |= p & (pos >> (3 * HEIGHT))
    p = (pos << (H1 + 1)) & (pos << (2 * (H1 + 1)))    # diagonal /
    r |= p & (pos << (3 * (H1 + 1)))
    r |= p & (pos >> (H1 + 1))
    p = (pos >> (H1 + 1)) & (pos >> (2 * (H1 + 1)))
    r |= p & (pos << (H1 + 1))
    r |= p & (pos >> (3 * (H1 + 1)))
    return r & (BOARD_MASK ^ mask)


_C_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef uint64_t u64;
#define W 7
#define H 6
#define H1 7
static u64 BOTTOM, BOARD;
static const int ORDER[7] = {3, 2, 4, 1, 5, 0, 6};
typedef struct { u64 key; signed char kind; signed char v; } Entry;
static Entry *T = 0;
static u64 TSIZE = 1u << 22;
static long long nodes_, limit_;
static int aborted_;

static u64 winning(u64 p, u64 mask) {
    u64 r = (p << 1) & (p << 2) & (p << 3), q;
    q = (p << H1) & (p << (2*H1));
    r |= q & (p << (3*H1)); r |= q & (p >> H1);
    q = (p >> H1) & (p >> (2*H1));
    r |= q & (p << H1); r |= q & (p >> (3*H1));
    q = (p << H) & (p << (2*H));
    r |= q & (p << (3*H)); r |= q & (p >> H);
    q = (p >> H) & (p >> (2*H));
    r |= q & (p << H); r |= q & (p >> (3*H));
    q = (p << (H+2)) & (p << (2*(H+2)));
    r |= q & (p << (3*(H+2))); r |= q & (p >> (H+2));
    q = (p >> (H+2)) & (p >> (2*(H+2)));
    r |= q & (p << (H+2)); r |= q & (p >> (3*(H+2)));
    return r & (BOARD ^ mask);
}
static inline u64 colmask(int c) { return (((u64)1 << H) - 1) << (c * H1); }

void c4_init(void) {
    BOTTOM = 0;
    for (int c = 0; c < W; c++) BOTTOM |= (u64)1 << (c * H1);
    BOARD = BOTTOM * (((u64)1 << H) - 1);
    if (!T) T = (Entry *)calloc(TSIZE, sizeof(Entry));
}
void c4_reset(void) { if (T) memset(T, 0, TSIZE * sizeof(Entry)); }

static int negamax(u64 pos, u64 mask, int moves, int alpha, int beta) {
    nodes_++;
    if (limit_ && nodes_ > limit_) { aborted_ = 1; return 0; }
    u64 possible = (mask + BOTTOM) & BOARD;
    u64 opp = winning(pos ^ mask, mask);
    u64 forced = possible & opp;
    if (forced) {
        if (forced & (forced - 1)) return -((W*H - moves) / 2);
        possible = forced;
    }
    possible &= ~(opp >> 1);
    if (!possible) return -((W*H - moves) / 2);
    if (moves >= W*H - 2) return 0;
    int lo = -((W*H - 2 - moves) / 2);
    if (alpha < lo) { alpha = lo; if (alpha >= beta) return alpha; }
    int hi = (W*H - 1 - moves) / 2;
    if (beta > hi) { beta = hi; if (alpha >= beta) return beta; }
    u64 key = pos + mask;
    Entry *e = &T[(key * 0x9E3779B97F4A7C15ULL) >> 42];
    if (e->kind && e->key == key) {
        if (e->kind == 1) { if (beta > e->v) { beta = e->v; if (alpha >= beta) return beta; } }
        else { if (alpha < e->v) { alpha = e->v; if (alpha >= beta) return alpha; } }
    }
    u64 mv[7]; int sc[7]; int n = 0;
    for (int i = 0; i < W; i++) {
        u64 m = possible & colmask(ORDER[i]);
        if (!m) continue;
        int s = __builtin_popcountll(winning(pos | m, mask | m));
        int j = n++;
        while (j > 0 && sc[j-1] < s) { mv[j] = mv[j-1]; sc[j] = sc[j-1]; j--; }
        mv[j] = m; sc[j] = s;
    }
    for (int i = 0; i < n; i++) {
        int s = -negamax(pos ^ mask, mask | mv[i], moves + 1, -beta, -alpha);
        if (aborted_) return 0;
        if (s >= beta) { e = &T[(key * 0x9E3779B97F4A7C15ULL) >> 42]; e->key = key; e->kind = 2; e->v = (signed char)s; return s; }
        if (s > alpha) alpha = s;
    }
    e = &T[(key * 0x9E3779B97F4A7C15ULL) >> 42];
    e->key = key; e->kind = 1; e->v = (signed char)alpha;
    return alpha;
}

/* returns the score; *status = 0 ok, 1 node limit hit */
int c4_negamax(u64 pos, u64 mask, int moves, int alpha, int beta, long long limit, long long *nodes, int *status) {
    nodes_ = 0; limit_ = limit; aborted_ = 0;
    int r = negamax(pos, mask, moves, alpha, beta);
    *nodes = nodes_; *status = aborted_;
    return r;
}
"""

_lib = None
_lib_tried = False


def _load_c():
    """Compile (once) and load the C core; returns None if there's no compiler."""
    global _lib, _lib_tried
    if _lib_tried:
        return _lib
    _lib_tried = True
    import ctypes, hashlib, os, shutil, subprocess, sys, tempfile
    if os.environ.get("C4SHAP_NO_C"):
        return None
    cc = next((c for c in (os.environ.get("CC"), "cc", "gcc", "clang") if c and shutil.which(c)), None)
    if cc is None:
        return None
    ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
    tag = hashlib.sha1(_C_SOURCE.encode()).hexdigest()[:12]
    d = os.path.join(tempfile.gettempdir(), f"c4shap-{tag}")
    so = os.path.join(d, "c4solver" + ext)
    try:
        if not os.path.exists(so):
            os.makedirs(d, exist_ok=True)
            src = os.path.join(d, "c4solver.c")
            with open(src, "w") as f:
                f.write(_C_SOURCE)
            subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", so, src], check=True,
                           capture_output=True, timeout=120)
        lib = ctypes.CDLL(so)
        u64, i32, i64 = ctypes.c_uint64, ctypes.c_int, ctypes.c_longlong
        lib.c4_negamax.argtypes = [u64, u64, i32, i32, i32, i64, ctypes.POINTER(i64), ctypes.POINTER(i32)]
        lib.c4_negamax.restype = i32
        lib.c4_init()
        _lib = lib
    except Exception:
        _lib = None
    return _lib


class SearchLimit(RuntimeError):
    """Raised when `node_limit` is exceeded."""


class Position:
    """Bitboard position. `pos` = stones of the side to move, `mask` = all stones."""

    __slots__ = ("pos", "mask", "moves")

    def __init__(self, pos: int = 0, mask: int = 0, moves: int = 0):
        self.pos, self.mask, self.moves = pos, mask, moves

    @classmethod
    def from_moves(cls, seq: str) -> "Position":
        p, won = cls(), False
        for i, ch in enumerate(seq):
            c = int(ch) - 1
            if won:
                raise ValueError(f"game already won before ply {i + 1}")
            if not 0 <= c < WIDTH or not p.can_play(c):
                raise ValueError(f"illegal move {ch!r} at ply {i + 1}")
            won = p.is_winning_move(c)
            p = p.play(c)
        return p

    @classmethod
    def from_board(cls, grid: np.ndarray, ply: int) -> "Position":
        """grid: (6,7) mover-relative (+1 = side to move), row 0 = top. Must be gravity-valid."""
        pos = mask = 0
        for c in range(WIDTH):
            for r in range(HEIGHT):
                v = int(grid[HEIGHT - 1 - r, c])
                if v:
                    bit = 1 << (c * H1 + r)
                    mask |= bit
                    if v == 1:
                        pos |= bit
        return cls(pos, mask, ply)

    def can_play(self, c: int) -> bool:
        return (self.mask & TOP_MASKS[c]) == 0

    def play(self, c: int) -> "Position":
        pos = self.pos ^ self.mask            # stones of the player who just... becomes opponent
        mask = self.mask | (self.mask + BOT_MASKS[c])
        return Position(pos, mask, self.moves + 1)

    def is_winning_move(self, c: int) -> bool:
        return bool(self.winning_cells() & self.playable() & COL_MASKS[c])

    def playable(self) -> int:
        return (self.mask + BOTTOM) & BOARD_MASK

    def winning_cells(self) -> int:
        return winning_cells(self.pos, self.mask)

    def opponent_winning_cells(self) -> int:
        return winning_cells(self.pos ^ self.mask, self.mask)

    def key(self) -> int:
        return self.pos + self.mask  # unique per position (Pons' trick)

    def possible_non_losing(self) -> int:
        """Playable cells that don't hand the opponent an immediate win (0 = lost)."""
        possible = self.playable()
        opp_win = self.opponent_winning_cells()
        forced = possible & opp_win
        if forced:
            if forced & (forced - 1):
                return 0                       # two threats: cannot block both
            possible = forced                  # must block the single threat
        return possible & ~(opp_win >> 1)      # don't play under an opponent threat


class Solver:
    """backend: "auto" (C if a compiler is available, else Python), "c" or "python".
    All results are identical; the C core is just ~100x faster."""

    def __init__(self, node_limit: int | None = None, backend: str = "auto"):
        self.table: dict[int, tuple[int, int]] = {}
        self.nodes = 0
        self.node_limit = node_limit
        self._lib = _load_c() if backend in ("auto", "c") else None
        if backend == "c" and self._lib is None:
            raise RuntimeError("C backend requested but no working C compiler was found")
        self.backend = "c" if self._lib is not None else "python"

    def _search(self, p: Position, alpha: int, beta: int) -> int:
        if self._lib is None:
            return self._negamax(p, alpha, beta)
        import ctypes
        nodes, status = ctypes.c_longlong(0), ctypes.c_int(0)
        limit = 0 if self.node_limit is None else max(1, self.node_limit - self.nodes)
        r = self._lib.c4_negamax(p.pos, p.mask, p.moves, alpha, beta, limit,
                                 ctypes.byref(nodes), ctypes.byref(status))
        self.nodes += nodes.value
        if status.value:
            raise SearchLimit(f"exceeded {self.node_limit} nodes")
        return r

    # -- public API ----------------------------------------------------------------
    def solve(self, moves: str | Position, *, weak: bool = False) -> int:
        """Exact score for the side to move. `weak=True` only resolves win/draw/loss (-1/0/1)."""
        p = Position.from_moves(moves) if isinstance(moves, str) else moves
        if p.winning_cells() & p.playable():
            return 1 if weak else (WIDTH * HEIGHT + 1 - p.moves) // 2
        if weak:
            lo, hi = -1, 1
        else:
            lo = -((WIDTH * HEIGHT - p.moves) // 2)
            hi = (WIDTH * HEIGHT + 1 - p.moves) // 2
        # null-window binary search on the score
        while lo < hi:
            med = lo + (hi - lo) // 2
            if med <= 0 and _div2(lo) < med:
                med = _div2(lo)
            elif med >= 0 and _div2(hi) > med:
                med = _div2(hi)
            r = self._search(p, med, med + 1)
            if r <= med:
                hi = r
            else:
                lo = r
        return lo

    def move_scores(self, moves: str | Position, *, weak: bool = False) -> list[int | None]:
        """Exact score (from the mover's view) of each column; None for full columns."""
        p = Position.from_moves(moves) if isinstance(moves, str) else moves
        out: list[int | None] = []
        for c in range(WIDTH):
            if not p.can_play(c):
                out.append(None)
            elif p.is_winning_move(c):
                out.append(1 if weak else (WIDTH * HEIGHT + 1 - p.moves) // 2)
            else:
                out.append(-self.solve(p.play(c), weak=weak))
        return out

    def best_moves(self, moves: str | Position, *, weak: bool = False) -> list[int]:
        s = self.move_scores(moves, weak=weak)
        best = max(v for v in s if v is not None)
        return [c for c, v in enumerate(s) if v == best]

    # -- search --------------------------------------------------------------------
    def _negamax(self, p: Position, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            raise SearchLimit(f"exceeded {self.node_limit} nodes")

        possible = p.possible_non_losing()
        if possible == 0:
            return -((WIDTH * HEIGHT - p.moves) // 2)      # every move loses (or is forced to)
        if p.moves >= WIDTH * HEIGHT - 2:
            return 0                                       # board (almost) full: draw
        lo_bound = -((WIDTH * HEIGHT - 2 - p.moves) // 2)
        if alpha < lo_bound:
            alpha = lo_bound
            if alpha >= beta:
                return alpha
        hi_bound = (WIDTH * HEIGHT - 1 - p.moves) // 2
        if beta > hi_bound:
            beta = hi_bound
            if alpha >= beta:
                return beta

        key = p.key()
        entry = self.table.get(key)
        if entry is not None:
            kind, v = entry
            if kind == 1:                                   # stored upper bound
                if beta > v:
                    beta = v
                    if alpha >= beta:
                        return beta
            else:                                           # stored lower bound
                if alpha < v:
                    alpha = v
                    if alpha >= beta:
                        return alpha

        # order moves by number of own winning cells they create, centre first
        moves = []
        for idx, c in enumerate(ORDER):
            mv = possible & COL_MASKS[c]
            if mv:
                score = popcount(winning_cells(p.pos | mv, p.mask | mv))
                moves.append((score, -idx, mv))
        moves.sort(reverse=True)

        for _, _, mv in moves:
            child = Position(p.pos ^ p.mask, p.mask | mv, p.moves + 1)
            sc = -self._negamax(child, -beta, -alpha)
            if sc >= beta:
                self.table[key] = (2, sc)                   # lower bound
                return sc
            if sc > alpha:
                alpha = sc
        self.table[key] = (1, alpha)                        # upper bound
        return alpha


def score_to_text(score: int, moves: int) -> str:
    """Describe an exact score for the side to move, given `moves` stones already played."""
    if score == 0:
        return "draw with best play"
    s = abs(score)
    # the game ends on ply T with score s = (44 - T) // 2, so T is 43-2s or 44-2s;
    # the mover wins on a ply with T-moves odd, the opponent on one with T-moves even
    want_odd = score > 0
    T = next(t for t in (43 - 2 * s, 44 - 2 * s) if ((t - moves) % 2 == 1) == want_odd)
    if score > 0:
        return f"side to move wins on ply {T} ({(T - moves + 1) // 2} of its own moves from now)"
    return f"side to move loses on ply {T} (opponent needs {(T - moves) // 2} more moves)"
