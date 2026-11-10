# c4shap

Explanations for Connect 4 AIs: Shapley values for the `steven0226/alphazero-connect4`
policy/value network, a **perfect solver** to play and check against, and an experimental
**group-based explainer** that credits stone groups (a three-in-a-row, a gapped three...)
for a recommended move.

    pip install -e ".[az,dev]"
    pip install git+https://github.com/kuotunyu/alphazero-connect4   # provides the `az` package

## Quick start

    c4shap solve 7356536662322                  # exact scores per column (perfect play)
    c4shap 336224143 --explainer groups --out poisoned.png
    c4shap doctor                               # check the AlphaZero net against the solver
    c4shap 4453 --demo                          # Shapley over columns, toy model, no weights
    c4shap 4453 --model solver --target value   # Shapley of the perfect player's value
    c4shap 44533122116655 --groups cells        # per-piece, permutation sampling
    pytest -q                                   # C4SHAP_REAL=1 also runs the real-weights checks

Move strings use columns 1-7, first player first. The side to move is always the one explained.

## 1. The AI: what was fixed

`AZModel` used to contain `VERIFY` guesses (network class, plane order, row orientation, side
plane, output order). It now **checks itself**: `AZModel()` runs a short battery of
solver-labelled positions through every plausible encoding (plane order, row flip, side-to-move
plane, absolute vs mover-relative colours, 2- or 3-plane input), keeps the best one, and raises
with a report if none beats chance, instead of silently explaining a mis-fed network.
`c4shap doctor` prints that report. `_build_net` now discovers the network class in `az.model`
and filters the config to its constructor, and output heads are recognised by shape, so
`(policy, value)` and `(value, policy)` both work.

Also fixed: importing `c4shap` no longer requires `shap` (it is imported inside `explain`).

> Not verified here: I had no network access, so I could not load the real weights or the `az`
> package. The calibration is what protects you; run `c4shap doctor` once and
> `C4SHAP_REAL=1 pytest -k real` to confirm.

## 2. The perfect alternative: `c4shap.solver`

Exact game-theoretic solver (bitboards, negamax + alpha-beta, transposition table, forced-move
pruning, threat-based move ordering, the approach behind Pascal Pons' solver).

    from c4shap import Solver, SolverModel
    Solver().move_scores("7356536662322")   # [-8, -11, -11, 1, -2, -11, -11]
    SolverModel()                           # same interface as AZModel/ToyModel -> use with explain()

Scores follow Pons: `> 0` win (`22 - stones the winner plays`), `0` draw, `< 0` loss.

* A small C core is compiled on first use if `cc`/`gcc`/`clang` exists; otherwise the identical
  pure-Python search runs (~100x slower). `C4SHAP_NO_C=1` forces Python.
* Verified against an independent grid minimax (exact scores on random endgames) and
  C == Python on random positions.
* **Limits.** With the C core every position with 5+ stones solves in seconds (usually
  milliseconds), but positions with 4 or fewer stones can take hours (no opening book is
  included). `SolverModel` therefore raises `SearchLimit` past `node_limit` rather than hanging.
  Shapley coalitions are often that sparse, so the CLI's `--model solver` caps the search and
  uses the toy model for boards it can't finish (and tells you how many).

## 3. Experimental: the group explainer (`c4shap.groups`)

    from c4shap import Board, explain_groups
    print(explain_groups(Board.from_moves("152537")).table())

A **group** is the set of same-colour stones sharing one still-live four-cell window (no enemy
stone in it): a contiguous three, or a gapped `X X . X`. A group with `k` stones is worth
`{2: 1, 3: 10} x reach`, where reach is `1/(1+gap)` for each empty cell it still needs (gap =
empty cells below it; 0 = playable now). A move's score is the sum, over groups, of the
change in that group's value (blocking an opponent group counts for the mover), plus three
one-reply tactical overlays: **completes four** `+1000`, **opponent can win next move**
`-1000` (split into "leaves unblocked" / "enables opp win"), **double threat** `+500`.
Because the score *is* the sum of per-group items, the attribution is exact by construction.

`examples/README.md` shows 12 positions with images (regenerate: `python examples/make_examples.py`):
horizontal/vertical/gapped/diagonal threes, a forced block, an open-two double threat, a poisoned
square, plus four positions where the solver sees what groups can't. `c4shap.positions.POSITIONS`
holds them with solver-verified answers.

**How good is it?** Against the exact solver on ~12,000 random positions with 13-24 stones, its
top move was solver-optimal 85% of the time (and kept a winning move in 91% of won positions).
It is a one-move-ahead heuristic: it misses parity/zugzwang plans and deep combinations
(`quiet_only_move`, `late_squeeze` in the examples are real failures). Treat it as an
explanation vocabulary, not an engine.

## Notes

Shapley players are columns (<= 7, exact) or pieces (sampled). A removed piece is deleted and
the rest of its column falls (`--no-settle` leaves floating pieces; not allowed for the
solver). This explains the raw network, not any search on top of it.
