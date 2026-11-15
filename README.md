# c4shap

Exact Shapley values for the `steven0226/alphazero-connect4` network: which columns (or pieces)
made it want this move?

    pip install -e ".[az,dev]"
    pip install git+https://github.com/kuotunyu/alphazero-connect4     # the `az` package

    c4shap check                          # is the network being fed correctly? (run once)
    c4shap 152637                         # explain the network's favourite move
    c4shap 152637 --move 3 --out a.png    # explain column 3 instead, save a heatmap
    c4shap 152637 --target value          # explain the value head instead
    c4shap 4453 --players cells           # one player per piece (up to 16 pieces)

Move strings are columns 1-7, first player first. Replace the digits with your own game; the
side to move is the one explained.

## How it works (the whole idea is in `explain.py`)

1. **Players.** Each column's pieces are one player (<= 7 players). Or each piece, for small boards.
2. **Coalitions.** For every subset S of players, build the board that keeps S's pieces and
   removes everyone else's. That is 2^n boards (128 for 7 columns); one batched network call.
3. **Value.** v(S) = the probability the network gives the explained move on that board
   (or its value head with `--target value`).
4. **Shapley value.** For player i: average over all orders of adding players one by one of
   how much v rises when i joins (`shapley_values`, 8 lines). It is exact, not sampled.

Because it is exact, the values always sum to `v(all pieces) - v(empty board)`, and the table
prints both so you can see it.

## Files

| file | what |
|---|---|
| `board.py` | move string -> 6x7 grid, always from the side-to-move's view |
| `model.py` | the network: loads it and turns a board into its input |
| `explain.py` | players, coalitions, exact Shapley |
| `viz.py` | heatmap |
| `cli.py` | the `c4shap` command |

## About the network input

The network's exact input layout isn't documented, so `AZModel` tries the plausible encodings
on 7 positions where the right move is forced (take the win / block the threat), keeps the one
that plays correctly, and raises if none does. `c4shap check` shows what it picked. This is
the part that has never been run against the real weights, so check it first.

Removed pieces: the rest of their column falls (`settle_columns`) so every board is legal.
For column players nothing falls, because whole columns are removed.
