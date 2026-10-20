# c4shap

Shapley-value explanations for the `steven0226/alphazero-connect4` policy/value network.

    pip install -e ".[az,dev]"
    pip install git+https://github.com/kuotunyu/alphazero-connect4   # provides the `az` package

    c4shap 4453 --demo                         # toy model, no weights
    c4shap 4453 --groups columns --out x.png   # real model, exact Shapley over columns
    c4shap 44533122116655 --groups cells       # per-piece, permutation sampling
    c4shap 4453 --target value                 # explain the value head instead
    pytest -q                                  # C4SHAP_REAL=1 also runs the encoding check

Players are columns (<= 7, exact) or individual pieces (sampled). A removed piece is
deleted and the rest of its column falls (`--no-settle` leaves floating pieces).
This explains the raw network, not the MCTS search on top of it.
