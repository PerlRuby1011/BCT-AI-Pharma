# Changes Log

## Phase 2 — realistic distributions (2026-07-27T22:02:02)
- Temperature reading std dev: 1.2C -> 0.8C (N(5.0, 0.8) cold-chain distribution)
- Counterfeit injection: uniform random -> clustered by manufacturer (30% of manufacturer nodes treated as a compromised cluster)
- Custody transfer timing: uniform arrival -> exponential(lambda=0.1) inter-arrival
Rationale: Realistic pharmaceutical distributions improve model generalization: uniform placement/timing is an unrealistic simplification of real supply-chain data.

## Phase 3 — IF contamination 0.10->0.05 (2026-07-27T22:02:13)
- isolation_forest.contamination: 0.10 -> 0.05
Rationale: Matches the true anomaly injection rate of 5% (simulation.anomaly_rate).

## Phase 4 — best hyperparams (grid search) (2026-07-27T22:03:56)
- LSTM hyperparameters: grid search over units_layer1, units_layer2, dropout, learning_rate (winning combination recorded below)
- Grid search winner: {'units_layer1': 96, 'units_layer2': 48, 'dropout': 0.2, 'learning_rate': 0.001}
Rationale: Grid search selects the architecture/optimizer configuration that minimizes validation MAPE on held-out synthetic sequences.

## Phase 5 — CNN overlap 0.18->0.10 (2026-07-27T22:14:48)
- cnn.overlap_factor: 0.18 -> 0.10
Rationale: Lower synthetic class-overlap reduces label noise in the packaging-verification training data, improving CNN discriminability.

## Phase 6 — all improvements combined (2026-07-27T22:24:51)
- All Phase 2-5 improvements applied simultaneously
Rationale: Combined ablation: measures the total effect and any interaction (redundancy or synergy) between the four individually-validated improvements.
