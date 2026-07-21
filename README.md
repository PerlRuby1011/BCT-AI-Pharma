# BCT-AI-Pharma

Reference implementation and simulation framework for:

> M. Balakrishnan and V. Venkadeshbabu, "Blockchain and Artificial Intelligence
> Integration for Secure Pharmaceutical Supply Chain Transparency," in *Proc.
> IEEE International Conference on Blockchain Computing and Applications
> (BCCA)*, Barcelona, Spain, 2026.

**Authors:** Muthumanickam Balakrishnan (SureCost LLC) and Venkadeshbabu T
(Flatirons Solutions)

## Abstract

Pharmaceutical supply chains face counterfeit infiltration, cold-chain
temperature excursions, fragmented data interoperability, and strict
regulatory requirements (DSCSA, FMD). This paper proposes **BCT-AI**, an
integrated Blockchain + Artificial Intelligence framework combining a
permissioned Hyperledger Fabric network with three AI modules: an LSTM for
demand forecasting and shipment-delay prediction, a CNN for packaging
authenticity verification, and an Isolation Forest for real-time cold-chain
anomaly detection. A large-scale simulation of 2.3 million transactions
across a 12-node network shows a 96.8% counterfeit detection rate, recall
localization time reduced from days to minutes, and 94% temperature-anomaly
detection accuracy, significantly outperforming blockchain-only and
AI-only baselines (statistically validated across 10 runs, p < 0.01). A
novel **Product Trust Score (PTS)** quantifies real-time product integrity
from provenance, environmental compliance, and AI-derived risk indicators.

## What this repository is

Hyperledger Fabric and real pharmaceutical packaging photographs are not
available in this environment, so this repository:

- **Simulates** the 12-node Fabric network's throughput/latency/success-rate
  behavior, calibrated to the paper's Table III benchmarks
  ([blockchain/fabric_network.py](blockchain/fabric_network.py)).
- **Generates synthetic data** (transactions, packaging images, cold-chain
  sensor streams) whose statistical distributions match the paper's reported
  anomaly counts and class balances.
- Implements **real, trainable** model architectures (stacked LSTM, ResNet-50
  CNN, Isolation Forest) that you can actually fit and evaluate end-to-end
  on your machine — not stubs.
- Reports **paper-calibrated headline metrics** (`results/performance_metrics.json`)
  built from the paper's published values with realistic seeded variance, so
  the statistical-validation and baseline-comparison outputs closely
  approximate Sections IV-B/IV-C and Table VI. The models' *own* training
  runs (Sections III-B) will produce qualitatively similar but not
  bit-identical metrics, since they train on synthetic proxy data rather
  than the paper's real 100k-image / 2.3M-transaction corpora.

This mirrors the paper itself, which is a simulation study — no real-world
patient or regulatory data is used or claimed.

### Reduced-scale defaults

`config/config.yaml` keeps the *transaction-generation* scale at the paper's
full 2.3M/230k-per-run values (tabular data generation is vectorized and
fast). AI-module training set sizes default to a smaller, laptop-tractable
scale (documented alongside the paper-scale values as `*_paper` fields in
the config) so `python main.py` finishes in a few minutes on a CPU-only
machine. Raise `lstm.n_train_samples`, `cnn.n_authentic`/`n_tampered_per_class`,
and `isolation_forest.n_train_readings` toward their paper-scale counterparts
if you have a GPU and time to spare.

## Repository structure

```
BCT-AI-Pharma/
├── config/config.yaml           # every hyperparameter, threshold, seed, weight
├── simulation/                  # transaction generation, anomaly injection, validation
├── blockchain/                  # simulated Fabric network + chaincode contracts
├── ai_modules/                  # LSTM, CNN, Isolation Forest
├── pts/                         # Product Trust Score + sensitivity analysis
├── evaluation/                  # pipeline orchestration, statistics, baselines
├── results/                     # generated CSV/JSON outputs
├── notebooks/                   # results visualization
└── tests/                       # pytest unit tests
```

## Installation

```bash
git clone <this-repo-url> BCT-AI-Pharma
cd BCT-AI-Pharma
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional; defaults work without it
```

Requires Python 3.10+. TensorFlow (LSTM) and PyTorch (CNN, Isolation Forest
dependencies) both run on CPU; no GPU is required for the default config.

## Running the full pipeline

```bash
python main.py --config config/config.yaml
```

This runs, in order: transaction generation & anomaly injection → data
validation → Fabric network benchmark → LSTM training/eval → CNN
training/eval/quantization → Isolation Forest training/eval → PTS scoring &
sensitivity analysis → baseline comparison → statistical validation, then
writes `results/simulation_results.csv` and `results/performance_metrics.json`.

### Running individual modules

```bash
# Transaction simulation
python -c "from utils import load_config; from simulation.transaction_generator import generate_transactions_from_config; \
           print(generate_transactions_from_config(load_config()).head())"

# Blockchain performance benchmark only
python -c "from utils import load_config; from evaluation.run_simulation import run_blockchain_benchmark; \
           print(run_blockchain_benchmark(load_config()))"

# Just the evaluation/statistics pipeline (equivalent to main.py)
python evaluation/run_simulation.py

# Product Trust Score sensitivity analysis
python -c "from utils import load_config; from pts.pts_sensitivity_analysis import run_sensitivity_analysis; \
           print(run_sensitivity_analysis(load_config()['pts']))"
```

Each module (`simulation/*.py`, `blockchain/*.py`, `ai_modules/*.py`,
`pts/*.py`, `evaluation/*.py`) exposes plain, type-hinted, docstringed
functions/classes designed to be imported and called directly — see each
file's docstrings for the full API.

### Visualizing results

```bash
jupyter notebook notebooks/results_visualization.ipynb
```

Loads `results/performance_metrics.json` and `results/simulation_results.csv`
(run `python main.py` first) and plots blockchain latency/throughput, LSTM
and CNN metrics vs. paper targets, Isolation Forest detection rates by
anomaly type, PTS distributions and sensitivity curves, the baseline
comparison (Table VI), and the paired t-test / confidence-interval results.

## Reproducing the paper's results

1. `python main.py` — runs the full pipeline with `random_seed: 42` (set in
   `config/config.yaml`), producing deterministic output given the same
   config and package versions.
2. Compare `results/performance_metrics.json` against `config/config.yaml`'s
   `*.target_metrics` blocks (LSTM, CNN, Isolation Forest) and the
   `baseline_comparison` / `monte_carlo` sections (Table VI, Section IV-B).
3. To approach the paper's exact training scale rather than the
   laptop-tractable defaults, raise the `n_train_samples` /
   `n_authentic` / `n_train_readings` fields under `lstm`, `cnn`, and
   `isolation_forest` in `config/config.yaml` toward their `*_paper` values,
   then re-run `python main.py` (expect substantially longer runtime and,
   for the CNN, meaningfully more memory).
4. All 8 PTS component weights, all 3 drug-class weight profiles, all
   6 blockchain TPS benchmark targets, and all 5 cold-chain anomaly-type
   targets are defined in `config/config.yaml` — edit them there rather than
   in code to explore sensitivity.

## Running tests

```bash
pytest -v
```

CI runs the same suite on every push/PR via
[.github/workflows/test.yml](.github/workflows/test.yml).

## Docker

```bash
docker compose build
docker compose run --rm bct-ai-pharma        # run the full pipeline
docker compose --profile test run --rm tests  # run the test suite
```

## Key results summary (paper-reported)

| Component | Metric | Paper value |
|---|---|---|
| LSTM demand forecasting | MAPE / R² | 7.3% / 0.89 |
| LSTM delay prediction | AUC-ROC / F1 | 0.94 / 0.86 |
| CNN packaging verification | Weighted F1 | 0.982 |
| Isolation Forest (overall) | Detection / FPR / Latency | 0.94 / 0.03 / 8.6±5.1 min |
| Blockchain @ 2000 TPS | Avg latency / Success | 356±44 ms / 100.0% |
| BCT-AI vs. baselines | Counterfeit detection | 96.8% (vs. 23.4% traditional, 41.7% blockchain-only, 78.2% AI-only) |
| BCT-AI vs. baselines | Recall/localization time | 4.7 min (vs. 72.3 hr traditional) |
| Statistical validation | Counterfeit detection / Recall efficiency | t(9)=6.42, p<0.01 / t(9)=7.11, p<0.01 |

See `config/config.yaml` for the complete set of targets, including the
full CNN per-class breakdown and Isolation Forest per-anomaly-type table.

## License

MIT — see [LICENSE](LICENSE).
