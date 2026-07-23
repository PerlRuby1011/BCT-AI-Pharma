"""End-to-end entry point for the BCT-AI-Pharma reproduction pipeline.

Usage:
    python main.py [--config config/config.yaml]

Runs transaction simulation, blockchain performance benchmarking, all three
AI modules (LSTM, CNN, Isolation Forest), Product Trust Score computation,
baseline comparison, and statistical validation, then writes
``results/simulation_results.csv`` and ``results/performance_metrics.json``.
"""
from __future__ import annotations

import argparse
import os
import time

from evaluation.run_simulation import run_full_pipeline, write_results
from utils import get_logger, load_config, set_global_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with a ``config`` path.
    """
    parser = argparse.ArgumentParser(description="Run the BCT-AI-Pharma simulation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("BCT_AI_CONFIG_PATH", "config/config.yaml"),
        help="Path to the YAML configuration file (default: $BCT_AI_CONFIG_PATH or config/config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete BCT-AI-Pharma pipeline and report a summary to stdout."""
    args = parse_args()
    config = load_config(args.config)
    set_global_seed(config.get("random_seed", 42))

    logger.info("Starting BCT-AI-Pharma end-to-end pipeline (seed=%s)", config.get("random_seed"))
    start = time.time()

    results = run_full_pipeline(config)
    write_results(results, config)

    elapsed = time.time() - start
    logger.info("Pipeline finished in %.1fs", elapsed)

    print("\n=== BCT-AI-Pharma Simulation Summary ===")
    print(f"Transactions simulated : {results['transaction_simulation']['n_transactions']}")
    print(f"Anomalies injected      : {results['transaction_simulation']['n_anomalies']}")
    print(f"LSTM  MAPE / R^2        : {results['lstm']['metrics']['mape']:.2f}% / "
          f"{results['lstm']['metrics']['r2']:.3f}")
    print(f"CNN weighted F1         : {results['cnn']['metrics']['weighted_avg']['f1']:.3f}")
    print(f"Isolation Forest detect : {results['isolation_forest']['metrics']['overall']['detection']:.3f}")
    counterfeit_ttest = results["statistical_validation"]["counterfeit_detection_ttest"]
    print(
        f"Counterfeit t-test      : t({counterfeit_ttest['degrees_of_freedom']}) = "
        f"{counterfeit_ttest['t_statistic']:.2f}, p = {counterfeit_ttest['p_value']:.4f}"
    )
    print(f"Results written to      : {config['paths']['results_dir']}/")


if __name__ == "__main__":
    main()
