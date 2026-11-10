"""Checks that conference-reported reference values cannot silently enter
computed results (release 1.1.0).

Three layers:
  1. Static: no current-evaluation module imports ``historical`` or reads the
     reference configuration section, and the removed helpers are gone from
     the current code.
  2. Configuration: the legacy ``baseline_comparison`` section is gone and the
     reference section is labelled.
  3. Runtime: every output of the historical module is tagged, the guard
     detects the tag anywhere in a nested result, and the results writers
     refuse to write tagged material.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.result_guards import (
    LEGACY_CONFIG_KEY,
    REFERENCE_CONFIG_KEY,
    REFERENCE_MARKER,
    ReferenceValueLeak,
    assert_no_reference_values,
    find_reference_values,
)
from historical import bcca2026_reference as ref
from utils import load_config

REPO = Path(__file__).resolve().parents[1]

# Files allowed to name the reference section: the historical module itself,
# the guard (which defines the key names), and tests.
ALLOWED = {
    REPO / "historical" / "bcca2026_reference.py",
    REPO / "historical" / "__init__.py",
    REPO / "evaluation" / "result_guards.py",
}
REMOVED_NAMES = {
    "build_baseline_comparison_table",
    "simulate_per_run_detection_rates",
    "compute_improvement_ratios",
    "compute_relative_improvement",
    "AI_ONLY_BASELINE_PCT",
}
# Output/config keys that carried conference-reported values up to 1.0.x.
REFERENCE_KEYS = {"target_metrics", "paper_reference", "real_world_range", "ai_only_baseline_pct"}


def _current_python_files():
    skip_dirs = {"tests", "historical", ".venv", "venv", ".git", "results", "__pycache__", "notebooks"}
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO)
        if rel.parts and rel.parts[0] in skip_dirs:
            continue
        if any(part in skip_dirs for part in rel.parts):
            continue
        yield path


# --------------------------------------------------------------------------- 1. static
@pytest.mark.parametrize("path", sorted(_current_python_files()), ids=lambda p: str(p.relative_to(REPO)))
def test_current_code_does_not_use_reference_values(path: Path) -> None:
    if path in ALLOWED:
        return
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "historical":
                    problems.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "historical":
                problems.append(f"line {node.lineno}: from {node.module} import ...")
            for alias in node.names:
                if alias.name in REMOVED_NAMES:
                    problems.append(f"line {node.lineno}: imports removed helper {alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in (REFERENCE_CONFIG_KEY, LEGACY_CONFIG_KEY):
                problems.append(f"line {node.lineno}: reads config section '{node.value}'")
            if node.value in REFERENCE_KEYS:
                problems.append(f"line {node.lineno}: uses reference key '{node.value}'")
        elif isinstance(node, (ast.FunctionDef, ast.Name, ast.Attribute)):
            name = getattr(node, "name", None) or getattr(node, "id", None) or getattr(node, "attr", None)
            if name in REMOVED_NAMES:
                problems.append(f"line {node.lineno}: uses removed helper {name}")
    assert not problems, f"{path.relative_to(REPO)} touches reference values: {problems}"


def test_notebook_does_not_plot_reference_values() -> None:
    nb = json.loads((REPO / "notebooks" / "results_visualization.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for key in (LEGACY_CONFIG_KEY, REFERENCE_CONFIG_KEY, *REFERENCE_KEYS):
            assert key not in src, (key, src[:200])


def test_full_pipeline_no_longer_returns_reference_sections() -> None:
    from evaluation import run_simulation

    src = inspect.getsource(run_simulation.run_full_pipeline)
    for key in ('"baseline_comparison"', '"improvement_ratios"', "relative_improvement"):
        assert key not in src
    assert "assert_no_reference_values" in inspect.getsource(run_simulation.write_results)


# --------------------------------------------------------------------------- 2. config
def test_config_reference_section_is_renamed_and_labelled() -> None:
    config = load_config()
    assert LEGACY_CONFIG_KEY not in config, "legacy section would let old code read the values silently"
    section = config[REFERENCE_CONFIG_KEY]
    assert "not computed" in section["provenance"]
    assert section["BCT-AI"]["detection_rate_pct"] == 96.8
    assert section["AI-Only"]["detection_rate_pct"] == 78.2


def test_config_inputs_hold_no_conference_values() -> None:
    config = load_config()
    for section in ("lstm", "cnn", "isolation_forest"):
        assert "target_metrics" not in config[section]
    assert set(config["monte_carlo"]) == {"sensor_drift"}
    assert set(config["monte_carlo"]["sensor_drift"]) == {"perturbation_celsius"}
    assert set(config["statistical_validation"]) == {"n_runs", "confidence_level"}
    section = config[REFERENCE_CONFIG_KEY]
    assert section["cnn_metrics"]["weighted_avg"]["f1"] == 0.982
    assert section["statistical_tests"]["counterfeit_detection_ttest"]["t_statistic"] == 6.42
    assert section["monte_carlo"]["real_world_range"] == [0.91, 0.95]


def test_legacy_access_fails_loudly() -> None:
    config = load_config()
    with pytest.raises(KeyError):
        config[LEGACY_CONFIG_KEY]


# --------------------------------------------------------------------------- 3. runtime
def test_historical_table_preserved_and_tagged() -> None:
    table = ref.build_reference_table(load_config())
    assert list(table["method"]) == ["Traditional", "Blockchain-Only", "AI-Only", "BCT-AI"]
    assert list(table["detection_rate_pct"]) == [23.4, 41.7, 78.2, 96.8]
    assert (table["provenance"] == REFERENCE_MARKER).all()


def test_historical_jitter_algorithm_unchanged_and_tagged() -> None:
    config = load_config()
    samples = ref.simulate_per_run_detection_rates(config, n_runs=10, seed=42)
    # Re-implement the 1.0.x algorithm independently and compare exactly.
    rng = np.random.default_rng(42)
    expected = []
    for rate in (23.4, 41.7, 78.2, 96.8):
        expected.extend(np.clip(rng.normal(rate, max(0.5, rate * 0.01), size=10), 0, 100))
    assert np.array_equal(samples["detection_rate_pct"].to_numpy(), np.array(expected))
    assert (samples["provenance"] == REFERENCE_MARKER).all()


@pytest.mark.parametrize("producer", [
    lambda c: ref.build_reference_table(c),
    lambda c: ref.simulate_per_run_detection_rates(c, n_runs=3),
    lambda c: ref.compute_improvement_ratios(c),
    lambda c: ref.compute_relative_improvement(c),
    lambda c: ref.reference_recall_localization(c),
    lambda c: ref.reference_component_metrics(c),
    lambda c: ref.reference_statistical_tests(c),
    lambda c: ref.reference_monte_carlo(c),
    lambda c: ref.reference_ai_only_detection_pct(c),
    lambda c: ref.build_reference_table(c).to_dict(orient="records"),
    lambda c: {"nested": [{"deeper": ref.compute_improvement_ratios(c)}]},
    lambda c: {"bcca2026_reference_values": c[REFERENCE_CONFIG_KEY]},
])
def test_guard_detects_reference_material(producer) -> None:
    obj = producer(load_config())
    assert find_reference_values(obj)
    with pytest.raises(ReferenceValueLeak):
        assert_no_reference_values(obj)


def test_guard_accepts_computed_results() -> None:
    clean = {"bct_ai_counterfeit_detection": 0.714, "per_seed": [{"seed": 42, "detection": 0.71}],
             "table": pd.DataFrame({"method": ["BCT-AI"], "detection_rate_pct": [71.4]})}
    assert_no_reference_values(clean)


def test_write_results_refuses_reference_values(tmp_path: Path) -> None:
    from evaluation.run_simulation import write_results

    config = load_config()
    config["paths"]["results_dir"] = str(tmp_path)
    config["paths"]["simulation_results_csv"] = str(tmp_path / "sim.csv")
    config["paths"]["performance_metrics_json"] = str(tmp_path / "metrics.json")
    results = {"simulation_results_table": pd.DataFrame({"a": [1]}),
               "baseline": ref.build_reference_table(config).to_dict(orient="records")}
    with pytest.raises(ReferenceValueLeak):
        write_results(results, config)
    assert not (tmp_path / "metrics.json").exists()

    clean = {"simulation_results_table": pd.DataFrame({"a": [1]}), "x": 1.0}
    write_results(clean, config)
    assert (tmp_path / "metrics.json").exists()
