"""Guards that keep conference-reported reference values out of computed results.

How the separation works (release 1.1.0)
----------------------------------------
* The BCCA 2026 conference-reported values live only in
  ``config.yaml:bcca2026_reference_values`` and are read only by
  ``historical/bcca2026_reference.py``. Every record that module returns
  carries the string :data:`REFERENCE_MARKER`.
* :func:`find_reference_values` / :func:`assert_no_reference_values` detect
  that marker. They are called by the writers of computed results:
  ``evaluation.run_simulation.write_results``, ``orchestrate.save_json``,
  ``run_live_inference_experiment.py`` and
  ``pts/calibration_diagnostic.build_payload``.
* ``tests/test_reference_value_isolation.py`` additionally checks that no
  current module imports ``historical`` or names the reference section, and
  runs stub-based output tests asserting that pipeline outputs contain none
  of the reference values.

What the marker check detects
-----------------------------
The marker as a substring of: any ``str``; any dict key; any element of a
list, tuple or set; any column name of a ``pandas.DataFrame``; any cell of an
object-dtype DataFrame column; any value of an object-dtype ``Series``. It
also flags a dict key equal to the reference section name or its pre-1.1.0
name (``baseline_comparison``), which indicates a configuration section was
copied into results.

What it does not detect
-----------------------
A bare number taken from the reference section without its marker (for
example ``config[...]["BCT-AI"]["detection_rate_pct"]`` copied into a result
as a float), DataFrame index labels, ``Series.name``, NumPy arrays, and other
containers or objects. Those paths are covered instead by the static checks
(no current code can read the section) and by the stub-based output tests,
not by this runtime check.
"""
from __future__ import annotations

from typing import Any, List

import pandas as pd

#: Provenance tag attached to every record produced from the reference values.
REFERENCE_MARKER = "BCCA2026_CONFERENCE_REFERENCE__NOT_COMPUTED"

#: Configuration section holding the reference values.
REFERENCE_CONFIG_KEY = "bcca2026_reference_values"

#: Pre-1.1.0 name of that section; must not reappear in the current config.
LEGACY_CONFIG_KEY = "baseline_comparison"


class ReferenceValueLeak(ValueError):
    """Raised when conference reference values reach a computed-results output."""


def find_reference_values(obj: Any, path: str = "$") -> List[str]:
    """Return the paths at which ``obj`` contains reference-value material.

    Detects the provenance marker in any string, dict key, list element,
    DataFrame cell or column name, and the reference configuration key itself
    (which would mean a whole config section was copied into results).
    """
    hits: List[str] = []
    if isinstance(obj, str):
        if REFERENCE_MARKER in obj:
            hits.append(path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            if REFERENCE_MARKER in key_str or key_str in (REFERENCE_CONFIG_KEY, LEGACY_CONFIG_KEY):
                hits.append(f"{path}.{key_str}")
            hits.extend(find_reference_values(value, f"{path}.{key_str}"))
    elif isinstance(obj, (list, tuple, set)):
        for i, value in enumerate(obj):
            hits.extend(find_reference_values(value, f"{path}[{i}]"))
    elif isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            if REFERENCE_MARKER in str(col):
                hits.append(f"{path}.columns[{col}]")
        for col in obj.columns:
            if obj[col].dtype == object:
                if obj[col].astype(str).str.contains(REFERENCE_MARKER, regex=False).any():
                    hits.append(f"{path}[{col}]")
    elif isinstance(obj, pd.Series):
        if obj.dtype == object and obj.astype(str).str.contains(REFERENCE_MARKER, regex=False).any():
            hits.append(path)
    return hits


def assert_no_reference_values(obj: Any, context: str = "computed results") -> None:
    """Raise :class:`ReferenceValueLeak` if ``obj`` contains reference values."""
    hits = find_reference_values(obj)
    if hits:
        raise ReferenceValueLeak(
            f"{context} contain BCCA 2026 conference-reported reference values "
            f"(not computed results) at: {', '.join(hits[:10])}"
        )
