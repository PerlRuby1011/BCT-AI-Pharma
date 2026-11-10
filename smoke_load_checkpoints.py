"""Checkpoint-loading smoke test (run before V2, on the machine that will run V2).

Exercises what compilation and the guard tests do not:
  * deserialization of all four checkpoints (torch state dicts, pickled IF + calibration);
  * reconstruction of the CNN architecture and loading of the weights into it;
  * weight / pickle SHA-256 against fingerprints.json;
  * probe outputs computed on THIS machine against the recorded probe hashes
    (published CNN on 512 rows; published IF on 2,000 readings);
  * calibration bounds inside the pickles against fingerprints.json;
  * a finite-output forward pass of the degraded CNN and IF (no recorded probe exists for them).
No retraining or replay. Writes a fresh timestamped recheck report by default;
--out selects a report path. Existing reports are never overwritten.
Exit 0 only if every check passes.

Usage:  python smoke_load_checkpoints.py [results/models_v1/session_1]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import run_v1_interception as v1  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", nargs="?", type=Path, default=Path("results/models_v1/session_1"))
    parser.add_argument("--out", type=Path, help="new report path; existing files are refused")
    args = parser.parse_args()
    report_path = args.out or Path(f"results/checkpoint_smoke_recheck_{time.time_ns()}.json")
    report_path = report_path if report_path.is_absolute() else REPO / report_path
    if report_path.exists():
        parser.error(f"refusing to overwrite existing report: {report_path}")
    mdir = args.checkpoint_dir
    mdir = mdir if mdir.is_absolute() else REPO / mdir
    log = v1.setup_logging(); config = v1.load_config()
    fp = json.loads((mdir / "fingerprints.json").read_text())
    checks = {}
    t0 = time.time()
    try:
        models, info = v1.load_models(mdir, config, log)      # deserialize, rebuild, hash, probe; SystemExit on mismatch
        checks["deserialize_rebuild_weights_probe"] = True
    except SystemExit as e:
        checks["deserialize_rebuild_weights_probe"] = False
        info = {"error": str(e)}; models = None
    if models is not None:
        for name in ("if_published", "if_degraded"):
            checks[f"{name}_calibration_matches"] = bool(np.allclose(
                list(models[f"{name}_calibration"]), fp[name]["calibration"], rtol=0, atol=0))
        X = v1._generate_normal_readings(200, np.random.default_rng(7))
        s = v1.normalized_anomaly_scores(models["if_degraded"], X, models["if_degraded_calibration"])
        checks["if_degraded_finite"] = bool(np.isfinite(s).all())
        import torch
        cfg = config["cnn"]; n = cfg["runtime_image_size"]
        with torch.no_grad():
            out = models["cnn_degraded"](torch.zeros(2, 3, n, n))   # (authenticity_logit, tamper_logits)
        outs = out if isinstance(out, (tuple, list)) else (out,)
        checks["cnn_degraded_finite"] = bool(len(outs) == 2 and all(torch.isfinite(o).all() for o in outs))
    import sklearn
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "sklearn": sklearn.__version__}
    try:
        import torch; env["torch"] = torch.__version__
    except Exception:
        pass
    ok = all(checks.values()) and len(checks) == 5
    rep = {"checkpoint_dir": str(mdir.relative_to(REPO)), "checks": checks, "all_passed": ok,
           "load_info": info, "environment": env, "runtime_s": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as report_file:
        report_file.write(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
