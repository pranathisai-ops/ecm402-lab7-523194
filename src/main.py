"""Reproduces every figure/table of Lab 7:  python src/main.py

Seeds (all fixed):
  Part A : np.random.seed(0) sources; default_rng(1) Gaussian sources;
           default_rng(2) random rotations; FastICA random_state=0
  Part B-D: make_dataset seed=0 (SNR sweep also seed=0); KMeans random_state=0, n_init=10;
           GMM/Spectral random_state=0; D13 repeats use dataset seeds 100..109
Ground-truth labels are used ONLY for (i) colouring plots and (ii) ARI/NMI evaluation.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import get_features            # noqa: E402
import bss, dimred, clustering               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

_log_lines = []


def log(msg):
    print(msg)
    _log_lines.append(str(msg))


def main():
    np.random.seed(0)
    results = {}

    log("=" * 20 + " PART A: ICA / BSS " + "=" * 20)
    results.update(bss.run_partA(FIG, TAB, log))

    log("\n" + "=" * 20 + " PART B: PCA (modulation dataset, 20 dB) " + "=" * 20)
    Xs, y, _ = get_features(snr_db=20, seed=0)
    rb, Z2 = dimred.run_partB(Xs, y, FIG, TAB, log)
    results.update(rb)

    log("\n" + "=" * 20 + " PART C: k-means " + "=" * 20)
    r9, m4 = clustering.run_C9(Xs, y, TAB, log)
    results.update(r9)
    results.update(clustering.run_C10(Xs, y, FIG, TAB, log))
    results.update(clustering.run_C11(Xs, y, Z2, TAB, log))
    results.update(clustering.run_C12(FIG, TAB, log))

    log("\n" + "=" * 20 + " PART D: alternatives " + "=" * 20)
    results.update(clustering.run_D13(Xs, y, TAB, log))
    results.update(clustering.run_D14(FIG, log))

    (ROOT / "results" / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (ROOT / "results" / "log.txt").write_text("\n".join(_log_lines))
    print("\nDone. Figures -> results/figures, tables -> results/tables, log -> results/log.txt")


if __name__ == "__main__":
    main()
