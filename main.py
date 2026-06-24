"""Unified entry point for HNSD OOD and misclassification experiments."""

from __future__ import annotations

import json

from hnsd.config import build_parser
from hnsd.trainer import run_experiment


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(args)
    best = result["best"]
    metrics = best["test"]
    print("\n========== FINAL RESULT ==========")
    print(f"dataset: {result['dataset']}")
    print(f"task: {result['task']}")
    print(f"selected epoch by validation AUROC: {best['epoch']}")
    print(f"test accuracy: {100 * best['test_accuracy']:.2f}")
    print(f"test AUROC: {100 * metrics['auroc']:.2f}")
    print(f"test AUPR succ/in: {100 * metrics['aupr_pos']:.2f}")
    print(f"test AUPR err/out: {100 * metrics['aupr_neg']:.2f}")
    print(f"test FPR95: {100 * metrics['fpr95']:.2f}")
    print("FINAL_JSON=" + json.dumps(result["best"], allow_nan=True))


if __name__ == "__main__":
    main()

