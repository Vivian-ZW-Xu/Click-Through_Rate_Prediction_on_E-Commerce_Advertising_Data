#!/usr/bin/env python3
"""Run the fixed validation-only hyperparameter search for logistic regression."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "src" / "08_train_logistic_regression.py"
RESULT_DIR = ROOT / "results" / "lr_tuning"
PREDICTION_ROOT = ROOT / "artifacts" / "predictions"
DEFAULT_ALPHAS = (1e-7, 1e-6, 1e-5)
DEFAULT_ETA0S = (0.001, 0.003, 0.005, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune LR on validation only; the frozen test split is never evaluated."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--eta0s", type=float, nargs="+", default=DEFAULT_ETA0S)
    parser.add_argument(
        "--logloss-tolerance",
        type=float,
        default=0.0002,
        help=(
            "Configurations within this distance of the best validation LogLoss "
            "form the near-optimal set; highest AUC wins within that set."
        ),
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional protocol test limit. Omit for the complete search.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 10_000:
        parser.error("--batch-size must be at least 10,000")
    if any(value <= 0 for value in [*args.alphas, *args.eta0s]):
        parser.error("All alpha and eta0 values must be positive")
    if args.logloss_tolerance < 0:
        parser.error("--logloss-tolerance cannot be negative")
    if args.max_configs is not None and args.max_configs < 1:
        parser.error("--max-configs must be positive")
    return args


def scientific_token(value: float) -> str:
    return f"{value:.0e}".replace("+", "p").replace("-", "m")


def decimal_token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("+", "p").replace("-", "m")


def run_name(alpha: float, eta0: float) -> str:
    return f"lr_tune_f123_a{scientific_token(alpha)}_e{decimal_token(eta0)}"


def read_validation_metrics(name: str) -> dict[str, object]:
    path = ROOT / "results" / name / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training completed without expected metrics: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1 or rows[0]["split"] != "val":
        raise RuntimeError(
            f"{name} must contain exactly one validation row during tuning; found {rows}"
        )
    row = rows[0]
    metrics_json_path = ROOT / "results" / name / "metrics.json"
    metrics_document = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    test_prediction = PREDICTION_ROOT / name / "test_predictions.parquet"
    if test_prediction.exists() or metrics_document.get("test") is not None:
        raise RuntimeError(f"Frozen test was evaluated during tuning run {name}")
    return {
        "best_epoch": int(metrics_document["best_epoch"]),
        "val_rows": int(row["rows"]),
        "val_auc": float(row["auc"]),
        "val_logloss": float(row["log_loss"]),
        "val_pr_auc": float(row["pr_auc"]),
        "val_brier_score": float(row["brier_score"]),
        "val_mean_prediction": float(row["mean_prediction"]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    configurations = list(itertools.product(args.alphas, args.eta0s))
    if args.max_configs is not None:
        configurations = configurations[: args.max_configs]

    print("=" * 78)
    print("VALIDATION-ONLY LOGISTIC REGRESSION SEARCH")
    print("=" * 78)
    print("Feature set:       f1_f2_f3")
    print(f"Configurations:    {len(configurations)}")
    print(f"Epochs per config: {args.epochs}")
    print("Frozen test:       disabled")

    results: list[dict[str, object]] = []
    for index, (alpha, eta0) in enumerate(configurations, start=1):
        name = run_name(alpha, eta0)
        metrics_path = ROOT / "results" / name / "metrics.csv"
        print("\n" + "-" * 78)
        print(
            f"CONFIG {index}/{len(configurations)}: "
            f"alpha={alpha:g}, eta0={eta0:g}, run={name}"
        )
        print("-" * 78)
        started = time.perf_counter()
        if metrics_path.exists():
            print("Existing completed run found; validating and reusing it.")
        else:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--run-name",
                name,
                "--feature-set",
                "f1_f2_f3",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--alpha",
                str(alpha),
                "--eta0",
                str(eta0),
                "--seed",
                str(args.seed),
            ]
            # Deliberately no --evaluate-test flag.
            subprocess.run(command, cwd=ROOT, check=True)

        metrics = read_validation_metrics(name)
        elapsed = time.perf_counter() - started
        result = {
            "run_name": name,
            "alpha": alpha,
            "eta0": eta0,
            "epochs_requested": args.epochs,
            **metrics,
            "orchestration_seconds": elapsed,
            "selected": False,
        }
        results.append(result)
        print(
            f"Result: LogLoss={result['val_logloss']:.6f}, "
            f"AUC={result['val_auc']:.6f}, "
            f"best_epoch={result['best_epoch']}"
        )

    best_logloss = min(float(row["val_logloss"]) for row in results)
    near_optimal = [
        row
        for row in results
        if float(row["val_logloss"]) <= best_logloss + args.logloss_tolerance
    ]
    selected = sorted(
        near_optimal,
        key=lambda row: (-float(row["val_auc"]), float(row["val_logloss"])),
    )[0]
    selected["selected"] = True
    results.sort(key=lambda row: (float(row["val_logloss"]), -float(row["val_auc"])))
    write_csv(RESULT_DIR / "search_results.csv", results)

    selection = {
        "selection_split": "validation",
        "test_evaluated": False,
        "feature_set": "f1_f2_f3",
        "rule": (
            "highest AUC among configurations within logloss_tolerance of "
            "the best validation LogLoss"
        ),
        "logloss_tolerance": args.logloss_tolerance,
        "best_validation_logloss": best_logloss,
        "near_optimal_configurations": len(near_optimal),
        "selected_run": selected["run_name"],
        "selected_alpha": selected["alpha"],
        "selected_eta0": selected["eta0"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_validation_auc": selected["val_auc"],
        "selected_validation_logloss": selected["val_logloss"],
        "grid": {
            "alphas": list(args.alphas),
            "eta0s": list(args.eta0s),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }
    (RESULT_DIR / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("LR SEARCH COMPLETED")
    print("=" * 78)
    print(f"Best validation LogLoss: {best_logloss:.6f}")
    print(f"Near-optimal configs:    {len(near_optimal)}")
    print(f"Selected run:            {selected['run_name']}")
    print(f"Selected alpha:          {selected['alpha']:g}")
    print(f"Selected eta0:           {selected['eta0']:g}")
    print(f"Selected val AUC:        {selected['val_auc']:.6f}")
    print(f"Selected val LogLoss:    {selected['val_logloss']:.6f}")
    print(f"Results:                 {RESULT_DIR}")


if __name__ == "__main__":
    main()
