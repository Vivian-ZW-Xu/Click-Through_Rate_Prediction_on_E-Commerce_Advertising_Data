#!/usr/bin/env python3
"""Run a small validation-only search for Wide & Deep and DeepFM."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


TRAIN_SCRIPT = Path("src/16_train_tabular_nn.py")
SUMMARY_PATH = Path("results/nn_tuning_summary.csv")

CONFIGS = [
    {
        "model": "wide_deep",
        "run_name": "wide_deep_f123_lr1em3",
        "learning_rate": "0.001",
    },
    {
        "model": "wide_deep",
        "run_name": "wide_deep_f123_lr5em4",
        "learning_rate": "0.0005",
    },
    {
        "model": "deepfm",
        "run_name": "deepfm_f123_lr1em3",
        "learning_rate": "0.001",
    },
    {
        "model": "deepfm",
        "run_name": "deepfm_f123_lr5em4",
        "learning_rate": "0.0005",
    },
]


def completed(run_name: str) -> bool:
    result_dir = Path("results") / run_name
    return (
        (result_dir / "metrics.csv").exists()
        and (result_dir / "run_manifest.json").exists()
    )


def run_configuration(config: dict[str, str], index: int) -> None:
    run_name = config["run_name"]
    print("\n" + "=" * 78, flush=True)
    print(
        f"CONFIG {index}/{len(CONFIGS)}: {run_name}",
        flush=True,
    )
    print("=" * 78, flush=True)

    if completed(run_name):
        print("Existing completed result found; skipping training.", flush=True)
        return

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        config["model"],
        "--run-name",
        run_name,
        "--epochs",
        "4",
        "--patience",
        "1",
        "--batch-size",
        "8192",
        "--learning-rate",
        config["learning_rate"],
        "--weight-decay",
        "0.000001",
        "--embedding-dim",
        "16",
        "--hidden-dims",
        "256",
        "128",
        "--dropout",
        "0.2",
        "--seed",
        "42",
        "--device",
        "mps",
    ]
    subprocess.run(command, check=True)


def read_result(config: dict[str, str]) -> dict[str, object]:
    result_dir = Path("results") / config["run_name"]
    with (result_dir / "metrics.csv").open(newline="") as file:
        metric_rows = list(csv.DictReader(file))
    validation = next(row for row in metric_rows if row["split"] == "val")
    manifest = json.loads((result_dir / "run_manifest.json").read_text())

    return {
        "run_name": config["run_name"],
        "model": config["model"],
        "learning_rate": float(config["learning_rate"]),
        "best_epoch": int(manifest["best_epoch"]),
        "parameter_count": int(manifest["parameter_count"]),
        "training_minutes": float(manifest["training_minutes"]),
        "val_auc": float(validation["auc"]),
        "val_log_loss": float(validation["log_loss"]),
        "val_pr_auc": float(validation["pr_auc"]),
        "val_mean_prediction": float(validation["mean_prediction"]),
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with SUMMARY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")

    print("=" * 78)
    print("VALIDATION-ONLY TABULAR NEURAL SEARCH")
    print("=" * 78)
    print("Models:          Wide & Deep, DeepFM")
    print("Configurations:  2 learning rates per model")
    print("Shared capacity: embedding=16, hidden=[256, 128], dropout=0.2")
    print("Frozen test:     disabled")
    print("Resume behavior: completed runs are skipped")

    for index, config in enumerate(CONFIGS, start=1):
        run_configuration(config, index)

    rows = [read_result(config) for config in CONFIGS]
    rows.sort(key=lambda row: (row["model"], row["val_log_loss"]))
    write_summary(rows)

    print("\n" + "=" * 78)
    print("VALIDATION SUMMARY")
    print("=" * 78)
    for row in rows:
        print(
            f"{row['run_name']:<28} "
            f"epoch={row['best_epoch']} "
            f"AUC={row['val_auc']:.6f} "
            f"LogLoss={row['val_log_loss']:.6f} "
            f"PR-AUC={row['val_pr_auc']:.6f} "
            f"time={row['training_minutes']:.1f}m"
        )

    print("\nBest configuration within each model family:")
    for model in ("wide_deep", "deepfm"):
        candidates = [row for row in rows if row["model"] == model]
        best = min(candidates, key=lambda row: row["val_log_loss"])
        print(
            f"  {model:<10} -> {best['run_name']} "
            f"(LogLoss={best['val_log_loss']:.6f}, "
            f"AUC={best['val_auc']:.6f})"
        )

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print("The frozen test split was not read.")


if __name__ == "__main__":
    main()
