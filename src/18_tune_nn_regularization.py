#!/usr/bin/env python3
"""Validation-only capacity and regularization search for tabular CTR NNs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


TRAIN_SCRIPT = Path("src/16_train_tabular_nn.py")
SUMMARY_PATH = Path("results/nn_regularization_summary.csv")


def config(
    model: str,
    suffix: str,
    learning_rate: str,
    weight_decay: str,
    embedding_dim: int,
    hidden_dims: list[int],
    dropout: str,
) -> dict[str, object]:
    return {
        "model": model,
        "run_name": f"{model}_f123_{suffix}",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "embedding_dim": embedding_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }


CONFIGS = []
for model_name in ("wide_deep", "deepfm"):
    CONFIGS.extend(
        [
            config(
                model_name,
                "lr2em4",
                "0.0002",
                "0.000001",
                16,
                [256, 128],
                "0.2",
            ),
            config(
                model_name,
                "regularized",
                "0.0005",
                "0.00001",
                16,
                [256, 128],
                "0.3",
            ),
            config(
                model_name,
                "compact",
                "0.0005",
                "0.00001",
                8,
                [128, 64],
                "0.3",
            ),
        ]
    )


EXISTING_RUNS = [
    "wide_deep_f123_lr1em3",
    "wide_deep_f123_lr5em4",
    "deepfm_f123_lr1em3",
    "deepfm_f123_lr5em4",
]


def result_complete(run_name: str) -> bool:
    result_dir = Path("results") / run_name
    return (
        (result_dir / "metrics.csv").exists()
        and (result_dir / "run_manifest.json").exists()
    )


def run_configuration(item: dict[str, object], index: int) -> None:
    run_name = str(item["run_name"])
    print("\n" + "=" * 78, flush=True)
    print(f"CONFIG {index}/{len(CONFIGS)}: {run_name}", flush=True)
    print("=" * 78, flush=True)
    if result_complete(run_name):
        print("Existing completed result found; skipping training.", flush=True)
        return

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        str(item["model"]),
        "--run-name",
        run_name,
        "--epochs",
        "5",
        "--patience",
        "1",
        "--batch-size",
        "8192",
        "--learning-rate",
        str(item["learning_rate"]),
        "--weight-decay",
        str(item["weight_decay"]),
        "--embedding-dim",
        str(item["embedding_dim"]),
        "--hidden-dims",
        *[str(value) for value in item["hidden_dims"]],
        "--dropout",
        str(item["dropout"]),
        "--seed",
        "42",
        "--device",
        "mps",
    ]
    subprocess.run(command, check=True)


def read_result(run_name: str) -> dict[str, object]:
    result_dir = Path("results") / run_name
    with (result_dir / "metrics.csv").open(newline="") as file:
        metric_rows = list(csv.DictReader(file))
    validation = next(row for row in metric_rows if row["split"] == "val")
    manifest = json.loads((result_dir / "run_manifest.json").read_text())
    return {
        "run_name": run_name,
        "model": manifest["model"],
        "learning_rate": manifest["learning_rate"],
        "weight_decay": manifest["weight_decay"],
        "embedding_dim": manifest["embedding_dim"],
        "hidden_dims": "-".join(str(x) for x in manifest["hidden_dims"]),
        "dropout": manifest["dropout"],
        "best_epoch": manifest["best_epoch"],
        "parameter_count": manifest["parameter_count"],
        "training_minutes": manifest["training_minutes"],
        "val_auc": float(validation["auc"]),
        "val_log_loss": float(validation["log_loss"]),
        "val_pr_auc": float(validation["pr_auc"]),
        "val_mean_prediction": float(validation["mean_prediction"]),
    }


def main() -> None:
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")

    print("=" * 78)
    print("VALIDATION-ONLY NEURAL REGULARIZATION SEARCH")
    print("=" * 78)
    print("New configurations: 6")
    print("Questions: lower LR, stronger regularization, smaller capacity")
    print("Frozen test: disabled")
    print("Resume behavior: completed runs are skipped")

    for index, item in enumerate(CONFIGS, start=1):
        run_configuration(item, index)

    run_names = [*EXISTING_RUNS, *[str(item["run_name"]) for item in CONFIGS]]
    missing = [name for name in run_names if not result_complete(name)]
    if missing:
        raise RuntimeError(f"Missing completed results: {missing}")

    rows = [read_result(name) for name in run_names]
    rows.sort(key=lambda row: (str(row["model"]), float(row["val_log_loss"])))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("VALIDATION SUMMARY")
    print("=" * 78)
    for row in rows:
        print(
            f"{row['run_name']:<31} "
            f"epoch={row['best_epoch']} "
            f"AUC={row['val_auc']:.6f} "
            f"LogLoss={row['val_log_loss']:.6f} "
            f"PR-AUC={row['val_pr_auc']:.6f}"
        )

    print("\nBest configuration within each model family:")
    for model in ("wide_deep", "deepfm"):
        candidates = [row for row in rows if row["model"] == model]
        best = min(candidates, key=lambda row: float(row["val_log_loss"]))
        print(
            f"  {model:<10} -> {best['run_name']} "
            f"(LogLoss={best['val_log_loss']:.6f}, AUC={best['val_auc']:.6f})"
        )

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print("The frozen test split was not read.")


if __name__ == "__main__":
    main()
