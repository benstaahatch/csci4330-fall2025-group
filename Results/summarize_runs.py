#!/usr/bin/env python3

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUNS_DIR = BASE_DIR / "runs"
SUMMARIES_DIR = BASE_DIR / "summaries"
OUTPUT_PATH = SUMMARIES_DIR / f"summary_{TIMESTAMP}.json"
CSV_OUTPUT_PATH = SUMMARIES_DIR / f"summary_{TIMESTAMP}.csv"
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "runtime_sec"]
GROUP_KEYS = [
    "model_name",
    "mpi_ranks",
    "omp_threads",
    "train_fraction",
    "n_trees",
    "max_depth",
    "min_samples_split",
    "max_features",
]


def discover_run_files() -> list[Path]:
    return sorted(RUNS_DIR.glob("run_*/metrics.json"))


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    data["_path"] = str(path)
    return data


def group_key(run: dict) -> tuple:
    return tuple(run.get(key) for key in GROUP_KEYS)


def summarize_group(runs: list[dict]) -> dict:
    base = {key: runs[0].get(key) for key in GROUP_KEYS}
    summary = dict(base)
    summary["run_count"] = len(runs)
    summary["split_seeds"] = sorted(run.get("split_seed") for run in runs if "split_seed" in run)
    summary["run_paths"] = [run["_path"] for run in runs]

    for metric in METRIC_KEYS:
        values = [run[metric] for run in runs if metric in run]
        if values:
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

    return summary


def write_csv(summaries: list[dict], output_path: Path) -> None:
    fieldnames = [
        "model_name",
        "mpi_ranks",
        "omp_threads",
        "train_fraction",
        "n_trees",
        "max_depth",
        "min_samples_split",
        "max_features",
        "run_count",
        "split_seeds",
        "accuracy_mean",
        "accuracy_min",
        "accuracy_max",
        "accuracy_std",
        "precision_mean",
        "precision_min",
        "precision_max",
        "precision_std",
        "recall_mean",
        "recall_min",
        "recall_max",
        "recall_std",
        "f1_mean",
        "f1_min",
        "f1_max",
        "f1_std",
        "runtime_sec_mean",
        "runtime_sec_min",
        "runtime_sec_max",
        "runtime_sec_std",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            row = {key: item.get(key) for key in fieldnames}
            if row["split_seeds"] is not None:
                row["split_seeds"] = " ".join(str(seed) for seed in row["split_seeds"])
            writer.writerow(row)


def main() -> int:
    run_files = discover_run_files()
    if not run_files:
        print("No run metrics found under Results/runs/run_*/metrics.json")
        return 1

    runs = [load_metrics(path) for path in run_files]

    grouped: dict[tuple, list[dict]] = {}
    for run in runs:
        grouped.setdefault(group_key(run), []).append(run)

    summaries = [summarize_group(group_runs) for group_runs in grouped.values()]
    summaries.sort(key=lambda item: item.get("f1_mean", 0.0), reverse=True)

    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as outfile:
        json.dump(summaries, outfile, indent=2)
    write_csv(summaries, CSV_OUTPUT_PATH)

    print(f"Wrote summary to {OUTPUT_PATH}")
    print(f"Wrote summary CSV to {CSV_OUTPUT_PATH}")
    print("")
    print("Average validation metrics by configuration:")
    for item in summaries:
        print(
            f"- {item.get('model_name', 'Unknown')} | "
            f"trees={item.get('n_trees')} | "
            f"mf={item.get('max_features')} | "
            f"depth={item.get('max_depth')} | "
            f"minsplit={item.get('min_samples_split')} | "
            f"train={item.get('train_fraction')} | "
            f"ranks={item.get('mpi_ranks')} | "
            f"threads={item.get('omp_threads')} | "
            f"runs={item.get('run_count')} | "
            f"mean_f1={item.get('f1_mean', 0.0):.4f} | "
            f"mean_acc={item.get('accuracy_mean', 0.0):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
