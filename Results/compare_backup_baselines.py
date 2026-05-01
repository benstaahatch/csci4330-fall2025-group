#!/usr/bin/env python3

import json
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent / "runs"
MODEL_ORDER = [
    "Random Forest (Backup Serial)",
    "Random Forest (Backup OpenMP)",
    "Random Forest (Hybrid)",
]


def load_run(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    data["_path"] = str(path)
    data["_run_name"] = path.parent.name
    return data


def latest_run_by_model() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for metrics_path in sorted(RUNS_DIR.glob("run_*/metrics.json")):
        run = load_run(metrics_path)
        model_name = run.get("model_name")
        if not model_name:
            continue
        latest[model_name] = run
    return latest


def fmt_value(value, decimals=4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def main() -> int:
    latest = latest_run_by_model()

    rows = []
    for model_name in MODEL_ORDER:
        run = latest.get(model_name)
        if not run:
            continue
        rows.append(
            {
                "model": model_name,
                "accuracy": run.get("accuracy"),
                "precision": run.get("precision"),
                "recall": run.get("recall"),
                "f1": run.get("f1"),
                "runtime_sec": run.get("runtime_sec"),
                "mpi_ranks": run.get("mpi_ranks"),
                "omp_threads": run.get("omp_threads"),
                "trees": run.get("n_trees"),
                "depth": run.get("max_depth"),
                "minsplit": run.get("min_samples_split"),
                "max_features": run.get("max_features"),
                "seed": run.get("split_seed"),
                "path": run["_path"],
            }
        )

    if not rows:
        print("No baseline run artifacts found under Results/runs/run_*/metrics.json")
        return 1

    print("Latest baseline comparison:")
    print("")
    print(
        f"{'Model':34} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} "
        f"{'Runtime':>9} {'Ranks':>5} {'Thr':>4} {'Trees':>5} {'Depth':>5} {'MS':>4} {'MF':>4}"
    )
    print("-" * 120)
    for row in rows:
        print(
            f"{row['model'][:34]:34} "
            f"{fmt_value(row['f1'], 4):>8} "
            f"{fmt_value(row['accuracy'], 4):>8} "
            f"{fmt_value(row['precision'], 4):>8} "
            f"{fmt_value(row['recall'], 4):>8} "
            f"{fmt_value(row['runtime_sec'], 4):>9} "
            f"{fmt_value(row['mpi_ranks'], 0):>5} "
            f"{fmt_value(row['omp_threads'], 0):>4} "
            f"{fmt_value(row['trees'], 0):>5} "
            f"{fmt_value(row['depth'], 0):>5} "
            f"{fmt_value(row['minsplit'], 0):>4} "
            f"{fmt_value(row['max_features'], 0):>4}"
        )

    print("")
    print("Run artifacts:")
    for row in rows:
        print(f"- {row['model']}: {row['path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
