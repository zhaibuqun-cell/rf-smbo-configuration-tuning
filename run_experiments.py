from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class EvaluationMetadata:
    system: str
    optimum: float
    worst: float


@dataclass(frozen=True)
class SystemDataset:
    name: str
    feature_names: list[str]
    x: np.ndarray
    objective_values: np.ndarray

    @property
    def n_configs(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_options(self) -> int:
        return int(self.x.shape[1])

    def measure(self, config_index: int) -> float:
        # The measured dataset emulates a black-box system: runtime is only revealed
        # when a valid configuration row is selected for measurement.
        return float(self.objective_values[config_index])

    def evaluation_metadata(self) -> EvaluationMetadata:
        # The optimum and worst values are derived only for post-hoc evaluation and
        # never participate in Random Search or RF-SMBO decisions.
        return EvaluationMetadata(
            system=self.name,
            optimum=float(np.min(self.objective_values)),
            worst=float(np.max(self.objective_values)),
        )


def load_system(dataset_path: Path) -> SystemDataset:
    frame = pd.read_csv(dataset_path)
    feature_names = list(frame.columns[:-1])
    return SystemDataset(
        name=dataset_path.stem,
        feature_names=feature_names,
        # The valid rows in the Lab 3 dataset define the candidate configuration space.
        x=frame[feature_names].to_numpy(dtype=float),
        objective_values=frame[frame.columns[-1]].to_numpy(dtype=float),
    )


def normalised_gap(best_found: float, optimum: float, worst: float) -> float:
    scale = worst - optimum
    if scale <= 0:
        return 0.0
    return float((best_found - optimum) / scale)


def run_random_search(system: SystemDataset, max_budget: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(system.n_configs)[:max_budget]

    sampled_indices: list[int] = []
    sampled_values: list[float] = []
    best_history: list[float] = []
    best_value = math.inf

    for idx in order:
        idx = int(idx)
        measured_value = system.measure(idx)
        sampled_indices.append(idx)
        sampled_values.append(measured_value)
        best_value = min(best_value, measured_value)
        best_history.append(best_value)

    return {
        "sampled_indices": np.asarray(sampled_indices, dtype=int),
        "sampled_values": np.asarray(sampled_values, dtype=float),
        "best_history": np.asarray(best_history, dtype=float),
        "best_index": int(sampled_indices[int(np.argmin(sampled_values))]),
    }


def predict_with_uncertainty(model: RandomForestRegressor, x_candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree_predictions = np.asarray([tree.predict(x_candidate) for tree in model.estimators_], dtype=float)
    return tree_predictions.mean(axis=0), tree_predictions.std(axis=0)


def run_rf_smbo(
    system: SystemDataset,
    max_budget: int,
    seed: int,
    initial_design: int,
    candidate_pool: int,
    kappa: float,
    explore_prob: float,
) -> dict[str, object]:
    if initial_design < 1:
        raise ValueError("initial_design must be at least 1")

    rng = np.random.default_rng(seed)
    measured_mask = np.zeros(system.n_configs, dtype=bool)
    sampled_indices: list[int] = []
    sampled_values: list[float] = []
    best_history: list[float] = []

    # The coursework method starts with six random measurements, but very small
    # budgets should gracefully cap that number instead of overspending.
    init_budget = min(max_budget, initial_design)
    init_indices = rng.choice(system.n_configs, size=init_budget, replace=False)

    best_value = math.inf
    for idx in init_indices:
        idx = int(idx)
        measured_mask[idx] = True
        sampled_indices.append(idx)
        measured_value = system.measure(idx)
        sampled_values.append(measured_value)
        best_value = min(best_value, measured_value)
        best_history.append(best_value)

    while len(sampled_indices) < max_budget:
        observed = np.asarray(sampled_indices, dtype=int)
        observed_values = np.asarray(sampled_values, dtype=float)
        remaining = np.flatnonzero(~measured_mask)
        if remaining.size == 0:
            break

        # The surrogate is retrained after every newly revealed measurement.
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=seed + len(sampled_indices),
            min_samples_leaf=2,
            n_jobs=1,
        )
        model.fit(system.x[observed], observed_values)

        pool_size = min(max(1, candidate_pool), int(remaining.size))
        if remaining.size > pool_size:
            candidates = rng.choice(remaining, size=pool_size, replace=False)
        else:
            candidates = remaining

        candidate_x = system.x[candidates]
        mean_pred, std_pred = predict_with_uncertainty(model, candidate_x)

        if rng.random() < explore_prob:
            next_idx = int(candidates[int(np.argmax(std_pred))])
        else:
            acquisition = mean_pred - (kappa * std_pred)
            next_idx = int(candidates[int(np.argmin(acquisition))])

        measured_mask[next_idx] = True
        sampled_indices.append(next_idx)
        measured_value = system.measure(next_idx)
        sampled_values.append(measured_value)
        best_value = min(best_value, measured_value)
        best_history.append(best_value)

    sampled_array = np.asarray(sampled_indices, dtype=int)
    sampled_values_array = np.asarray(sampled_values, dtype=float)
    return {
        "sampled_indices": sampled_array,
        "sampled_values": sampled_values_array,
        "best_history": np.asarray(best_history, dtype=float),
        "best_index": int(sampled_array[int(np.argmin(sampled_values_array))]),
    }


def build_trajectory_rows(
    system: SystemDataset,
    algorithm: str,
    repeat: int,
    run_seed: int,
    run_result: dict[str, object],
    evaluation: EvaluationMetadata,
) -> list[dict[str, object]]:
    sampled_indices = np.asarray(run_result["sampled_indices"], dtype=int)
    sampled_values = np.asarray(run_result["sampled_values"], dtype=float)
    best_history = np.asarray(run_result["best_history"], dtype=float)

    rows: list[dict[str, object]] = []
    for budget, (idx, measured_value, best_value) in enumerate(
        zip(sampled_indices, sampled_values, best_history, strict=True),
        start=1,
    ):
        rows.append(
            {
                "system": system.name,
                "algorithm": algorithm,
                "repeat": repeat,
                "run_seed": run_seed,
                "budget": budget,
                "sampled_index": int(idx),
                "sampled_performance": float(measured_value),
                "best_found": float(best_value),
                "normalised_gap": normalised_gap(best_value, evaluation.optimum, evaluation.worst),
                "optimum": evaluation.optimum,
                "worst": evaluation.worst,
            }
        )
    return rows


def evaluate_systems(
    systems: list[SystemDataset],
    budgets: list[int],
    repeats: int,
    base_seed: int,
    initial_design: int,
    candidate_pool: int,
    kappa: float,
    explore_prob: float,
) -> pd.DataFrame:
    max_budget = max(budgets)
    evaluations = {system.name: system.evaluation_metadata() for system in systems}
    rows: list[dict[str, object]] = []

    for system_idx, system in enumerate(systems):
        system_seed = base_seed + (system_idx * 10_000)
        evaluation = evaluations[system.name]
        for repeat in range(repeats):
            # Matched seeds keep both algorithms paired for a fair run-by-run comparison.
            run_seed = system_seed + repeat
            random_result = run_random_search(system, max_budget=max_budget, seed=run_seed)
            rf_smbo_result = run_rf_smbo(
                system,
                max_budget=max_budget,
                seed=run_seed,
                initial_design=initial_design,
                candidate_pool=candidate_pool,
                kappa=kappa,
                explore_prob=explore_prob,
            )
            rows.extend(
                build_trajectory_rows(system, "random_search", repeat, run_seed, random_result, evaluation)
            )
            rows.extend(
                build_trajectory_rows(system, "rf_smbo", repeat, run_seed, rf_smbo_result, evaluation)
            )

    return pd.DataFrame(rows)


def summarise_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(["system", "algorithm", "budget"], as_index=False)
        .agg(
            mean_best_found=("best_found", "mean"),
            std_best_found=("best_found", "std"),
            mean_gap=("normalised_gap", "mean"),
            std_gap=("normalised_gap", "std"),
            runs=("repeat", "count"),
            optimum=("optimum", "first"),
            worst=("worst", "first"),
        )
        .sort_values(["system", "budget", "algorithm"])
    )

    overall = (
        results.groupby(["algorithm", "budget"], as_index=False)
        .agg(
            mean_gap=("normalised_gap", "mean"),
            std_gap=("normalised_gap", "std"),
            runs=("repeat", "count"),
        )
        .sort_values(["budget", "algorithm"])
    )
    overall["sem_gap"] = overall["std_gap"] / np.sqrt(overall["runs"])
    overall["ci95"] = 1.96 * overall["sem_gap"]

    tests: list[dict[str, object]] = []
    for (system_name, budget), frame in results.groupby(["system", "budget"]):
        pivot = frame.pivot(index="repeat", columns="algorithm", values="normalised_gap")
        if {"random_search", "rf_smbo"} - set(pivot.columns):
            continue

        random_values = pivot["random_search"].to_numpy(dtype=float)
        rf_smbo_values = pivot["rf_smbo"].to_numpy(dtype=float)
        improvement = random_values - rf_smbo_values

        if np.allclose(improvement, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            test_result = wilcoxon(random_values, rf_smbo_values, alternative="greater", zero_method="pratt")
            statistic = float(test_result.statistic)
            p_value = float(test_result.pvalue)

        tests.append(
            {
                "system": system_name,
                "budget": int(budget),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rf_smbo_win_rate": float(np.mean(rf_smbo_values < random_values)),
                "rf_smbo_mean_gap": float(np.mean(rf_smbo_values)),
                "random_search_mean_gap": float(np.mean(random_values)),
                "median_gap_improvement": float(np.median(improvement)),
            }
        )

    statistics = pd.DataFrame(tests).sort_values(["system", "budget"])
    return summary, overall, statistics


def plot_budget_curve(overall: pd.DataFrame, output_path: Path) -> None:
    colours = {"random_search": "#3b6fb6", "rf_smbo": "#cc503e"}
    labels = {"random_search": "Random Search", "rf_smbo": "RF-SMBO"}

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for algorithm, frame in overall.groupby("algorithm"):
        frame = frame.sort_values("budget")
        ax.plot(
            frame["budget"],
            frame["mean_gap"],
            label=labels[algorithm],
            color=colours[algorithm],
            linewidth=2.3,
            marker="o",
            markersize=4.5,
        )
        ax.fill_between(
            frame["budget"],
            frame["mean_gap"] - frame["ci95"],
            frame["mean_gap"] + frame["ci95"],
            color=colours[algorithm],
            alpha=0.18,
        )

    ax.set_xlabel("Measurement budget")
    ax.set_ylabel("Mean normalised gap to optimum")
    ax.set_title("Figure 1. Convergence under increasing measurement budgets")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_budget_by_system(summary: pd.DataFrame, final_budget: int, output_path: Path) -> None:
    colours = {"random_search": "#3b6fb6", "rf_smbo": "#cc503e"}
    labels = {"random_search": "Random Search", "rf_smbo": "RF-SMBO"}
    final_frame = summary[summary["budget"] == final_budget].copy()
    systems = list(final_frame["system"].drop_duplicates())
    algorithms = ["random_search", "rf_smbo"]
    x = np.arange(len(systems))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, algorithm in enumerate(algorithms):
        algo_frame = (
            final_frame[final_frame["algorithm"] == algorithm]
            .set_index("system")
            .reindex(systems)
            .reset_index()
        )
        positions = x + ((offset - 0.5) * width)
        ax.bar(
            positions,
            algo_frame["mean_gap"],
            width=width,
            label=labels[algorithm],
            color=colours[algorithm],
            alpha=0.94,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("Mean normalised gap to optimum")
    ax.set_title(f"Figure 2. Per-system gap comparison at budget {final_budget}")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_metadata_frame(systems: Iterable[SystemDataset]) -> pd.DataFrame:
    rows = []
    for system in systems:
        evaluation = system.evaluation_metadata()
        rows.append(
            {
                "system": system.name,
                "options": system.n_options,
                "configurations": system.n_configs,
                "optimum": evaluation.optimum,
                "worst": evaluation.worst,
            }
        )
    return pd.DataFrame(rows).sort_values("system")


def write_report_notes(
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
    final_budget: int,
    output_path: Path,
) -> None:
    final_frame = summary[summary["budget"] == final_budget].copy()
    merged = final_frame.pivot(index="system", columns="algorithm", values="mean_gap").reset_index()
    merged["gap_reduction_pct"] = (
        (merged["random_search"] - merged["rf_smbo"]) / merged["random_search"].replace(0, np.nan) * 100.0
    ).fillna(0.0)
    significant = statistics[(statistics["budget"] == final_budget) & (statistics["p_value"] < 0.05)]

    notes = {
        "final_budget": final_budget,
        "systems_evaluated": metadata.to_dict(orient="records"),
        "final_budget_summary": merged.sort_values("gap_reduction_pct", ascending=False).to_dict(orient="records"),
        "significant_systems_at_final_budget": significant.to_dict(orient="records"),
        "project_note": (
            "The valid dataset rows define the candidate configuration space. Runtime values are only revealed "
            "when a row is measured. Global optimum and worst values are reserved for post-hoc evaluation."
        ),
    }
    output_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")


def write_output_copies(
    output_dir: Path,
    metadata: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    files_to_write = {
        "system_metadata.csv": metadata,
        "raw_results.csv": results,
        "summary_results.csv": summary,
        "statistics.csv": statistics,
        "overall_budget_curve.csv": overall,
        # Legacy names are kept for backwards compatibility with earlier drafts.
        "raw_trajectories.csv": results,
        "summary_by_system.csv": summary,
        "statistical_tests.csv": statistics,
    }
    for filename, frame in files_to_write.items():
        frame.to_csv(output_dir / filename, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Lab 3 tuning experiments.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "final_run")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Dataset stems to run. Defaults to all CSV files in the data directory.",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--initial-design", type=int, default=6)
    parser.add_argument("--candidate-pool", type=int, default=4096)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--explore-prob", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = sorted(args.data_dir.glob("*.csv"))
    if not dataset_paths:
        raise FileNotFoundError(f"No CSV files were found in {args.data_dir}")

    if args.systems:
        requested = set(args.systems)
        dataset_paths = [path for path in dataset_paths if path.stem in requested]
        missing = requested - {path.stem for path in dataset_paths}
        if missing:
            raise ValueError(f"Missing datasets: {sorted(missing)}")

    systems = [load_system(path) for path in dataset_paths]
    max_budget = max(args.budgets)
    too_small = [system.name for system in systems if system.n_configs < max_budget]
    if too_small:
        raise ValueError(
            f"The budget {max_budget} exceeds the configuration count for: {', '.join(sorted(too_small))}"
        )

    metadata = build_metadata_frame(systems)
    results = evaluate_systems(
        systems=systems,
        budgets=args.budgets,
        repeats=args.repeats,
        base_seed=args.seed,
        initial_design=args.initial_design,
        candidate_pool=args.candidate_pool,
        kappa=args.kappa,
        explore_prob=args.explore_prob,
    )
    summary, overall, statistics = summarise_results(results)

    write_output_copies(args.output_dir, metadata, results, summary, overall, statistics)
    plot_budget_curve(overall, args.output_dir / "figure_1_convergence.png")
    plot_final_budget_by_system(summary, max_budget, args.output_dir / "figure_2_per_system.png")
    # Legacy figure names remain available for older report-generation code paths.
    plot_budget_curve(overall, args.output_dir / "overall_budget_curve.png")
    plot_final_budget_by_system(summary, max_budget, args.output_dir / "final_budget_by_system.png")
    write_report_notes(metadata, summary, statistics, max_budget, args.output_dir / "report_notes.json")

    print(f"Saved outputs to: {args.output_dir}")
    print((args.output_dir / "summary_results.csv").resolve())
    print((args.output_dir / "statistics.csv").resolve())


if __name__ == "__main__":
    main()
