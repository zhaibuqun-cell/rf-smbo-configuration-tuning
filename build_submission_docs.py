from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

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
from matplotlib.backends.backend_pdf import PdfPages


ARTIFACT_LINK_PLACEHOLDER = "TO_BE_REPLACED_WITH_FINAL_GITHUB_URL"


@dataclass(frozen=True)
class SystemResult:
    system: str
    random_best: float
    rf_best: float
    random_gap: float
    rf_gap: float
    p_value: float
    win_rate: float

    @property
    def performance_improvement_pct(self) -> float:
        return ((self.random_best - self.rf_best) / self.random_best) * 100.0

    @property
    def gap_reduction_pct(self) -> float:
        return ((self.random_gap - self.rf_gap) / self.random_gap) * 100.0

    @property
    def win_rate_pct(self) -> float:
        return self.win_rate * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate markdown and PDF submission artifacts.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_run",
        help="Directory containing experiment outputs.",
    )
    parser.add_argument(
        "--artifact-link",
        default=ARTIFACT_LINK_PLACEHOLDER,
        help="Repository URL placeholder to insert into the documents.",
    )
    return parser.parse_args()


def resolve_writable_pdf_path(output_path: Path) -> Path:
    try:
        with open(output_path, "ab"):
            pass
        return output_path
    except PermissionError:
        return output_path.with_name(f"{output_path.stem}.rebuilt{output_path.suffix}")


def read_result_csv(result_dir: Path, preferred_name: str, fallback_name: str | None = None) -> pd.DataFrame:
    preferred = result_dir / preferred_name
    if preferred.exists():
        return pd.read_csv(preferred)
    if fallback_name:
        fallback = result_dir / fallback_name
        if fallback.exists():
            return pd.read_csv(fallback)
    raise FileNotFoundError(f"Could not find {preferred_name} in {result_dir}")


def load_results(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = read_result_csv(result_dir, "summary_results.csv", "summary_by_system.csv")
    statistics = read_result_csv(result_dir, "statistics.csv", "statistical_tests.csv")
    overall = read_result_csv(result_dir, "overall_budget_curve.csv")
    metadata = read_result_csv(result_dir, "system_metadata.csv")
    raw = read_result_csv(result_dir, "raw_results.csv", "raw_trajectories.csv")
    return summary, statistics, overall, metadata, raw


def package_versions() -> dict[str, str]:
    import matplotlib as mpl
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
    }


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def get_final_budget(overall: pd.DataFrame) -> int:
    return int(overall["budget"].max())


def build_system_result(summary: pd.DataFrame, statistics: pd.DataFrame, system_name: str, budget: int) -> SystemResult:
    final_summary = summary[(summary["system"] == system_name) & (summary["budget"] == budget)].set_index("algorithm")
    final_stats = statistics[(statistics["system"] == system_name) & (statistics["budget"] == budget)].iloc[0]
    return SystemResult(
        system=system_name,
        random_best=float(final_summary.loc["random_search", "mean_best_found"]),
        rf_best=float(final_summary.loc["rf_smbo", "mean_best_found"]),
        random_gap=float(final_summary.loc["random_search", "mean_gap"]),
        rf_gap=float(final_summary.loc["rf_smbo", "mean_gap"]),
        p_value=float(final_stats["p_value"]),
        win_rate=float(final_stats["rf_smbo_win_rate"]),
    )


def compute_overall_gap_reduction(overall: pd.DataFrame, final_budget: int) -> tuple[float, float, float]:
    final = overall[overall["budget"] == final_budget].set_index("algorithm")
    random_gap = float(final.loc["random_search", "mean_gap"])
    rf_gap = float(final.loc["rf_smbo", "mean_gap"])
    reduction_pct = ((random_gap - rf_gap) / random_gap) * 100.0
    return random_gap, rf_gap, reduction_pct


def new_page() -> tuple[plt.Figure, float]:
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(8.27, 11.69))
    return fig, 0.95


def render_wrapped_text(
    fig: plt.Figure,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 10.0,
    weight: str | None = None,
    width: int = 94,
    line_height: float = 0.0205,
    family: str | None = None,
) -> float:
    wrapped = textwrap.fill(text, width=width)
    fig.text(
        x,
        y,
        wrapped,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight=weight,
        fontfamily=family,
        linespacing=1.32,
    )
    lines = wrapped.count("\n") + 1
    return y - (lines * line_height * (fontsize / 10.0))


def render_bullets(
    fig: plt.Figure,
    x: float,
    y: float,
    items: list[str],
    *,
    width: int = 90,
    fontsize: float = 10.0,
    gap: float = 0.006,
) -> float:
    for item in items:
        y = render_wrapped_text(fig, x, y, f"- {item}", width=width, fontsize=fontsize)
        y -= gap
    return y


def render_code_block(
    fig: plt.Figure,
    x: float,
    y: float,
    title: str,
    code_lines: list[str],
    *,
    fontsize: float = 8.6,
) -> float:
    y = render_wrapped_text(fig, x, y, title, fontsize=11.0, weight="bold", width=70)
    block = "\n".join(code_lines)
    fig.text(
        x + 0.012,
        y - 0.006,
        block,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontfamily="DejaVu Sans Mono",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f4f4f4", "edgecolor": "#d0d0d0"},
    )
    lines = len(code_lines)
    return y - 0.02 - (lines * 0.0175)


def ensure_space(pdf: PdfPages, fig: plt.Figure, y: float, needed: float) -> tuple[plt.Figure, float]:
    if y >= needed:
        return fig, y
    pdf.savefig(fig)
    plt.close(fig)
    return new_page()


def add_document_section(
    pdf: PdfPages,
    fig: plt.Figure,
    y: float,
    heading: str,
    paragraphs: list[str] | None = None,
    bullets: list[str] | None = None,
    code_blocks: list[tuple[str, list[str]]] | None = None,
) -> tuple[plt.Figure, float]:
    fig, y = ensure_space(pdf, fig, y, 0.18)
    y = render_wrapped_text(fig, 0.08, y, heading, fontsize=13.0, weight="bold", width=60)
    y -= 0.004

    if paragraphs:
        for paragraph in paragraphs:
            fig, y = ensure_space(pdf, fig, y, 0.16)
            y = render_wrapped_text(fig, 0.08, y, paragraph)
            y -= 0.008

    if bullets:
        fig, y = ensure_space(pdf, fig, y, 0.20)
        y = render_bullets(fig, 0.10, y, bullets)
        y -= 0.004

    if code_blocks:
        for title, code_lines in code_blocks:
            fig, y = ensure_space(pdf, fig, y, 0.32)
            y = render_code_block(fig, 0.08, y, title, code_lines)
            y -= 0.006

    return fig, y


def report_markdown(
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
    overall: pd.DataFrame,
    metadata: pd.DataFrame,
    artifact_link: str,
) -> str:
    final_budget = get_final_budget(overall)
    random_gap, rf_gap, overall_reduction = compute_overall_gap_reduction(overall, final_budget)
    llvm = build_system_result(summary, statistics, "LLVM", final_budget)
    x264 = build_system_result(summary, statistics, "x264", final_budget)
    z7 = build_system_result(summary, statistics, "7z", final_budget)

    return f"""# Configuration Performance Tuning with a Random-Forest Sequential Model

## Introduction

This coursework presents a budget-limited intelligent software engineering tool for Lab 3: Configuration Performance Tuning. The task is to search for high-performing configurations while consuming only a small number of measurements, which mirrors the practical setting where each configuration evaluation may be expensive.

The lab baseline is **Random Search**, while the proposed tool is **Random-Forest Sequential Model-Based Optimisation (RF-SMBO)**. Under the final budget of {final_budget} measurements, RF-SMBO reduces the overall mean normalised gap from {random_gap:.4f} to {rf_gap:.4f}, a reduction of {overall_reduction:.2f}%. However, this result should not be stated too strongly: RF-SMBO overall improves the mean normalised gap mainly because it performs much better on **LLVM** and **x264**, but it underperforms **Random Search** on **7z**.

## Related Work

Random Search is a natural baseline for black-box optimisation because it is simple, budget-aware, and surprisingly competitive in high-dimensional spaces when compared with naive exhaustive strategies [1]. For this reason it is appropriate as the baseline in Lab 3, where the search budget is explicitly limited and every measurement is treated as costly.

Sequential Model-Based Optimisation (SMBO) improves on blind sampling by learning a surrogate model from past observations and using that surrogate to guide the next measurement [2]. Bayesian optimisation is a closely related family of methods that uses predictive uncertainty to balance exploration and exploitation [3]. This idea is directly relevant to the current coursework because the objective is not to evaluate every configuration, but to spend a small budget on measurements that are likely to be informative.

Random-forest surrogates are especially appealing in configurable software systems because they can handle non-linear interactions, mixed discrete or integer-like options, and irregular performance landscapes. Prior work on software configuration tuning shows that uncertainty-aware and surrogate-guided optimisation can be effective for performance-sensitive systems [4], and that even imperfect learners can still help identify useful configurations in search-based software engineering settings [5]. The present project builds on that idea by using a random forest as the surrogate inside an SMBO loop.

## Solution

The search space is defined by the valid rows already present in each dataset. Each row represents one legal candidate configuration. Runtime values are **not** used in advance to rank all candidates; instead, a runtime value is only revealed when that row is selected for measurement. After each new measurement, RF-SMBO retrains a `RandomForestRegressor` on the observed configurations and their measured runtimes. A random forest is suitable here because it combines many decision-tree regressors and averages their predictions, which improves prediction stability and helps control over-fitting while still modelling non-linear option interactions.

### Algorithm 1: RF-SMBO for Configuration Performance Tuning

```text
1. Initialise the observed set with six random configurations.
2. Measure those configurations and store the revealed runtimes.
3. Train RandomForestRegressor on the observed configurations.
4. Predict mean and uncertainty for unmeasured configurations.
5. Compute acquisition score = predicted_mean - 0.5 * predicted_std.
6. With 10% probability, select the most uncertain candidate.
7. Otherwise, select the candidate with the lowest acquisition score.
8. Measure the selected configuration and update the observed set.
9. Repeat until the measurement budget is exhausted.
10. Return the best observed configuration.
```

The baseline Random Search samples unseen valid configurations uniformly at random without replacement.

## Setup

The final evaluation uses three Lab 3 systems: **7z** ({int(metadata.loc[metadata["system"] == "7z", "configurations"].iloc[0]):,} configurations, {int(metadata.loc[metadata["system"] == "7z", "options"].iloc[0])} options), **LLVM** ({int(metadata.loc[metadata["system"] == "LLVM", "configurations"].iloc[0]):,} configurations, {int(metadata.loc[metadata["system"] == "LLVM", "options"].iloc[0])} options), and **x264** ({int(metadata.loc[metadata["system"] == "x264", "configurations"].iloc[0]):,} configurations, {int(metadata.loc[metadata["system"] == "x264", "options"].iloc[0])} options). All objectives are minimisation problems. Budgets are 10, 20, 30, 40, and 50 measurements, and each algorithm is repeated 30 times with paired seeds for fair comparison.

The primary metric is the **normalised gap to the optimum**, defined as `(best_found - optimum) / (worst - optimum)`, where lower is better. The secondary metric is the best runtime found under the given budget. Statistical significance is assessed using a **one-sided paired Wilcoxon signed-rank test**, testing whether RF-SMBO achieves a lower normalised gap than Random Search.

The global optimum and worst performance values are used **only for post-hoc evaluation and normalisation**. They are **never** used by Random Search or RF-SMBO during the search process.

## Experiments

Figure 1 shows convergence as the budget increases. The proposed method generally improves faster than the baseline because the surrogate becomes more informative after each new measurement.

Table 1 summarises the final-budget results at 50 measurements, while Figure 2 shows the per-system differences. RF-SMBO improves clearly on **LLVM** and **x264** but loses on **7z**, so the overall result must be interpreted as a mixed but promising outcome rather than a universal win.

| System | Random best | RF-SMBO best | Random gap | RF-SMBO gap | Gap reduction (%) | p-value | win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7z | {z7.random_best:.2f} | {z7.rf_best:.2f} | {z7.random_gap:.4f} | {z7.rf_gap:.4f} | {z7.gap_reduction_pct:.2f} | {format_p_value(z7.p_value)} | {z7.win_rate_pct:.1f}% |
| LLVM | {llvm.random_best:.2f} | {llvm.rf_best:.2f} | {llvm.random_gap:.4f} | {llvm.rf_gap:.4f} | {llvm.gap_reduction_pct:.2f} | {format_p_value(llvm.p_value)} | {llvm.win_rate_pct:.1f}% |
| x264 | {x264.random_best:.4f} | {x264.rf_best:.4f} | {x264.random_gap:.4f} | {x264.rf_gap:.4f} | {x264.gap_reduction_pct:.2f} | {format_p_value(x264.p_value)} | {x264.win_rate_pct:.1f}% |

On **LLVM**, RF-SMBO reduces the mean best runtime from {llvm.random_best:.2f} to {llvm.rf_best:.2f} and reduces the normalised gap by {llvm.gap_reduction_pct:.2f}% (`p = {format_p_value(llvm.p_value)}`). On **x264**, the runtime improves from {x264.random_best:.4f} to {x264.rf_best:.4f}, with a {x264.gap_reduction_pct:.2f}% reduction in normalised gap (`p = {format_p_value(x264.p_value)}`). In contrast, **7z** remains a failure case where RF-SMBO is worse than the baseline.

## Reflection and Conclusion

The negative result on **7z** should be kept rather than hidden, because it is a scientifically valid result. It shows that the surrogate model is not always reliable, even when the same method works well on other systems. Several factors may explain this outcome. First, the 7z search landscape may be rugged or deceptive, making it harder for a small surrogate model to rank candidates well. Second, the measurement budget may provide too little training data for accurate model fitting. Third, the uncertainty estimate derived from tree disagreement in the random forest is only an approximation and may not reflect the true search uncertainty. Fourth, some configuration spaces may reward broader exploration more than model-guided exploitation.

Despite that limitation, RF-SMBO is still useful overall because it delivers strong improvements on LLVM and x264 under the same limited budgets. Future work should investigate **Expected Improvement** or rank-based acquisition, an adaptive exploration coefficient instead of the fixed value 0.5, comparisons against stronger baselines such as local search, genetic algorithms, or A-T-EGLS, multi-objective runtime-energy tuning, and additional systems from Lab 3.

In conclusion, RF-SMBO improves the overall mean normalised gap under a limited measurement budget, but the benefit is driven by strong gains on LLVM and x264 rather than universal superiority. The 7z failure case therefore strengthens the reflection section and makes the submission more scientifically credible.

## Artifact

Artifact link: {artifact_link}

The repository includes source code, raw CSV outputs, generated figures, `requirements.pdf`, `manual.pdf`, and `replication.pdf`.

## References

[1] J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," *Journal of Machine Learning Research*, vol. 13, no. 10, pp. 281-305, 2012. Available: https://www.jmlr.org/papers/v13/bergstra12a.html

[2] F. Hutter, H. H. Hoos, and K. Leyton-Brown, "Sequential Model-Based Optimization for General Algorithm Configuration," *LION 5 / UBC Technical Report TR-2010-10*, 2011. Available: https://www.cs.ubc.ca/tr/2010/tr-2010-10

[3] J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," in *Advances in Neural Information Processing Systems*, 2012. Available: https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms

[4] P. Jamshidi and G. Casale, "An Uncertainty-Aware Approach to Optimal Configuration of Stream Processing Systems," in *MASCOTS*, 2016. Available: https://pooyanjamshidi.github.io/resources/papers/bo4co.pdf

[5] V. Nair, T. Menzies, N. Siegmund, and S. Apel, "Using Bad Learners to find Good Configurations," in *ESEC/FSE*, 2017. Available: https://arxiv.org/abs/1702.05701

[6] ideas-labo, "ISE Lab 3 repository." Available: https://github.com/ideas-labo/ISE/tree/main/lab3
"""


def requirements_markdown(versions: dict[str, str]) -> str:
    return f"""# Requirements

## Runtime Environment

- Operating system assumption: tested on Windows 11, but the scripts are pure Python and should also run on Linux or macOS with the same package versions.
- Python version: {versions["python"]}

## Python Dependencies

- numpy=={versions["numpy"]}
- pandas=={versions["pandas"]}
- scikit-learn=={versions["scikit-learn"]}
- scipy=={versions["scipy"]}
- matplotlib=={versions["matplotlib"]}

## Installation

From the project root:

```bash
python -m pip install -r requirements.txt
```

The project also contains an optional local `vendor/` directory that was used during development, but a clean setup should rely on `requirements.txt` rather than committing a full package cache.

## Repository Contents Required for Replication

- experiment scripts: `run_experiments.py`, `build_submission_docs.py`
- datasets: `data/`
- final results: `outputs/final_run/`
- report and support files: `coursework_report.pdf`, `requirements.pdf`, `manual.pdf`, `replication.pdf`
"""


def manual_markdown(artifact_link: str) -> str:
    return f"""# Manual

## Purpose

This tool solves the Lab 3 coursework task: configuration performance tuning under a limited measurement budget. It compares two algorithms:

- Random Search
- RF-SMBO (Random-Forest Sequential Model-Based Optimisation)

## Input Format

The tool expects CSV datasets stored in `data/`. Each file should contain:

- one row per valid configuration
- feature columns for configuration options
- a final column named `performance`

The included Lab 3 datasets already follow this format.

## Running the Experiment Tool

From the project root:

```bash
python run_experiments.py --systems 7z LLVM x264 --budgets 10 20 30 40 50 --repeats 30 --seed 20260424 --initial-design 6 --candidate-pool 4096 --kappa 0.5 --explore-prob 0.1 --output-dir outputs/final_run
```

## Rebuilding the Submission Documents

```bash
python build_submission_docs.py --artifact-link "{artifact_link}"
```

## Output Files

The main output directory is `outputs/final_run/`. Important files are:

- `raw_results.csv`: full trajectory-level measurements for every run
- `summary_results.csv`: mean and standard deviation summaries by system, budget, and algorithm
- `statistics.csv`: Wilcoxon tests and win rates
- `figure_1_convergence.png`: convergence curve over budgets
- `figure_2_per_system.png`: per-system final-budget comparison
- `system_metadata.csv`: configuration counts and evaluation metadata

## Interpreting the Results

- Lower `best_found` runtime is better.
- Lower `normalised_gap` is better.
- `win rate` is the percentage of paired runs where RF-SMBO finishes with a lower final normalised gap than Random Search.
- A small one-sided Wilcoxon `p-value` indicates evidence that RF-SMBO outperforms Random Search on that system and budget.
"""


def replication_markdown(artifact_link: str, overall: pd.DataFrame, statistics: pd.DataFrame) -> str:
    final_budget = get_final_budget(overall)
    random_gap, rf_gap, _ = compute_overall_gap_reduction(overall, final_budget)
    llvm_p = format_p_value(
        float(statistics[(statistics["system"] == "LLVM") & (statistics["budget"] == final_budget)]["p_value"].iloc[0])
    )
    x264_p = format_p_value(
        float(statistics[(statistics["system"] == "x264") & (statistics["budget"] == final_budget)]["p_value"].iloc[0])
    )
    return f"""# Replication

## Overview

This document explains how to reproduce the final coursework results from a clean checkout.

## Step-by-Step Instructions

1. Clone or copy the repository.
2. Ensure the Lab 3 CSV datasets are present in `data/`. The current repository already includes them.
3. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run the final experiment configuration:

```bash
python run_experiments.py --systems 7z LLVM x264 --budgets 10 20 30 40 50 --repeats 30 --seed 20260424 --initial-design 6 --candidate-pool 4096 --kappa 0.5 --explore-prob 0.1 --output-dir outputs/final_run
```

5. Rebuild the markdown and PDF documents:

```bash
python build_submission_docs.py --artifact-link "{artifact_link}"
```

## Expected Outputs

After the experiment command finishes, `outputs/final_run/` should contain:

- `raw_results.csv`
- `summary_results.csv`
- `statistics.csv`
- `overall_budget_curve.csv`
- `figure_1_convergence.png`
- `figure_2_per_system.png`
- `system_metadata.csv`

The second command should regenerate:

- `coursework_report.md` and `coursework_report.pdf`
- `requirements.md` and `requirements.pdf`
- `manual.md` and `manual.pdf`
- `replication.md` and `replication.pdf`

## Reproducibility Notes

- The base seed is `20260424`.
- Paired seeds are used so that Random Search and RF-SMBO are compared fairly within the same system and repeat.
- Optimum and worst values are used only after search for normalised-gap evaluation.
- The expected overall mean normalised gap at budget {final_budget} is:
  - Random Search: {random_gap:.4f}
  - RF-SMBO: {rf_gap:.4f}
- Expected headline p-values at budget {final_budget}:
  - LLVM: {llvm_p}
  - x264: {x264_p}

## Regenerating Figures and Tables

The report figures are generated directly by `run_experiments.py`, while Table 1 and the final PDFs are rebuilt by `build_submission_docs.py`. If the artifact link changes after GitHub upload, rerun `build_submission_docs.py` so that the PDFs contain the final repository URL.
"""


def write_markdown_files(
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
    overall: pd.DataFrame,
    metadata: pd.DataFrame,
    artifact_link: str,
) -> None:
    versions = package_versions()
    (PROJECT_ROOT / "coursework_report.md").write_text(
        report_markdown(summary, statistics, overall, metadata, artifact_link),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "requirements.md").write_text(requirements_markdown(versions), encoding="utf-8")
    (PROJECT_ROOT / "manual.md").write_text(manual_markdown(artifact_link), encoding="utf-8")
    (PROJECT_ROOT / "replication.md").write_text(
        replication_markdown(artifact_link, overall, statistics),
        encoding="utf-8",
    )


def build_coursework_report_pdf(
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
    overall: pd.DataFrame,
    metadata: pd.DataFrame,
    result_dir: Path,
    artifact_link: str,
) -> Path:
    final_budget = get_final_budget(overall)
    random_gap, rf_gap, overall_reduction = compute_overall_gap_reduction(overall, final_budget)
    llvm = build_system_result(summary, statistics, "LLVM", final_budget)
    x264 = build_system_result(summary, statistics, "x264", final_budget)
    z7 = build_system_result(summary, statistics, "7z", final_budget)

    figure_1_path = result_dir / "figure_1_convergence.png"
    if not figure_1_path.exists():
        figure_1_path = result_dir / "overall_budget_curve.png"

    figure_2_path = result_dir / "figure_2_per_system.png"
    if not figure_2_path.exists():
        figure_2_path = result_dir / "final_budget_by_system.png"

    output_path = resolve_writable_pdf_path(PROJECT_ROOT / "coursework_report.pdf")
    with PdfPages(output_path) as pdf:
        fig, y = new_page()
        y = render_wrapped_text(
            fig,
            0.08,
            y,
            "Configuration Performance Tuning with a Random-Forest Sequential Model",
            fontsize=16,
            weight="bold",
            width=58,
            line_height=0.022,
        )
        y = render_wrapped_text(
            fig,
            0.08,
            y - 0.01,
            "Intelligent Software Engineering Coursework Report",
            fontsize=11,
            width=64,
        )
        y -= 0.014
        y = render_wrapped_text(fig, 0.08, y, "Introduction", fontsize=13, weight="bold", width=62)
        intro_paragraphs = [
            "This project builds a limited-budget intelligent software engineering tool for configuration performance tuning in the Lab 3 setting. The core challenge is to find strong configurations while treating every measurement as expensive, rather than exhaustively traversing the full configuration space.",
            (
                f"The baseline is Random Search and the proposed method is Random-Forest Sequential Model-Based Optimisation (RF-SMBO). "
                f"At budget {final_budget}, RF-SMBO lowers the overall mean normalised gap from {random_gap:.4f} to {rf_gap:.4f}, "
                f"a reduction of {overall_reduction:.2f}%. However, this result is not uniform across all systems: the overall improvement is driven mainly by strong gains on LLVM and x264, while RF-SMBO underperforms Random Search on 7z."
            ),
        ]
        for paragraph in intro_paragraphs:
            y = render_wrapped_text(fig, 0.08, y - 0.008, paragraph)
        y -= 0.008
        y = render_wrapped_text(fig, 0.08, y, "Related Work", fontsize=13, weight="bold", width=62)
        related_paragraphs = [
            "Random Search is a suitable baseline for black-box configuration optimisation because it is simple, fair under a fixed budget, and often stronger than naive exhaustive alternatives in high-dimensional spaces [1]. In the present coursework it provides a transparent reference point for understanding whether a learned search policy adds value.",
            "Sequential Model-Based Optimisation (SMBO) improves on unguided search by fitting a surrogate on previous measurements and using it to decide which configuration to evaluate next [2]. Bayesian optimisation follows the same high-level idea and emphasises the balance between exploitation and uncertainty-driven exploration [3]. This budget-aware perspective closely matches the Lab 3 objective, where only a small number of measurements may be spent.",
            "Random-forest surrogates are attractive in configurable software systems because they can model non-linear option interactions without heavy feature engineering, while remaining robust on mixed discrete or integer-like search spaces. Prior studies in software-system tuning and search-based software engineering show that uncertainty-aware or surrogate-guided methods can outperform purely random search in expensive tuning tasks [4], [5].",
        ]
        for paragraph in related_paragraphs:
            y = render_wrapped_text(fig, 0.08, y - 0.008, paragraph)
        pdf.savefig(fig)
        plt.close(fig)

        fig, y = new_page()
        y = render_wrapped_text(fig, 0.08, y, "Solution", fontsize=13, weight="bold", width=62)
        solution_paragraphs = [
            "The valid rows in each dataset define the candidate configuration space. A runtime value is not used to rank all candidates in advance; instead, it is revealed only when a row is selected for measurement. RF-SMBO begins with six random measurements, retrains a surrogate after every new observation, and then uses that surrogate to choose the next valid configuration.",
            "The surrogate is scikit-learn's RandomForestRegressor, which is a meta-estimator that averages the predictions of multiple decision-tree regressors. This is a natural fit here because averaging improves stability, helps control over-fitting, and still allows the model to capture non-linear interactions between configuration options.",
        ]
        for paragraph in solution_paragraphs:
            y = render_wrapped_text(fig, 0.08, y - 0.008, paragraph)
        y -= 0.002
        algorithm_lines = [
            "1. Initialise the observed set with six random configurations.",
            "2. Measure those configurations and store the revealed runtimes.",
            "3. Train RandomForestRegressor on observed configurations.",
            "4. Predict mean and uncertainty for unmeasured configurations.",
            "5. Compute score = predicted_mean - 0.5 * predicted_std.",
            "6. With 10% probability, pick the most uncertain candidate.",
            "7. Otherwise, pick the candidate with the lowest score.",
            "8. Measure that configuration and update the observed set.",
            "9. Repeat until the budget is exhausted.",
            "10. Return the best observed configuration.",
        ]
        y = render_code_block(
            fig,
            0.08,
            y - 0.004,
            "Algorithm 1: RF-SMBO for Configuration Performance Tuning",
            algorithm_lines,
        )
        y = render_wrapped_text(
            fig,
            0.08,
            y - 0.002,
            "The baseline Random Search uses the same valid candidate space but samples unseen configurations uniformly at random without replacement.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, y = new_page()
        y = render_wrapped_text(fig, 0.08, y, "Setup", fontsize=13, weight="bold", width=62)
        setup_paragraphs = [
            (
                f"The final study evaluates three systems from the Lab 3 datasets: 7z "
                f"({int(metadata.loc[metadata['system'] == '7z', 'configurations'].iloc[0]):,} configurations, "
                f"{int(metadata.loc[metadata['system'] == '7z', 'options'].iloc[0])} options), LLVM "
                f"({int(metadata.loc[metadata['system'] == 'LLVM', 'configurations'].iloc[0]):,} configurations, "
                f"{int(metadata.loc[metadata['system'] == 'LLVM', 'options'].iloc[0])} options), and x264 "
                f"({int(metadata.loc[metadata['system'] == 'x264', 'configurations'].iloc[0]):,} configurations, "
                f"{int(metadata.loc[metadata['system'] == 'x264', 'options'].iloc[0])} options). All objectives are minimisation tasks."
            ),
            "Budgets are 10, 20, 30, 40, and 50 measurements. Each system-budget pair is repeated 30 times, and paired seeds are used so that Random Search and RF-SMBO are compared under the same random seed for each repeat.",
            "The primary metric is the normalised gap to the optimum, where lower is better. The secondary metric is the best runtime found at a given budget. Statistical significance is assessed using a one-sided paired Wilcoxon signed-rank test.",
            "The global optimum and worst performance values are used only for post-hoc evaluation and normalisation. They are never used by Random Search or RF-SMBO during the search process.",
        ]
        for paragraph in setup_paragraphs:
            y = render_wrapped_text(fig, 0.08, y - 0.008, paragraph)
        y -= 0.01
        y = render_bullets(
            fig,
            0.10,
            y,
            [
                "Primary metric: normalised gap to optimum, where lower is better.",
                "Secondary metric: best runtime found under the same budget.",
                "Paired comparison: one run seed per system-repeat pair, shared by both algorithms.",
                "Search policy restriction: only valid dataset rows may be selected and measured.",
            ],
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, y = new_page()
        y = render_wrapped_text(fig, 0.08, y, "Experiments", fontsize=13, weight="bold", width=62)
        y = render_wrapped_text(
            fig,
            0.08,
            y - 0.008,
            "Figure 1 shows convergence under increasing budgets. The gap between RF-SMBO and Random Search widens as more measurements are spent, which indicates that the surrogate becomes more useful once a small observation history has been collected.",
        )
        image_ax = fig.add_axes([0.09, 0.28, 0.82, 0.56])
        image_ax.imshow(plt.imread(figure_1_path))
        image_ax.axis("off")
        y = render_wrapped_text(
            fig,
            0.08,
            0.22,
            "Figure 1. Mean normalised gap under increasing measurement budgets, with 95% confidence intervals.",
            fontsize=9,
            width=92,
        )
        y = render_wrapped_text(
            fig,
            0.08,
            y - 0.012,
            (
                f"At the final budget of {final_budget}, the overall mean normalised gap decreases from {random_gap:.4f} for Random Search "
                f"to {rf_gap:.4f} for RF-SMBO. This overall trend is positive, but it should be interpreted together with the per-system results because one dataset remains a clear negative case."
            ),
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, y = new_page()
        y = render_wrapped_text(fig, 0.08, y, "Experiments", fontsize=13, weight="bold", width=62)
        y = render_wrapped_text(
            fig,
            0.08,
            y - 0.008,
            "Table 1 reports the final-budget results at 50 measurements, including the paired-run win rate. Figure 2 then visualises the final per-system differences.",
        )
        table_ax = fig.add_axes([0.05, 0.67, 0.90, 0.16])
        table_ax.axis("off")
        table_rows = [
            [
                "7z",
                f"{z7.random_best:.2f}",
                f"{z7.rf_best:.2f}",
                f"{z7.random_gap:.4f}",
                f"{z7.rf_gap:.4f}",
                f"{z7.gap_reduction_pct:.2f}",
                format_p_value(z7.p_value),
                f"{z7.win_rate_pct:.1f}%",
            ],
            [
                "LLVM",
                f"{llvm.random_best:.2f}",
                f"{llvm.rf_best:.2f}",
                f"{llvm.random_gap:.4f}",
                f"{llvm.rf_gap:.4f}",
                f"{llvm.gap_reduction_pct:.2f}",
                format_p_value(llvm.p_value),
                f"{llvm.win_rate_pct:.1f}%",
            ],
            [
                "x264",
                f"{x264.random_best:.4f}",
                f"{x264.rf_best:.4f}",
                f"{x264.random_gap:.4f}",
                f"{x264.rf_gap:.4f}",
                f"{x264.gap_reduction_pct:.2f}",
                format_p_value(x264.p_value),
                f"{x264.win_rate_pct:.1f}%",
            ],
        ]
        table = table_ax.table(
            cellText=table_rows,
            colLabels=[
                "System",
                "Random best",
                "RF-SMBO best",
                "Random gap",
                "RF-SMBO gap",
                "Gap red. (%)",
                "p-value",
                "win rate",
            ],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.7)
        table.scale(1, 1.35)
        y = render_wrapped_text(
            fig,
            0.08,
            0.64,
            "Table 1. Final-budget results at 50 measurements. Gap reduction is relative to Random Search and win rate is the percentage of paired runs where RF-SMBO attains a lower final normalised gap.",
            fontsize=8.8,
            width=96,
        )
        image_ax = fig.add_axes([0.09, 0.28, 0.82, 0.28])
        image_ax.imshow(plt.imread(figure_2_path))
        image_ax.axis("off")
        y = render_wrapped_text(
            fig,
            0.08,
            0.24,
            "Figure 2. Per-system final-budget comparison. RF-SMBO wins clearly on LLVM and x264, but loses on 7z.",
            fontsize=9,
            width=92,
        )
        y = render_wrapped_text(
            fig,
            0.08,
            y - 0.01,
            (
                f"RF-SMBO improves strongly on LLVM (gap reduction {llvm.gap_reduction_pct:.2f}%, p = {format_p_value(llvm.p_value)}, "
                f"win rate {llvm.win_rate_pct:.1f}%) and x264 (gap reduction {x264.gap_reduction_pct:.2f}%, "
                f"p = {format_p_value(x264.p_value)}, win rate {x264.win_rate_pct:.1f}%). "
                f"On 7z, however, the mean gap increases from {z7.random_gap:.4f} to {z7.rf_gap:.4f}, so the model-guided strategy is worse than the baseline there."
            ),
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, y = new_page()
        y = render_wrapped_text(fig, 0.08, y, "Reflection and Conclusion", fontsize=13, weight="bold", width=62)
        reflection_paragraphs = [
            (
                f"The 7z result should be treated as a valid negative result, not hidden. RF-SMBO fails on that dataset, "
                f"with a higher final mean gap ({z7.rf_gap:.4f}) than Random Search ({z7.random_gap:.4f}). "
                "This shows that the surrogate model is useful overall but not universally reliable."
            ),
        ]
        for paragraph in reflection_paragraphs:
            y = render_wrapped_text(fig, 0.08, y - 0.008, paragraph)
        y -= 0.004
        y = render_bullets(
            fig,
            0.10,
            y,
            [
                "The 7z search landscape may be rugged or deceptive, making it difficult for a small surrogate to rank promising configurations.",
                "A very small measurement budget may not provide enough training data for stable model fitting.",
                "Random-forest uncertainty estimated from tree disagreement is only an approximation and may be imperfect.",
                "Some configuration spaces may reward broader exploration more than model-guided exploitation.",
            ],
            fontsize=9.6,
        )
        y -= 0.002
        y = render_wrapped_text(
            fig,
            0.08,
            y,
            "These observations strengthen the submission rather than weaken it, because they show where the method works and where it does not.",
        )
        y -= 0.006
        y = render_wrapped_text(fig, 0.08, y, "Future Work", fontsize=11.5, weight="bold", width=60)
        y = render_bullets(
            fig,
            0.10,
            y - 0.006,
            [
                "Expected Improvement or rank-based acquisition.",
                "An adaptive exploration coefficient instead of the fixed value 0.5.",
                "Comparison against local search, genetic algorithms, or A-T-EGLS.",
                "Multi-objective runtime-energy tuning.",
                "Additional systems from Lab 3.",
            ],
            fontsize=9.6,
        )
        y -= 0.002
        conclusion = (
            f"Overall, RF-SMBO improves the mean normalised gap under limited measurement budgets, mainly due to strong gains on LLVM and x264, "
            f"but it underperforms Random Search on 7z. The final submission therefore presents a promising tool together with a scientifically meaningful negative result."
        )
        y = render_wrapped_text(fig, 0.08, y, conclusion)
        y -= 0.004
        y = render_wrapped_text(fig, 0.08, y, "Artifact", fontsize=11.5, weight="bold", width=60)
        y = render_wrapped_text(
            fig,
            0.08,
            y - 0.006,
            (
                f"Artifact link: {artifact_link}. The repository includes source code, raw CSV outputs, generated figures, "
                "requirements.pdf, manual.pdf, and replication.pdf."
            ),
        )
        y -= 0.002
        y = render_wrapped_text(fig, 0.08, y, "References", fontsize=11.5, weight="bold", width=60)
        references = [
            '[1] J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," Journal of Machine Learning Research, vol. 13, no. 10, pp. 281-305, 2012.',
            '[2] F. Hutter, H. H. Hoos, and K. Leyton-Brown, "Sequential Model-Based Optimization for General Algorithm Configuration," LION 5 / UBC TR-2010-10, 2011.',
            '[3] J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," NeurIPS, 2012.',
            '[4] P. Jamshidi and G. Casale, "An Uncertainty-Aware Approach to Optimal Configuration of Stream Processing Systems," MASCOTS, 2016.',
            '[5] V. Nair, T. Menzies, N. Siegmund, and S. Apel, "Using Bad Learners to find Good Configurations," ESEC/FSE, 2017.',
            '[6] ideas-labo, "ISE Lab 3 repository," https://github.com/ideas-labo/ISE/tree/main/lab3',
        ]
        render_bullets(fig, 0.10, y - 0.006, references, fontsize=8.6, width=88, gap=0.004)
        pdf.savefig(fig)
        plt.close(fig)

    return output_path


def build_text_pdf(title: str, sections: list[dict[str, object]], output_path: Path) -> Path:
    actual_output_path = resolve_writable_pdf_path(output_path)
    with PdfPages(actual_output_path) as pdf:
        fig, y = new_page()
        y = render_wrapped_text(fig, 0.08, y, title, fontsize=16, weight="bold", width=60)
        y -= 0.01
        for section in sections:
            fig, y = add_document_section(
                pdf,
                fig,
                y,
                section["heading"],
                paragraphs=section.get("paragraphs"),
                bullets=section.get("bullets"),
                code_blocks=section.get("code_blocks"),
            )
        pdf.savefig(fig)
        plt.close(fig)
    return actual_output_path


def build_support_pdfs(overall: pd.DataFrame, statistics: pd.DataFrame, artifact_link: str) -> list[Path]:
    versions = package_versions()
    requirements_sections = [
        {
            "heading": "Environment",
            "paragraphs": [
                "The project was tested on Windows 11. The scripts are standard Python programs and should also run on Linux or macOS provided that the same package versions are installed.",
                f"Python version used for the final submission: {versions['python']}.",
            ],
            "bullets": [
                f"numpy=={versions['numpy']}",
                f"pandas=={versions['pandas']}",
                f"scikit-learn=={versions['scikit-learn']}",
                f"scipy=={versions['scipy']}",
                f"matplotlib=={versions['matplotlib']}",
            ],
        },
        {
            "heading": "Installation",
            "paragraphs": [
                "A clean environment should install dependencies from requirements.txt rather than relying on the optional local vendor cache.",
            ],
            "code_blocks": [("Command", ["python -m pip install -r requirements.txt"])],
        },
        {
            "heading": "Repository Assumptions",
            "bullets": [
                "Run commands from the project root directory.",
                "Datasets should be stored in data/ using the filenames documented in data/README.md.",
                "Final generated outputs are written to outputs/final_run/.",
            ],
        },
    ]
    outputs = [build_text_pdf("Requirements", requirements_sections, PROJECT_ROOT / "requirements.pdf")]

    manual_sections = [
        {
            "heading": "Purpose",
            "paragraphs": [
                "The tool addresses Lab 3: Configuration Performance Tuning. It compares a Random Search baseline against RF-SMBO, a random-forest-guided sequential optimiser that operates under a fixed measurement budget.",
            ],
        },
        {
            "heading": "Expected Input",
            "paragraphs": [
                "The experiment runner expects CSV datasets in data/. Each file should contain one valid configuration per row, configuration options in the feature columns, and a final performance column containing the runtime to minimise.",
                "The included Lab 3 datasets already follow this structure and can be used directly without reformatting.",
            ],
        },
        {
            "heading": "How to Run the Tool",
            "code_blocks": [
                (
                    "Final experiment command",
                    [
                        "python run_experiments.py \\",
                        "  --systems 7z LLVM x264 \\",
                        "  --budgets 10 20 30 40 50 \\",
                        "  --repeats 30 \\",
                        "  --seed 20260424 \\",
                        "  --initial-design 6 \\",
                        "  --candidate-pool 4096 \\",
                        "  --kappa 0.5 \\",
                        "  --explore-prob 0.1 \\",
                        "  --output-dir outputs/final_run",
                    ],
                ),
                (
                    "Rebuild report PDFs",
                    [f'python build_submission_docs.py --artifact-link "{artifact_link}"'],
                ),
            ],
        },
        {
            "heading": "Output Files",
            "bullets": [
                "raw_results.csv: one row per measured step, including system, algorithm, repeat, seed, sampled index, and current best value.",
                "summary_results.csv: aggregated means and standard deviations by system, algorithm, and budget.",
                "statistics.csv: one-sided paired Wilcoxon tests and RF-SMBO win rates.",
                "figure_1_convergence.png and figure_2_per_system.png: report-ready matplotlib figures.",
            ],
        },
        {
            "heading": "How to Interpret Results",
            "bullets": [
                "Lower runtime is better.",
                "Lower normalised gap is better.",
                "The win rate is the percentage of paired runs where RF-SMBO ends with a smaller final normalised gap than Random Search.",
                "Small p-values provide evidence that RF-SMBO beats the baseline on that system and budget.",
            ],
        },
    ]
    outputs.append(build_text_pdf("Manual", manual_sections, PROJECT_ROOT / "manual.pdf"))

    final_budget = get_final_budget(overall)
    random_gap, rf_gap, _ = compute_overall_gap_reduction(overall, final_budget)
    llvm_p = format_p_value(
        float(statistics[(statistics["system"] == "LLVM") & (statistics["budget"] == final_budget)]["p_value"].iloc[0])
    )
    x264_p = format_p_value(
        float(statistics[(statistics["system"] == "x264") & (statistics["budget"] == final_budget)]["p_value"].iloc[0])
    )
    replication_sections = [
        {
            "heading": "Step 1: Prepare the Repository",
            "paragraphs": [
                "Clone or copy the repository into a clean working directory, then verify that the CSV datasets exist in data/. The repository already includes the required Lab 3 files.",
            ],
        },
        {
            "heading": "Step 2: Install Dependencies",
            "code_blocks": [("Command", ["python -m pip install -r requirements.txt"])],
        },
        {
            "heading": "Step 3: Run the Final Experiments",
            "code_blocks": [
                (
                    "Experiment command",
                    [
                        "python run_experiments.py \\",
                        "  --systems 7z LLVM x264 \\",
                        "  --budgets 10 20 30 40 50 \\",
                        "  --repeats 30 \\",
                        "  --seed 20260424 \\",
                        "  --initial-design 6 \\",
                        "  --candidate-pool 4096 \\",
                        "  --kappa 0.5 \\",
                        "  --explore-prob 0.1 \\",
                        "  --output-dir outputs/final_run",
                    ],
                )
            ],
        },
        {
            "heading": "Step 4: Rebuild Figures, Tables, and PDFs",
            "code_blocks": [
                ("Command", [f'python build_submission_docs.py --artifact-link "{artifact_link}"'])
            ],
        },
        {
            "heading": "Expected Outputs",
            "bullets": [
                "outputs/final_run/raw_results.csv",
                "outputs/final_run/summary_results.csv",
                "outputs/final_run/statistics.csv",
                "outputs/final_run/figure_1_convergence.png",
                "outputs/final_run/figure_2_per_system.png",
                "coursework_report.pdf, requirements.pdf, manual.pdf, replication.pdf",
            ],
        },
        {
            "heading": "Reproducibility Notes",
            "paragraphs": [
                "The base random seed is 20260424. Matched seeds are used so that each Random Search run is paired with an RF-SMBO run under the same system and repeat index.",
                "Optimum and worst values are used only after search to compute the normalised gap. They are not consulted during candidate selection.",
                f"At budget {final_budget}, the expected overall mean normalised gap is {random_gap:.4f} for Random Search and {rf_gap:.4f} for RF-SMBO. Expected headline p-values are {llvm_p} for LLVM and {x264_p} for x264.",
            ],
        },
    ]
    outputs.append(build_text_pdf("Replication", replication_sections, PROJECT_ROOT / "replication.pdf"))
    return outputs


def main() -> None:
    args = parse_args()
    summary, statistics, overall, metadata, _raw = load_results(args.result_dir)
    write_markdown_files(summary, statistics, overall, metadata, args.artifact_link)
    report_path = build_coursework_report_pdf(summary, statistics, overall, metadata, args.result_dir, args.artifact_link)
    support_paths = build_support_pdfs(overall, statistics, args.artifact_link)

    print(report_path.resolve())
    for path in support_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
