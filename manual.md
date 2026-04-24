# Manual

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
python build_submission_docs.py --artifact-link "https://github.com/zhaibuqun-cell/rf-smbo-configuration-tuning"
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
