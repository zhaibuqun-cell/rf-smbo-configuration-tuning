# RF-SMBO Configuration Tuning

This repository contains the coursework submission project for **Intelligent Software Engineering - Tool Building Project**, using **Lab 3: Configuration Performance Tuning** as the selected problem. The tool compares a baseline **Random Search** against a proposed **Random-Forest Sequential Model-Based Optimisation (RF-SMBO)** approach under limited measurement budgets.

## Project Summary

- Problem: configuration performance tuning under a fixed measurement budget
- Baseline: Random Search
- Proposed method: RF-SMBO using `RandomForestRegressor` as the surrogate model
- Final evaluated systems: `7z`, `LLVM`, `x264`
- Final budgets: `10`, `20`, `30`, `40`, `50`
- Repetitions: `30`

The results show that RF-SMBO improves overall search quality, mainly due to strong gains on **LLVM** and **x264**, but it underperforms Random Search on **7z**. That negative result is kept intentionally and discussed in the report reflection section.

## Repository Structure

```text
lab3_project/
├── README.md
├── requirements.txt
├── run_experiments.py
├── build_submission_docs.py
├── coursework_report.md
├── coursework_report.pdf
├── manual.md
├── manual.pdf
├── requirements.md
├── requirements.pdf
├── replication.md
├── replication.pdf
├── data/
├── outputs/
│   └── final_run/
└── .gitignore
```

## Algorithms

- `Random Search`: samples unseen valid configurations uniformly at random without replacement.
- `RF-SMBO`: starts from a small random design, trains a random forest surrogate after each measurement, and uses predictive mean plus uncertainty to choose the next configuration.

Both algorithms search only over the valid rows already present in the dataset. Runtime values are revealed only when a configuration is selected for measurement. Global optimum and worst values are reserved for post-hoc evaluation.

## Dataset Instructions

The repository already includes the Lab 3 CSV datasets in `data/`. See [data/README.md](/C:/Users/zsanity/Documents/New%20project/lab3_project/data/README.md) for filenames, source information, and expected structure.

## Installation

Create or activate a Python environment, then install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

The final submission was tested with Python 3.12 on Windows 11. The scripts should also run on Linux or macOS provided that the same package versions are available.

## How to Run Experiments

From the project root:

```bash
python run_experiments.py --systems 7z LLVM x264 --budgets 10 20 30 40 50 --repeats 30 --seed 20260424 --initial-design 6 --candidate-pool 4096 --kappa 0.5 --explore-prob 0.1 --output-dir outputs/final_run
```

## How to Rebuild the Report PDFs

```bash
python build_submission_docs.py --artifact-link "https://github.com/zhaibuqun-cell/rf-smbo-configuration-tuning"
```

## Where Results Are Stored

The final generated outputs are stored in [outputs/final_run](</C:/Users/zsanity/Documents/New project/lab3_project/outputs/final_run>), including:

- `raw_results.csv`
- `summary_results.csv`
- `statistics.csv`
- `figure_1_convergence.png`
- `figure_2_per_system.png`
- `system_metadata.csv`

## Reproducibility Notes

- Base seed: `20260424`
- Paired seeds are shared by Random Search and RF-SMBO for fair comparison within each system and repeat.
- All objectives in the final study are minimisation tasks.
- Lower normalised gap is better.
- Optimum and worst values are used only for post-hoc evaluation and never during search.
