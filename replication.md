# Replication

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
python build_submission_docs.py --artifact-link "https://github.com/zhaibuqun-cell/rf-smbo-configuration-tuning"
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
- The expected overall mean normalised gap at budget 50 is:
  - Random Search: 0.0487
  - RF-SMBO: 0.0113
- Expected headline p-values at budget 50:
  - LLVM: 9.31e-10
  - x264: 9.10e-07

## Regenerating Figures and Tables

The report figures are generated directly by `run_experiments.py`, while Table 1 and the final PDFs are rebuilt by `build_submission_docs.py`. If the artifact link changes after GitHub upload, rerun `build_submission_docs.py` so that the PDFs contain the final repository URL.
