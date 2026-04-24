# Requirements

## Runtime Environment

- Operating system assumption: tested on Windows 11, but the scripts are pure Python and should also run on Linux or macOS with the same package versions.
- Python version: 3.12.13

## Python Dependencies

- numpy==2.4.4
- pandas==3.0.1
- scikit-learn==1.8.0
- scipy==1.17.1
- matplotlib==3.10.9

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
