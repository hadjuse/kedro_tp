# 🚀 MLOps Pipeline on GCP with Kedro

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow)](https://mlflow.org/)
[![GCS](https://img.shields.io/badge/storage-GCS-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/storage)
[![Ruff](https://img.shields.io/badge/linting-ruff-000000)](https://docs.astral.sh/ruff/)

> 🎓 **Course project** — Built as part of a Machine Learning Engineering / MLOps curriculum. The goal is to apply industry-standard practices (pipeline orchestration, experiment tracking, cloud storage, code quality) in an end-to-end academic setting.

A production-grade MLOps pipeline built with **Kedro 1.2**, **LightGBM**, **MLflow**, and **Google Cloud Storage**. The project follows engineering best practices: modular pipeline structure, experiment tracking, hyperparameter optimization, automated code quality checks, and a clean data engineering convention.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Running the Pipeline](#running-the-pipeline)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Google Cloud Storage Integration](#google-cloud-storage-integration)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Working with Notebooks](#working-with-notebooks)
- [Project Conventions](#project-conventions)

---

## Overview

This project implements a reproducible machine learning pipeline using Kedro's framework for pipeline orchestration. It includes:

- **Data ingestion and preprocessing** following Kedro's layered data engineering convention
- **Model training** with LightGBM and hyperparameter tuning via Hyperopt
- **Experiment tracking** with MLflow
- **Remote artifact storage** on Google Cloud Storage
- **Code quality enforcement** via Ruff (linting + formatting) and pre-commit hooks

---

## Tech Stack

| Layer | Tool |
|---|---|
| Pipeline orchestration | Kedro 1.2 |
| ML model | LightGBM |
| Hyperparameter tuning | Hyperopt |
| Data manipulation | Pandas 2.x |
| Experiment tracking | MLflow |
| Remote storage | Google Cloud Storage |
| Linting / Formatting | Ruff |
| Pre-commit hooks | pre-commit + Ruff |
| Testing | pytest + pytest-cov |
| Python | ≥ 3.10 |

---

## Project Structure

```
MLOPS_GCP_KEDRO/
├── conf/                   # Kedro configuration (parameters, catalog, logging)
│   ├── base/               # Shared config (committed)
│   └── local/              # Local overrides and credentials (gitignored)
├── data/                   # Local data layers (raw → intermediate → primary → model output)
├── docs/source/            # Sphinx documentation source
├── notebooks/              # Exploratory notebooks
├── src/
│   └── project_mlops/      # Main package
│       ├── pipelines/      # Kedro pipeline definitions
│       └── settings.py
├── tests/                  # Unit and integration tests
├── .pre-commit-config.yaml # pre-commit hooks (Ruff check + format)
├── pyproject.toml          # Project metadata and tool configuration
├── requirements.txt        # Core dependencies
└── ruff.toml               # Ruff linter configuration
```

---

## Getting Started

### Prerequisites

- Python **3.10+**
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- A Google Cloud project with a GCS bucket (for remote data storage)
- MLflow tracking server or local tracking (see [MLflow section](#experiment-tracking-with-mlflow))

### Installation

**Using `uv` (recommended — lock file is provided):**

```bash
git clone https://github.com/hadjuse/MLOPS_GCP_KEDRO.git
cd MLOPS_GCP_KEDRO

uv sync
```

**Using `pip`:**

```bash
git clone https://github.com/hadjuse/MLOPS_GCP_KEDRO.git
cd MLOPS_GCP_KEDRO

pip install -e .
# or
pip install -r requirements.txt
```

**Install pre-commit hooks:**

```bash
pre-commit install
```

### Environment Variables

Sensitive configuration (GCS credentials, MLflow URI, etc.) must **never** be committed. Create a `conf/local/credentials.yml` file:

```yaml
# conf/local/credentials.yml
gcs:
  project: your-gcp-project-id

mlflow:
  tracking_uri: http://your-mlflow-server:5000
```

You can also use a `.env` file at the project root (loaded via `python-dotenv`):

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Running the Pipeline

Run the full pipeline:

```bash
kedro run
```

Run a specific named pipeline:

```bash
kedro run --pipeline <pipeline_name>
```

Visualise the pipeline graph in your browser:

```bash
kedro viz
```

---

## Experiment Tracking with MLflow

MLflow is used to log parameters, metrics, and model artifacts. To start a local tracking server:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. Configure the tracking URI in `conf/local/credentials.yml` or via the `MLFLOW_TRACKING_URI` environment variable.

---

## Google Cloud Storage Integration

> ⚠️ **The GCS bucket must be created and configured before running the pipeline.** Remote datasets will not resolve without it.

Remote datasets are managed via `kedro-datasets` with the `google-cloud-storage` backend. Datasets stored on GCS are defined in `conf/base/catalog.yml`. Authentication is handled through the `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to a GCP service account key file.

### Deploying the bucket

Create the bucket using the `gcloud` CLI:

```bash
gcloud storage buckets create gs://<your-bucket-name> \
  --project=<your-gcp-project-id> \
  --location=EU \
  --uniform-bucket-level-access
```

Or via the [GCP Console](https://console.cloud.google.com/storage/browser) → Cloud Storage → Create bucket.

Once created, update your `conf/local/credentials.yml` with the bucket name and make sure the service account has the **Storage Object Admin** role on it:

```bash
gcloud projects add-iam-policy-binding <your-gcp-project-id> \
  --member="serviceAccount:<sa-name>@<your-gcp-project-id>.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

---

## Code Quality

This project uses **Ruff** for both linting and formatting, enforced at commit time via pre-commit.

Run linting manually:

```bash
ruff check src/
```

Run formatting:

```bash
ruff format src/
```

Run all pre-commit hooks on staged files:

```bash
pre-commit run
```

Run on all files:

```bash
pre-commit run --all-files
```

The Ruff configuration is defined in `ruff.toml`. Active rule sets include Pyflakes (`F`), pycodestyle (`E`, `W`), isort (`I`), pyupgrade (`UP`), Pylint (`PL`), and print statement detection (`T201`).

---

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=src/project_mlops --cov-report=term-missing
```

Test files live in `tests/`. Coverage configuration is defined in `pyproject.toml` under `[tool.coverage.report]`.

---

## Working with Notebooks

Kedro injects `context`, `session`, `catalog`, and `pipelines` into notebook scope automatically.

Start a Jupyter notebook server:

```bash
kedro jupyter notebook
```

Start JupyterLab:

```bash
kedro jupyter lab
```

Start an IPython session:

```bash
kedro ipython
```

> **Tip:** Use [`nbstripout`](https://github.com/kynan/nbstripout) to automatically strip notebook outputs before committing:
> ```bash
> nbstripout --install
> ```

---

## Project Conventions

- **Do not commit data** — all data directories are gitignored.
- **Do not commit credentials** — keep secrets in `conf/local/`, never in `conf/base/`.
- **Follow the data engineering layers** — `raw → intermediate → primary → feature → model input → model output → reporting`.
- Pipeline results must be **reproducible**: parametrise everything via `conf/base/parameters.yml`.

---

## Resources

- [Kedro documentation](https://docs.kedro.org)
- [LightGBM documentation](https://lightgbm.readthedocs.io)
- [MLflow documentation](https://mlflow.org/docs/latest/index.html)
- [Hyperopt documentation](http://hyperopt.github.io/hyperopt/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [Google Cloud Storage Python client](https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python)
