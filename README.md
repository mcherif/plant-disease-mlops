# 🌿 Plant Disease Classifier (MLOps Zoomcamp Final Project)

This project fine-tunes a Vision Transformer (ViT) model to classify plant leaf diseases using the PlantVillage dataset. It integrates modern MLOps tools for training, evaluation, reproducibility, and automation.

## 🚀 Features

- Vision Transformer (ViT) fine-tuning with Hugging Face Transformers
- PyTorch + torchvision training pipeline
- MLflow for experiment tracking
- Prefect for orchestration
- Docker containerization
- GitHub Actions CI with Ruff + Pytest

---

### 🧭 Project Architecture

The architecture diagram below illustrates the full MLOps workflow, from data ingestion and training to deployment, serving, and monitoring. A mobile app integration may be explored in future iterations.



<img src="images/MLOps-Plant-Disease-Classifier.png" alt="Architecture Diagram" width="800"/>

## ⚙️ Setup Instructions

### 💻 Local Development (with GPU support)

If you're developing locally and want to use a CUDA-optimized version of PyTorch:

```bash
pip install -r requirements-dev.txt
```

This will:

- Install all core dependencies from `requirements.txt`
- Enable CUDA support via `torch==2.1.0+cu118` (via `--extra-index-url`)
- Install development tools like `pytest`, `ruff`, and `pylint`

> ⚠️ Make sure your GPU and CUDA toolkit are compatible with this version of PyTorch. Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) if unsure.

---

### 🤖 Continuous Integration (CI) – GitHub Actions

CI is handled via **GitHub Actions** and uses `requirements.txt` to:

- ✅ Run fast code linting with [`ruff`](https://docs.astral.sh/ruff/)
- ✅ Run unit tests using `pytest`
- ✅ (Optionally) build a Docker image to ensure reproducibility

CI runs automatically on every push or pull request to the `main` branch.

---

## 🧹 Linting & Code Quality

This project uses **GitHub Actions** for Continuous Integration (CI) to automate quality checks and ensure consistent, production-ready code.

CI tasks include:

✅ **Fast code linting with `ruff`**  
Runs on every push or pull request. It checks for PEP8 compliance and can auto-fix issues:

```bash
ruff check . --fix
```

✅ **Running tests with `pytest`**  
Ensures that core functionality behaves as expected.

✅ **(Optional) Docker image build**  
Verifies that the app builds correctly in a containerized environment.

We also use `pylint` locally for deeper static analysis and complexity checks.
You should run it manually before major commits:

```bash
pylint src/
```

This dual approach gives us both speed (CI) and depth (manual analysis). 💪

### File structure
```
plant-disease-classifier/
├── data/                        # Raw and processed datasets
├── notebooks/                   # Jupyter notebooks for EDA & dev
├── src/                         # All source code
│   ├── preprocessing/           # Image loaders, augmentation, resizers
│   ├── training/                # Model training code
│   ├── inference/               # FastAPI & prediction utils
│   └── monitoring/              # Drift detection, logging
├── flows/
│   └── prefect_flow.py          # Main Prefect pipeline definition
├── models/                      # Saved models (optional if MLflow used)
├── tests/                       # Pytest files for unit/integration testing
├── Dockerfile                   # App Docker build instructions
├── docker-compose.yaml          # Service orchestration: app, MLflow, monitoring
├── requirements.txt             # Python dependencies for CI or production
├── requirements-dev.txt         # Python dependencies for development
├── mlruns/                      # Local MLflow tracking artifacts
├── README.md                    # Project overview and instructions
└── .gitignore                   # Files and folders to ignore in version control
```

### 🛠️ Project Tool Breakdown

| **Module**            | **Zoomcamp Tools**                 | **Description**                                                                |
|-----------------------|------------------------------------|---------------------------------------------------------------------------------|
| Data/Training Pipeline| Prefect, Docker, MLflow            | Automate data ingestion, augmentation, training with MLflow tracking & registry |
| CI/CD                 | GitHub Actions                     | Run linting, tests, and trigger flows automatically                             |
| Model Serving         | FastAPI, Docker, MLflow            | Containerized inference API with models loaded from MLflow registry             |
| Deployment            | Docker Compose / Terraform         | Deploy app, Prefect agents, and serving endpoint using infra-as-code            |
| Monitoring            | Evidently, Prometheus, Grafana     | Track data/prediction drift and log model metrics                               |
| Retraining Trigger    | Prefect, Alerts (custom logic)     | Trigger retraining when drift or performance drops detected                     |
| E2E Tests             | pytest, integration tests          | Test end-to-end pipeline from ingestion to prediction                           |


### 📁 Script Overview

| Script                | Purpose                                      | Usage Context                  | Runs MLflow? | CI-Safe? | Prefect Required |
|------------------------|----------------------------------------------|--------------------------------|--------------|----------|------------------|
| `prefect_flow.py`      | Full pipeline: data → train → evaluate       | Production runs, automation    | ✅ Yes       | ⚠️ No\*  | ✅ Yes           |
| `train_vit.py`         | Train ViT model with MLflow logging          | Fast local iteration, dev work | ✅ Yes       | ⚠️ No\*  | ❌ No            |
| `evaluate_test.py`     | Load + evaluate model, log metrics           | CI/CD model sanity checks      | ✅ Yes       | ✅ Yes   | ❌ No            |
| `demo_inference.ipynb` | Interactive inference on single/test images  | Local testing, model demos     | ❌ No        | ✅ Yes   | ❌ No            |

> \* Not CI-safe by default unless model files are present or test skipping is used.
