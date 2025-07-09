# 🌿 Plant Disease Classifier (MLOps Zoomcamp Final Project)

This project builds an ML service to classify plant leaf images into disease categories (e.g. healthy vs. infected). Starting with the standard PlantVillage dataset, the project evolves into a full MLOps pipeline with orchestration, training, deployment, monitoring, and CI/CD automation. Its folder structure is below:


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



### 🧭 Project Architecture

The architecture diagram below illustrates the full MLOps workflow, from data ingestion and training to deployment, serving, and monitoring. A mobile app integration may be explored in future iterations.



<img src="images/MLOps-Plant-Disease-Classifier.png" alt="Architecture Diagram" width="800"/>

### 🧠 Model Choice

We use google/vit-base-patch16-224, a Vision Transformer (ViT) pretrained on ImageNet, as the core model for classifying plant diseases. We will fine-tune it using publicly available datasets such as PlantVillage to adapt it to agricultural disease recognition.
In future iterations (beyond the scope of this project), we may explore lightweight or quantized variants of ViT for mobile deployment as a standalone extension.

## 🧹 Linting & Code Quality

This project uses **GitHub Actions** for Continuous Integration (CI) to automate quality checks and ensure consistent, production-ready code.

CI tasks include:

- ✅ **Fast code linting with `ruff`**  
  Runs on every push or pull request. It checks for PEP8 compliance and can auto-fix issues:

  ```bash
  ruff check . --fix
  ```

- ✅ **Running tests with `pytest`**  
  Ensures that core functionality behaves as expected.

- ✅ **(Optional) Docker image build**  
  Verifies that the app builds correctly in a containerized environment.

---

We also use **`pylint` locally** for deeper static analysis and complexity checks.  
Run it manually before major commits:

```bash
pylint src/
```

This dual approach gives us both speed (CI) and depth (manual analysis). 💪
