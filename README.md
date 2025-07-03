# plant-disease-mlops
MLOps Zoomcamp final project: Plant disease classification: build an ML service to classify plant leaf images into disease categories (healthy vs various infections). The project starts from a standard dataset (PlantVillage) to grow it into a full MLOps pipeline with orchestration, deployment, testing, and monitoring. The project uses CNNs with MLflow, Prefect, Docker, and monitoring tools. It's folder structure is below:

```text
plant-disease-mlops/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── training/
│   ├── inference/
│   └── monitoring/
├── Dockerfile
├── prefect-flow.py
├── requirements.txt
└── README.md
```


### 🛠️ Project Tool Breakdown

| *Module*              | *Zoomcamp Tools*                   | *Description*                                                                   |
|-----------------------|------------------------------------|---------------------------------------------------------------------------------|
| Data/Training Pipeline| Prefect, Docker, MLflow            | Automate data ingestion, augmentation, training with MLflow tracking & registry |
| CI/CD                 | GitHub Actions                     | Run linting, tests, and trigger flows automatically                             |
| Model Serving         | FastAPI, Docker, MLflow            | Containerized inference API with models loaded from MLflow registry             |
| Deployment            | Docker Compose / Terraform         | Deploy app, Prefect agents, and serving endpoint using infra-as-code            |
| Monitoring            | Evidently, Prometheus, Grafana     | Track data/prediction drift and log model metrics                               |
| Retraining Trigger    | Prefect, Alerts (custom logic)     | Trigger retraining when drift or performance drops detected                     |
| E2E Tests             | pytest, integration tests          | Test end-to-end pipeline from ingestion to prediction                           |



The architecture diagram below gives a quick overview of the project's scope and design:


<img src="images/MLOps-Plant-Disease-Classifier.png" alt="Architecture Diagram" width="800"/>

