# Lead Intent Prediction Model

Production-grade end-to-end ML pipeline for lead intent prediction. Features automated data processing, model training, evaluation, and API serving.

## Tech Stack

- **Data Warehouse**: Snowflake
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM
- **Experiment Tracking**: MLflow
- **Orchestration**: Apache Airflow
- **API**: FastAPI
- **Deployment**: Docker & Docker Compose

## Project Structure

```
├── api/                 # FastAPI inference service
├── src/                 # Core ML library
│   ├── data/           # Data loading & validation
│   ├── features/       # Feature engineering
│   ├── models/         # Model definitions
│   ├── training/       # Training & evaluation
│   └── utils/          # Configuration & logging
├── dags/               # Airflow DAGs
├── config/             # Configuration files
├── tests/              # Test suite
├── scripts/            # Utility scripts
└── docker/             # Docker images
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Snowflake account (with database, schema, warehouse)

### Setup

1. **Clone and install dependencies**:
   ```bash
   git clone <repo>
   cd Lead_Intent_Prediction_Model
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your Snowflake credentials
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

   Services will be available at:
   - API: http://localhost:8000
   - MLflow: http://localhost:5000
   - Airflow: http://localhost:8080

### Development

**Train a model**:
```bash
python -m src.training.trainer
```

**Run tests**:
```bash
pytest tests/unit -v
pytest tests/integration -v
```

**Start API locally**:
```bash
uvicorn api.main:app --reload
```

**Access Airflow**:
- URL: http://localhost:8080
- Default credentials: admin/admin

## API Endpoints

### Health & Status

- `GET /health` - Service health check
- `GET /ready` - Readiness probe

### Predictions

- `POST /api/predict` - Single prediction
  ```json
  {
    "lead_score": 75,
    "company_size": 100,
    ...
  }
  ```

- `POST /api/predict/batch` - Batch predictions

### Documentation

- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

## Data Pipeline

1. **Data Ingestion** → Extract from Snowflake
2. **Feature Engineering** → Transform raw features
3. **Training** → Train XGBoost model with MLflow tracking
4. **Evaluation** → Compute metrics (AUC, F1, Precision, Recall)
5. **Registration** → Register to MLflow model registry
6. **Deployment** → Promote model to Production stage

## Configuration

Environment-specific configs in `config/`:

- `base.yaml` - Shared configuration
- `development.yaml` - Development overrides
- `staging.yaml` - Staging overrides
- `production.yaml` - Production overrides

Configuration is loaded via Dynaconf with environment variable interpolation.

## Model Versions

Models are versioned and tracked in MLflow:

- **Stages**: None (unregistered) → Staging → Production
- **Registry**: `lead_intent_model` (configurable)
- **Tracking URI**: http://localhost:5000

Register a model after training:
```bash
python scripts/promote_model.py --version 1 --stage Staging
python scripts/promote_model.py --version 1 --stage Production
```

## Monitoring & Observability

- **Structured Logging**: JSON formatted logs to `logs/app.log`
- **Metrics**: Prometheus-compatible metrics on port 8001
- **Traces**: Integrated with MLflow for experiment tracking
- **Model Monitoring**: Drift detection and alert configuration in `config/`

## Deployment

### Docker

Build and run containers:
```bash
docker build -f docker/api.Dockerfile -t lead-intent-api:latest .
docker run -p 8000:8000 lead-intent-api:latest
```

### Kubernetes

Create ConfigMap and Secret for credentials:
```bash
kubectl create secret generic snowflake-creds \
  --from-literal=account=xxx \
  --from-literal=user=xxx \
  --from-literal=password=xxx
```

Deploy using Helm charts (templates in `deploy/helm/` - coming soon)

## Testing

**Unit tests**:
```bash
pytest tests/unit -v --cov=src --cov=api
```

**Integration tests**:
```bash
pytest tests/integration -v
```

**E2E tests**:
```bash
pytest tests/e2e -v -m slow
```

## Troubleshooting

### Can't connect to Snowflake
- Verify credentials in `.env`
- Check network connectivity and firewall rules

### MLflow artifacts not saving
- Check artifact root directory is writable
- Verify S3/Azure credentials if using cloud storage

### Airflow DAG not triggering
- Check Airflow logs: `docker logs airflow-scheduler`
- Verify DAG syntax: `airflow dags validate dags/model_training_dag.py`

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest tests/`
3. Commit with conventional messages: `git commit -m "feat: add new feature"`
4. Push and create PR

## License

MIT License. See LICENSE file for details.

## Contact

Lead ML Team: ml-team@example.com
