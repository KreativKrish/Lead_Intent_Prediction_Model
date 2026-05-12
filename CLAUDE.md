# Lead Intent Prediction Model - Claude Code Context

## Project Overview

Production-grade machine learning pipeline for predicting lead intent at scale. End-to-end system: data ingestion from Snowflake → feature engineering → model training with MLflow → FastAPI serving.

**Key Purpose**: Identify high-intent leads for enterprise sales teams using XGBoost classification.

## Architecture

### Data Flow
```
Snowflake (data warehouse)
  ↓
Airflow (orchestration)
  ├─ data_ingestion_dag.py       (daily extraction)
  ├─ feature_engineering_dag.py  (feature transforms)
  ├─ model_training_dag.py       (weekly training)
  └─ model_deployment_dag.py     (manual promotion)
  ↓
MLflow (tracking + registry)
  └─ Models staged: None → Staging → Production
  ↓
FastAPI (inference API)
  ├─ /api/predict              (single predictions)
  └─ /api/predict/batch        (batch predictions)
```

### Stack Components

| Component | Purpose | Port |
|-----------|---------|------|
| Snowflake | Data warehouse | N/A (cloud) |
| MLflow | Model tracking & registry | 5000 |
| Airflow | Workflow orchestration | 8080 |
| PostgreSQL | Airflow metadata store | 5432 |
| FastAPI | Inference API | 8000 |

## Key Files & Modules

### Core ML Library (`src/`)

- **`data/`**: Snowflake connector + data loading
  - `snowflake_connector.py` — Context manager for SF sessions
  - `data_loader.py` — Load train/test splits
  - `validators.py` — Data quality checks (pandera/GE)

- **`features/`**: Feature engineering pipeline
  - `pipeline.py` — sklearn Pipeline assembly (numerical + categorical)
  - `encoders.py` — OneHotEncoder for categorical features
  - `selectors.py` — VarianceThreshold feature selection

- **`models/`**: Model definitions + registry
  - `lead_intent_model.py` — XGBoost classifier wrapper
  - `registry.py` — MLflow model loading/registration/promotion

- **`training/`**: Training orchestration
  - `trainer.py` — Main training loop with MLflow autolog
  - `evaluator.py` — Metrics computation (AUC, F1, precision, recall)
  - `hp_tuning.py` — Optuna hyperparameter search (skeleton)

- **`utils/`**: Shared utilities
  - `config.py` — Dynaconf config loader (YAML + env interpolation)
  - `logger.py` — structlog setup (JSON format)
  - `mlflow_utils.py` — MLflow manager (runs, registration, promotion)

### API (`api/`)

- **`main.py`** — FastAPI app factory with lifespan management
- **`schemas.py`** — Pydantic v2 request/response models
- **`dependencies.py`** — Dependency injection (model loading, request IDs)
- **`routers/`**:
  - `predict.py` — POST /api/predict, POST /api/predict/batch
  - `health.py` — GET /health, GET /ready

### Orchestration (`dags/`)

- **`data_ingestion_dag.py`** — Extract raw data from Snowflake
- **`feature_engineering_dag.py`** — Transform features, materialize store
- **`model_training_dag.py`** — Train, evaluate, register to MLflow
- **`model_deployment_dag.py`** — Promote model through stages

### Configuration (`config/`)

- **`base.yaml`** — Shared defaults (Snowflake, MLflow, training hyperparams)
- **`development.yaml`** — Dev overrides (smaller models, local URIs)
- **`staging.yaml`** — Staging overrides (medium models)
- **`production.yaml`** — Production overrides (full models, metrics thresholds)

### Testing (`tests/`)

- **`conftest.py`** — pytest fixtures (sample data, mock connectors)
- **`unit/`** — Model, feature, schema tests (mocked deps)
- **`integration/`** — API tests with TestClient
- **`e2e/`** — Full pipeline tests (slow)

### Deployment (`docker/`)

- **`api.Dockerfile`** — FastAPI container
- **`mlflow.Dockerfile`** — MLflow tracking server
- **`airflow.Dockerfile`** — Airflow scheduler + webserver
- **`docker-compose.yml`** — Local dev stack (all services)

## Development Workflow

### 1. Local Setup
```bash
cp .env.example .env
# Edit .env with real Snowflake credentials
docker-compose up -d
```

### 2. Train Model
```bash
python -m src.training.trainer  # Starts MLflow run, logs to registry
```

### 3. Invoke API
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"lead_score": 75, "company_size": 100, ...}'
```

### 4. Check MLflow
- UI: http://localhost:5000
- View runs, metrics, artifacts, registered models

### 5. Trigger Airflow
- UI: http://localhost:8080
- Manually trigger DAGs or view scheduled runs

## Important Conventions

### Configuration
- All config via YAML in `config/` (environment-specific)
- Environment variable interpolation: `"@{VAR_NAME|default}"`
- Dynaconf singleton pattern for accessing config

### Logging
- structlog for structured JSON logging
- Level controlled by `LOG_LEVEL` env var
- Logs written to `logs/app.log` + stdout

### MLflow
- Autolog enabled for sklearn/xgboost (logs params, metrics, models)
- Models registered as `lead_intent_model`
- Stages: None (candidate) → Staging (tested) → Production (live)
- Promotion via `ModelRegistry.transition_stage(version, stage)`

### Testing
- Mocked Snowflake connector in unit tests
- TestClient for API integration tests
- Fixtures in `conftest.py` for common data/objects

## Common Tasks

### Add a new feature
1. Add column to `features.numerical_features` or `.categorical_features` in `config/base.yaml`
2. Update `FeaturePipeline.build_pipeline()` if custom transform needed
3. Test in `tests/unit/test_features.py`

### Change model hyperparameters
1. Edit `training.model.xgboost` in `config/<env>.yaml`
2. Retrain: `python -m src.training.trainer`
3. Check metrics in MLflow UI

### Promote model to production
```bash
python scripts/promote_model.py --version 2 --stage Production
```

### Debug Snowflake issues
```python
from src.data import SnowflakeConnector
connector = SnowflakeConnector()
with connector.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM TRAINING_DATA")
    print(cursor.fetchone())
```

### View training metrics
1. Open MLflow UI: http://localhost:5000
2. Click experiment: `lead_intent_prediction`
3. Select run to see params, metrics, artifacts

## Dependencies

- **Production**: pandas, numpy, sklearn, xgboost, mlflow, fastapi, snowflake-connector
- **Development**: pytest, pytest-cov, ruff, mypy, jupyter
- See `requirements.txt` and `requirements-dev.txt`

## Known Limitations & TODOs

- Hyperparameter tuning (hp_tuning.py) is skeleton — implement full Optuna integration
- No SHAP explainability yet (TODO: add in evaluator)
- Model deployment to cloud (Kubernetes manifests coming)
- Feature store materialization not implemented (TODO: use Delta Lake or Feast)
- Real-time monitoring dashboard (TODO: Grafana + Prometheus)

## Troubleshooting Checklist

- **Snowflake connection fails**: Check `.env` creds, network, warehouse running
- **API won't start**: `docker logs api` — check logs, verify model load in MLflow
- **Airflow DAG missing**: Copy to `dags/` folder, restart scheduler
- **MLflow artifacts lost**: Check artifact root is writable, S3/cloud creds if remote
- **Tests fail**: Run `pytest tests/unit -v` for detailed output, check conftest fixtures

## Future Enhancements

1. **Feature Store**: Materialize features in Snowflake/Feast
2. **Hyperparameter Tuning**: Full Optuna integration with MLflow tracking
3. **Model Explainability**: SHAP values for predictions
4. **Real-time Pipeline**: Kafka ingestion + batch scoring
5. **Kubernetes**: Helm charts for production deployment
6. **Monitoring**: Prometheus metrics + Grafana dashboards
7. **CI/CD**: GitHub Actions for automated testing + deployment

---

**Last Updated**: 2026-05-12  
**Owner**: ML Platform Team  
**Status**: Production Ready
