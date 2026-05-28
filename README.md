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

#### Full pipeline (recommended)

Run these three steps in order:

```bash
# 1. Build CRM feature table from raw Snowflake data (~2–5 min for 1.3M rows)
python scripts/build_feature_table.py

# 2. Derive 7 composite scores + stratified split → LEAD_INTENT_SCORED
python scripts/build_scored_table.py

# 3. Train on the 7 scores, register best model to MLflow Production
python scripts/train_from_snowflake.py --mode scored
```

#### Training script reference

```
python scripts/train_from_snowflake.py [OPTIONS]
```

| Option | Values | Default | Description |
|---|---|---|---|
| `--mode` | `scored` \| `raw` | `raw` | Feature source (see below) |
| `--run-name` | any string | auto | MLflow parent run name |
| `--tiers` | `all` \| `champion` | `all` | Which model tiers to train |
| `--test-size` | 0.0–1.0 | `0.20` | Test fraction — raw mode only |

#### Mode: `scored` (recommended)

```bash
python scripts/train_from_snowflake.py --mode scored
python scripts/train_from_snowflake.py --mode scored --run-name prod_v2
python scripts/train_from_snowflake.py --mode scored --tiers champion   # XGBoost only
```

- **Source table**: `LEAD_INTENT_SCORED`
- **Features**: 7 composite scores — `eligibility_score`, `demographic_score`, `quality_score`, `engagement_score`, `intent_score`, `campaign_score`, `lead_aging`
- **Target**: `converted` (1 = enrolled)
- **Split**: uses the pre-defined `split` column from the table (train 70% / val 20% / test 10%)
- **Evaluation**: winner selected on validation set; final metrics reported on the held-out test set

#### Mode: `raw`

```bash
python scripts/train_from_snowflake.py --mode raw
python scripts/train_from_snowflake.py --mode raw --test-size 0.25
python scripts/train_from_snowflake.py --mode raw --tiers champion --run-name raw_xgb
```

- **Source table**: `FEATURES_LEAD_INTENT`
- **Features**: 68 CRM features (numerical + categorical with one-hot encoding)
- **Target**: `is_interested` (1 = interested)
- **Split**: random stratified split at runtime (default 80/20)

#### Model tiers

Both modes run the same 3-tier competition and log nested MLflow child runs:

| Tier | Algorithm | Hyperparameters | Imbalance handling |
|---|---|---|---|
| `baseline` | Logistic Regression | C=1.0, max_iter=1000 | `class_weight=balanced` |
| `challenger` | Gradient Boosting | 200 trees, depth=4, lr=0.05 | implicit via leaf weights |
| `champion` | XGBoost | 300 trees, depth=6, lr=0.05 | `scale_pos_weight = neg/pos` |

The tier with the highest ROC-AUC wins. The winning pipeline (preprocessor + classifier) is registered as `lead_intent_model → Production` in MLflow.

Use `--tiers champion` to skip LR and GB and train only XGBoost — useful for quick iterations after the initial comparison.

#### Quick local test (no Snowflake required)

```bash
python scripts/train_local.py   # synthetic data, seeds MLflow for API testing
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

```
Snowflake CRM (PROD_DATABASE.CRM)
  └─ LEAD_MASTERS, LEAD_ACTIVITIES, CONTACT_MASTER,
     FACEBOOK_CAPI_EVENTS, VOICE_BOT_API_LOGS, USERS
         ↓  scripts/build_feature_table.py
MARKETING_DATABASE.LEAD_INTENT_ML.FEATURES_LEAD_INTENT
  (~1.3 M rows, 80+ engineered features, target: is_interested)
         ↓  scripts/build_scored_table.py
MARKETING_DATABASE.LEAD_INTENT_ML.LEAD_INTENT_SCORED
  (7 composite scores 0–100 + converted target + split column)
         ↓  scripts/train_from_snowflake.py --mode scored
MLflow Model Registry  →  FastAPI  →  Lead scoring UI
```

### Step 1 — Build the raw CRM feature table

```bash
python scripts/build_feature_table.py
```

Reads raw CRM tables from Snowflake and engineers 80+ features across 15 groups:
source quality, campaign performance, follow-up velocity, lead freshness, owner performance,
engagement intensity, NLP intent score, time-to-response, funnel progression, affordability
indicators, profile completeness, contact signals, Facebook CAPI, voicebot interactions.

Writes to `MARKETING_DATABASE.LEAD_INTENT_ML.FEATURES_LEAD_INTENT`.

### Step 2 — Build the scored feature table

```bash
python scripts/build_scored_table.py
```

Reads `FEATURES_LEAD_INTENT` and derives 7 interpretable composite scores (each 0–100):

| Score | What it measures | Key inputs |
|---|---|---|
| `eligibility_score` | Professional / affordability eligibility | `is_employed_professional`, `has_ctc`, `has_experience`, `has_qualification` |
| `demographic_score` | Profile completeness & richness | `profile_completeness_score`, `has_real_email`, `has_alternate_mobile` |
| `quality_score` | Lead data quality & authenticity | `is_duplicate`, `is_chatbot`, `channel_historical_cvr`, `is_repeat_contact` |
| `engagement_score` | Multi-channel engagement depth | `engagement_intensity_score`, `call_answer_rate`, `has_multi_channel_engagement` |
| `intent_score` | Behavioural purchase intent signals | `net_intent_score`, `contacted_within_1hr`, `has_positive_intent` |
| `campaign_score` | Source / campaign acquisition quality | `channel_historical_cvr`, `campaign_historical_cvr`, `has_campaign` |
| `lead_aging` | Lead freshness (exponential decay) | `lead_age_days` → `100 × exp(−age / 30)` |

#### Score calculation detail

All scores are clipped to **[0, 100]**. CVR-based inputs are percentile-normalised against the 95th percentile of that column across the full dataset (stored as `normalization_params` in the build output).

---

**`eligibility_score`** — professional credentials and affordability signals

```
eligibility_score =
    is_employed_professional  × 25   # has both company name AND designation
  + has_ctc                   × 20   # CTC / annual package on file
  + has_experience             × 15   # experience field populated
  + has_salary_increment_goal  × 15   # salary increment as motivation signal
  + has_qualification          × 15   # education level provided
  + is_domestic                × 10   # domestic lead (local payment methods)
```

---

**`demographic_score`** — profile completeness and demographic richness

```
demographic_score =
    (profile_completeness_score / 12) × 60   # 0–12 field count → 0–60
  + has_real_email               × 15         # non-default email address
  + has_alternate_mobile         × 10         # second contact number
  + has_best_time_to_call        × 10         # preferred contact window known
  + has_pain_points              × 5          # learning motivation captured
```

`profile_completeness_score` is itself a sum of 12 binary flags (email, gender, DOB, state, city, alternate mobile, designation, company, CTC, experience, qualification, pain points).

---

**`quality_score`** — lead data trustworthiness and channel quality

```
channel_cvr_norm = clip(channel_historical_cvr / p95_channel_cvr, 0, 1)

quality_score =
    (1 − is_duplicate)     × 30   # not a duplicate lead
  + has_real_email          × 20   # real (non-placeholder) email
  + (1 − is_chatbot)        × 15   # human-originated lead
  + channel_cvr_norm        × 25   # normalised channel conversion rate
  + is_repeat_contact       × 10   # contact has interacted before
```

---

**`engagement_score`** — depth of multi-channel engagement

```
eng_norm = clip(engagement_intensity_score / p95_engagement, 0, 1)
  where engagement_intensity_score = call_count×3 + whatsapp_count×2
                                    + email_count + sms_count

engagement_score =
    eng_norm               × 40   # weighted multi-channel activity (normalised)
  + call_answer_rate        × 25   # answered_calls / total_calls_logged
  + has_multi_channel_engagement × 20  # used ≥2 distinct channels
  + has_positive_intent     × 15   # positive keywords found in activity notes
```

---

**`intent_score`** — behavioural signals of purchase intent

```
net_norm = clip((net_intent_score.clip(−10, 15) + 10) / 25, 0, 1)
  where net_intent_score = positive_intent_kw_count − negative_intent_kw_count

intent_score =
    net_norm                  × 35   # NLP keyword balance (−10..+15 → 0..1)
  + contacted_within_1hr      × 25   # first contact made within 60 minutes
  + has_positive_intent        × 20   # at least one positive keyword present
  + (1 − has_negative_intent)  × 20   # no negative keywords in activity notes
```

Positive keywords include: *interested, want to enroll, fee, admission, callback, will join.*  
Negative keywords include: *not interested, wrong number, do not call, not responding, switched off.*

---

**`campaign_score`** — acquisition source and campaign quality

```
ch_cvr_norm   = clip(channel_historical_cvr  / p95_channel_cvr,   0, 1)
camp_cvr_norm = clip(campaign_historical_cvr / p95_campaign_cvr,  0, 1)

campaign_score =
    ch_cvr_norm   × 50   # channel's historical conversion rate (normalised)
  + camp_cvr_norm × 40   # campaign's historical conversion rate (normalised)
  + has_campaign  × 10   # lead arrived via a tracked campaign
```

Leads with no campaign get `campaign_historical_cvr = 0`, so their score is driven entirely by channel quality.

---

**`lead_aging`** — lead freshness via exponential decay

```
lead_aging = 100 × exp(−lead_age_days / 30)
```

| Lead age | Score |
|---|---|
| Day 0 (brand new) | 100.0 |
| Day 7 | 79.4 |
| Day 14 | 63.0 |
| Day 30 | 36.8 |
| Day 60 | 13.5 |
| Day 90 | 5.0 |

The 30-day denominator sets a half-life of ≈21 days, reflecting that leads older than a month convert at significantly lower rates.

---

**Target variable**: `converted` (= `is_interested`; 1 if lead enrolled, 0 otherwise)

**Train / validation / test split** (stratified on `converted`):

| Split | Fraction | Column value |
|---|---|---|
| Train | 70% | `'train'` |
| Validation | 20% | `'validation'` |
| Test | 10% | `'test'` |

Output table: `MARKETING_DATABASE.LEAD_INTENT_ML.LEAD_INTENT_SCORED`  
Schema: `lead_id, eligibility_score, demographic_score, quality_score, engagement_score, intent_score, campaign_score, lead_aging, converted, split, scored_at`

### Step 3 — Train and register

Two modes:

**Scored mode** (recommended — trains on the 7 interpretable features):
```bash
python scripts/train_from_snowflake.py --mode scored --run-name scored_v1
```
- Loads `LEAD_INTENT_SCORED`; uses the pre-defined `split` column
- Trains all 3 tiers on `train`, selects winner on `validation`, reports final metrics on `test`

**Raw mode** (68 CRM features):
```bash
python scripts/train_from_snowflake.py --mode raw
```
- Loads `FEATURES_LEAD_INTENT`; random stratified 80/20 split

Both modes run the **3-tier model competition**:

| Tier | Algorithm | Class imbalance handling |
|---|---|---|
| Baseline | Logistic Regression | `class_weight=balanced` |
| Challenger | Gradient Boosting (200 trees) | implicit via leaf counts |
| Champion | XGBoost (300 trees) | `scale_pos_weight` = neg/pos ratio |

The winner is registered to MLflow as `lead_intent_model → Production`.

### Step 4 — Evaluation → Registration → Deployment

1. **Evaluation** → AUC-ROC, PR-AUC, F1, Precision, Recall logged per tier
2. **Registration** → Winning pipeline (preprocessor + classifier) registered in MLflow
3. **Deployment** → Promote model to Production stage

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
