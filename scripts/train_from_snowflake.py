"""Train Lead Intent ensemble model from Snowflake.

Modes:
  raw    (default) — 68 CRM features from FEATURES_LEAD_INTENT, random 80/20 split
  scored           — 7 composite scores from LEAD_INTENT_SCORED, pre-defined 70/20/10 split
                     Target: converted (enrolled = 1)

Usage:
    python scripts/train_from_snowflake.py
    python scripts/train_from_snowflake.py --mode scored --run-name scored_v1
    python scripts/train_from_snowflake.py --run-name my_experiment --test-size 0.25
"""

import argparse
import os   
import re
import sys
import time
from pathlib import Path

# ── Load .env before importing project modules ─────────────────────────────────
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _v = _v.strip()
        if _v.startswith('"') and _v.endswith('"'):
            _v = _v[1:-1]
        else:
            _v = _v.split("#")[0].strip()
        os.environ.setdefault(_k.strip(), _v)

os.environ.setdefault("MLFLOW_TRACKING_URI",   "http://localhost:5000")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME","lead_intent_prediction")
os.environ.setdefault("MODEL_NAME",            "lead_intent_model")
os.environ.setdefault("ENV",                   "production")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from src.training.model_selector import ModelResult, ModelSelector

# ── Constants ──────────────────────────────────────────────────────────────────

FEATURE_TABLE = "MARKETING_DATABASE.LEAD_INTENT_ML.FEATURES_LEAD_INTENT"
TARGET_COL    = "is_interested"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# Numeric features — all safe for prediction at lead-assignment time.
# Excluded (leakage / identifiers / timestamps):
#   lead_id, contact_id               → identifiers, not predictive
#   is_interested                     → target
#   lead_date, created_at, updated_at, feature_computed_at → timestamps
#   reached_interested_stage          → direct label leakage (outcome)
#   hours_to_interested_stage         → direct label leakage (outcome)
#   lead_status_id/name               → encodes current stage which IS the label
#   lead_sub_status_id/name           → same risk as above
#   previous_lead_status_id/name      → can encode prior-interested state
#   days_in_current_status            → duplicate of days_since_last_update
#   owner_role_id                     → sparse numeric ID, not a real number
#   course_id, university_id          → high-cardinality IDs without ordinal meaning
NUMERIC_FEATURES = [
    # 1. Source quality
    "channel_historical_cvr",
    "channel_volume",
    # 2. Campaign performance
    "has_campaign",
    "campaign_historical_cvr",
    "campaign_volume",
    # 3. Follow-up velocity
    "followup_count",
    "hours_to_first_followup",
    "avg_followup_interval_hours",
    # 4. Lead freshness
    "lead_age_days",
    "days_since_last_update",
    "is_fresh_7d",
    "is_fresh_30d",
    "lead_hour_of_day",
    "lead_day_of_week",
    "lead_month",
    "is_weekend_lead",
    "is_business_hours_lead",
    # 5. Owner performance
    "is_owner_active",
    "owner_historical_cvr",
    "owner_workload_30d",
    "owner_tenure_days",
    "was_reassigned",
    # 6. Engagement intensity
    "call_count",
    "sms_count",
    "email_count",
    "whatsapp_count",
    "engagement_intensity_score",
    "has_multi_channel_engagement",
    "total_activities",
    "total_calls_logged",
    "answered_calls",
    "total_call_duration_sec",
    "avg_call_duration_sec",
    "max_call_duration_sec",
    "call_answer_rate",
    "calls_with_recording",
    "unique_activity_types",
    "days_since_last_activity",
    # 7. NLP intent score
    "positive_intent_kw_count",
    "negative_intent_kw_count",
    "net_intent_score",
    "has_positive_intent",
    "has_negative_intent",
    # 8. Time-to-response
    "minutes_to_first_contact",
    "contacted_within_1hr",
    "contacted_within_24hrs",
    "never_contacted",
    # 9. Funnel progression (safe: no label encoding)
    "status_changed",
    # 10. Affordability indicators
    "has_ctc",
    "has_experience",
    "has_company",
    "has_designation",
    "is_employed_professional",
    "has_salary_increment_goal",
    # 11. Profile completeness
    "has_real_email",
    "has_gender",
    "has_dob",
    "has_state",
    "has_city",
    "has_alternate_mobile",
    "has_best_time_to_call",
    "has_qualification",
    "has_pain_points",
    "profile_completeness_score",
    # 12. Misc flags
    "is_chatbot",
    "is_voicebot",
    "is_duplicate",
    # 13. Contact
    "contact_lead_count",
    "is_domestic",
    "is_repeat_contact",
    # 14. Facebook
    "is_facebook_lead",
    "fb_events_count",
    "fb_events_success",
    "fb_event_success_rate",
    # 15. Voicebot
    "has_voicebot_interaction",
    "voicebot_call_count",
    "voicebot_initiated_count",
]

CATEGORICAL_FEATURES = [
    "lead_channel",    # organic / facebook / referral / etc.
    "source_medium",   # cpc / email / social / etc.
    "lead_type",       # inbound / outbound / etc.
]

# ── Scored-mode constants ──────────────────────────────────────────────────────
SCORED_TABLE = "MARKETING_DATABASE.LEAD_INTENT_ML.LEAD_INTENT_SCORED"
SCORED_FEATURES = [
    "eligibility_score",
    "demographic_score",
    "quality_score",
    "engagement_score",
    "intent_score",
    "campaign_score",
    "lead_aging",
]
SCORED_TARGET = "converted"

# ── Snowflake auth ─────────────────────────────────────────────────────────────

def _load_private_key() -> bytes:
    raw  = os.environ.get("SNOWFLAKE_PRIVATE_KEY", "")
    path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "")
    pw   = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")

    if path and Path(path).exists():
        pem = Path(path).read_bytes()
    else:
        m = re.match(
            r"(-----BEGIN [^-]+-----)([A-Za-z0-9+/=\s]+)(-----END [^-]+-----)", raw
        )
        if not m:
            raise ValueError("Cannot locate private key — set SNOWFLAKE_PRIVATE_KEY or SNOWFLAKE_PRIVATE_KEY_PATH")
        h, b, f = m.groups()
        b = b.replace(" ", "").replace("\n", "")
        pem = (f"{h}\n" + "\n".join(b[i:i+64] for i in range(0, len(b), 64)) + f"\n{f}").encode()

    pk = serialization.load_pem_private_key(
        pem, password=pw.encode() if pw else None, backend=default_backend()
    )
    return pk.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _connect() -> snowflake.connector.SnowflakeConnection:
    pkb = _load_private_key()
    return snowflake.connector.connect(
        account   = os.environ["SNOWFLAKE_ACCOUNT"],
        user      = os.environ["SNOWFLAKE_USER"],
        private_key = pkb,
        warehouse = os.environ["SNOWFLAKE_WAREHOUSE"],
        role      = os.environ["SNOWFLAKE_ROLE"],
    )

# ── Data loading ───────────────────────────────────────────────────────────────

def load_feature_table(conn) -> pd.DataFrame:
    print(f"  Loading {FEATURE_TABLE} ...")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {FEATURE_TABLE}")
    rows = cur.fetchall()
    cols = [d[0].lower() for d in cur.description]
    cur.close()
    df = pd.DataFrame(rows, columns=cols)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def load_scored_table(conn) -> pd.DataFrame:
    print(f"  Loading {SCORED_TABLE} ...")
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {SCORED_TABLE}")
    rows = cur.fetchall()
    cols = [d[0].lower() for d in cur.description]
    cur.close()
    df = pd.DataFrame(rows, columns=cols)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def prepare_data(
    df: pd.DataFrame,
    numeric_feats: list[str],
    categorical_feats: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    all_feats = numeric_feats + categorical_feats
    X = df[all_feats].copy()

    # Coerce numeric columns (Snowflake may return Decimal objects)
    for col in numeric_feats:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Coerce categoricals to string (None → "missing")
    for col in categorical_feats:
        X[col] = X[col].fillna("missing").astype(str).str.strip().str.lower()
        X[col] = X[col].replace({"": "missing", "nan": "missing", "none": "missing"})

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return X, y

# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def build_preprocessor(numeric_feats: list[str], categorical_feats: list[str]) -> ColumnTransformer:
    numeric_pipe = SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = SKPipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_feats:
        transformers.append(("num", numeric_pipe, numeric_feats))
    if categorical_feats:
        transformers.append(("cat", categorical_pipe, categorical_feats))
    return ColumnTransformer(transformers=transformers, remainder="drop")

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(pipeline: SKPipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    return {
        "roc_auc":  round(roc_auc_score(y, y_prob), 4),
        "pr_auc":   round(average_precision_score(y, y_prob), 4),
        "f1":       round(f1_score(y, y_pred, zero_division=0), 4),
        "precision":round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":   round(recall_score(y, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
    }

# ── Per-tier training ──────────────────────────────────────────────────────────

def _build_classifier(tier: str, n_pos: int, n_neg: int):
    spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

    if tier == "baseline":
        return LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE,
            verbose=1,
        ), {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}

    if tier == "challenger":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
            verbose=10,
        ), {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}

    # champion — XGBoost (no early stopping so pipeline.fit() works cleanly)
    clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="auc",
        random_state=RANDOM_STATE, verbosity=1,
    )
    return clf, {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "scale_pos_weight": spw,
    }


def train_tier(
    tier: str,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
    n_pos: int, n_neg: int,
) -> ModelResult:
    clf, params = _build_classifier(tier, n_pos, n_neg)

    with mlflow.start_run(run_name=f"tier_{tier}", nested=True) as run:
        mlflow.set_tag("model_tier", tier)
        mlflow.log_params(params)

        # Build end-to-end sklearn pipeline
        pipe = SKPipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf),
        ])

        pipe.fit(X_train, y_train)

        metrics = evaluate(pipe, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log full pipeline as sklearn artifact (works for all 3 tiers)
        mlflow.sklearn.log_model(pipe, "model")

        # Feature importance for tree models
        if hasattr(clf, "feature_importances_"):
            try:
                feat_names = (
                    preprocessor.get_feature_names_out()
                    if hasattr(preprocessor, "get_feature_names_out") else []
                )
                importances = clf.feature_importances_
                if len(feat_names) == len(importances):
                    top_idx = np.argsort(importances)[::-1][:20]
                    fi = {feat_names[i]: round(float(importances[i]), 4) for i in top_idx}
                    mlflow.log_dict(fi, "feature_importance.json")
            except Exception:
                pass

        run_id = run.info.run_id
        print(f"    [{tier}] roc_auc={metrics['roc_auc']}  pr_auc={metrics['pr_auc']}  "
              f"f1={metrics['f1']}  recall={metrics['recall']}  run={run_id}")

    return ModelResult(tier=tier, model=pipe, metrics=metrics, mlflow_run_id=run_id, params=params)

# ── Helpers ────────────────────────────────────────────────────────────────────

def sep(title: str = "", w: int = 65) -> None:
    if title:
        print(f"\n── {title} {'─' * max(0, w - len(title) - 4)}")
    else:
        print("─" * w)

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Lead Intent model from Snowflake feature table.")
    parser.add_argument("--mode", default="raw", choices=["raw", "scored"],
                        help="raw=68 CRM features (FEATURES_LEAD_INTENT), scored=7 composite scores (LEAD_INTENT_SCORED)")
    parser.add_argument("--run-name",  default=None,
                        help="MLflow parent run name (default: auto based on mode)")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="Fraction of data for test set — only used in raw mode (default: 0.20)")
    parser.add_argument("--tiers", default="all", choices=["all", "champion"],
                        help="Which tiers to train (default: all)")
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = "scored_table_training" if args.mode == "scored" else "crm_feature_table_training"

    table_label = SCORED_TABLE if args.mode == "scored" else FEATURE_TABLE
    sep(f"Lead Intent — Training ({args.mode} mode)")
    print(f"  Source table  : {table_label}")
    print(f"  MLflow URI    : {os.environ['MLFLOW_TRACKING_URI']}")
    print(f"  Experiment    : {os.environ['MLFLOW_EXPERIMENT_NAME']}")
    print(f"  Run name      : {args.run_name}")

    # ── Connect & load ─────────────────────────────────────────────────────────
    sep("Connecting to Snowflake")
    t0 = time.time()
    try:
        conn = _connect()
    except Exception as e:
        print(f"\n[ERROR] Snowflake connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "scored":
        df = load_scored_table(conn)
        conn.close()
        _run_scored(df, args)
    else:
        df = load_feature_table(conn)
        conn.close()
        _run_raw(df, args)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed : {elapsed:.0f}s")


def _run_scored(df: pd.DataFrame, args) -> None:
    """Train on the 7 composite scores from LEAD_INTENT_SCORED."""
    available = [f for f in SCORED_FEATURES if f in df.columns]
    missing = [f for f in SCORED_FEATURES if f not in df.columns]
    if missing:
        print(f"  [WARN] Missing score columns: {missing}")

    # Use pre-defined split column
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "validation"]
    test_df  = df[df["split"] == "test"]

    X_train = train_df[available].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df[SCORED_TARGET], errors="coerce").fillna(0).astype(int)
    X_val   = val_df[available].apply(pd.to_numeric, errors="coerce")
    y_val   = pd.to_numeric(val_df[SCORED_TARGET], errors="coerce").fillna(0).astype(int)
    X_test  = test_df[available].apply(pd.to_numeric, errors="coerce")
    y_test  = pd.to_numeric(test_df[SCORED_TARGET], errors="coerce").fillna(0).astype(int)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    pos_rate = n_pos / len(y_train) if len(y_train) else 0

    sep("Dataset (scored mode — 7 composite scores)")
    print(f"  Features      : {available}")
    print(f"  Target        : {SCORED_TARGET} (converted = enrolled)")
    print(f"  Train rows    : {len(X_train):>10,}  pos={int(y_train.sum()):,} ({y_train.mean():.2%})")
    print(f"  Val rows      : {len(X_val):>10,}  pos={int(y_val.sum()):,} ({y_val.mean():.2%})")
    print(f"  Test rows     : {len(X_test):>10,}  pos={int(y_test.sum()):,} ({y_test.mean():.2%})")
    print(f"  scale_pos_wt  : {n_neg/n_pos:.2f}")

    # Simple preprocessor — all features are already 0-100 numeric
    preprocessor = build_preprocessor(available, [])
    preprocessor.fit(X_train)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    tier_names = ["baseline", "challenger", "champion"] if args.tiers == "all" else ["champion"]
    results: list[ModelResult] = []

    sep("Training tiers")
    with mlflow.start_run(run_name=args.run_name) as parent_run:
        mlflow.set_tag("pipeline_type", "scored_table_training")
        mlflow.set_tag("feature_table", SCORED_TABLE)
        mlflow.log_params({
            "feature_table":   SCORED_TABLE,
            "n_features":      len(available),
            "n_train":         len(X_train),
            "n_val":           len(X_val),
            "n_test":          len(X_test),
            "positive_rate":   round(pos_rate, 4),
            "scale_pos_weight": round(n_neg / n_pos, 2) if n_pos else 1.0,
        })

        for tier in tier_names:
            print(f"\n  Training {tier} ...")
            # Evaluate on val for scored mode (separate held-out test set)
            result = train_tier(tier, preprocessor, X_train, y_train, X_val, y_val, n_pos, n_neg)
            results.append(result)

        selector = ModelSelector()
        winner = selector.select(results)

        # Final evaluation on test set (unseen during training + selection)
        sep("Final test-set evaluation")
        test_metrics = evaluate(winner.model, X_test, y_test)
        print(f"  [test] roc_auc={test_metrics['roc_auc']}  pr_auc={test_metrics['pr_auc']}  "
              f"f1={test_metrics['f1']}  recall={test_metrics['recall']}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.set_tag("winning_tier", winner.tier)
        mlflow.log_metric("winning_roc_auc", winner.metrics.get("roc_auc", 0))
        mlflow.log_metric("winning_pr_auc",  winner.metrics.get("pr_auc", 0))
        mlflow.log_metric("winning_f1",      winner.metrics.get("f1", 0))
        mlflow.log_metric("winning_recall",  winner.metrics.get("recall", 0))
        parent_run_id = parent_run.info.run_id

    _register_winner(winner, parent_run_id)


def _run_raw(df: pd.DataFrame, args) -> None:
    """Train on the 68 raw CRM features from FEATURES_LEAD_INTENT."""
    available_numeric = [f for f in NUMERIC_FEATURES if f in df.columns]
    available_categorical = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    missing = (
        [f for f in NUMERIC_FEATURES if f not in df.columns]
        + [f for f in CATEGORICAL_FEATURES if f not in df.columns]
    )
    if missing:
        print(f"  [WARN] {len(missing)} expected features not in table (skipped): {missing[:8]}{'...' if len(missing)>8 else ''}")

    sep("Dataset (raw mode — 68 CRM features)")
    X, y = prepare_data(df, available_numeric, available_categorical)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    pos_rate = n_pos / len(y) if len(y) else 0
    print(f"  Total rows       : {len(y):>12,}")
    print(f"  Positive class   : {n_pos:>12,}  ({pos_rate:.2%})")
    print(f"  Negative class   : {n_neg:>12,}")
    print(f"  scale_pos_weight : {n_neg/n_pos:.2f}")
    print(f"  Numeric features : {len(available_numeric)}")
    print(f"  Categorical feats: {len(available_categorical)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"\n  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    sep("Fitting preprocessor")
    preprocessor = build_preprocessor(available_numeric, available_categorical)
    preprocessor.fit(X_train)
    n_out_feats = preprocessor.transform(X_train[:1]).shape[1]
    print(f"  Output feature dim after encoding: {n_out_feats}")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    tier_names = ["baseline", "challenger", "champion"] if args.tiers == "all" else ["champion"]
    results: list[ModelResult] = []

    sep("Training tiers")
    with mlflow.start_run(run_name=args.run_name) as parent_run:
        mlflow.set_tag("pipeline_type", "crm_feature_table_training")
        mlflow.set_tag("feature_table", FEATURE_TABLE)
        mlflow.log_params({
            "feature_table":       FEATURE_TABLE,
            "n_train":             len(X_train),
            "n_test":              len(X_test),
            "n_input_features":    len(available_numeric) + len(available_categorical),
            "n_encoded_features":  n_out_feats,
            "positive_rate":       round(pos_rate, 4),
            "scale_pos_weight":    round(n_neg / n_pos, 2) if n_pos else 1.0,
            "test_size":           args.test_size,
            "random_state":        RANDOM_STATE,
        })

        for tier in tier_names:
            print(f"\n  Training {tier} ...")
            result = train_tier(tier, preprocessor, X_train, y_train, X_test, y_test, n_pos, n_neg)
            results.append(result)

        selector = ModelSelector()
        winner = selector.select(results)

        mlflow.set_tag("winning_tier", winner.tier)
        mlflow.log_metric("winning_roc_auc", winner.metrics.get("roc_auc", 0))
        mlflow.log_metric("winning_pr_auc",  winner.metrics.get("pr_auc", 0))
        mlflow.log_metric("winning_f1",      winner.metrics.get("f1", 0))
        mlflow.log_metric("winning_recall",  winner.metrics.get("recall", 0))
        parent_run_id = parent_run.info.run_id

    _register_winner(winner, parent_run_id)


def _register_winner(winner: ModelResult, parent_run_id: str) -> None:
    sep("Model selection & registration")
    print(f"\n  Winner        : {winner.tier}")
    print(f"  roc_auc       : {winner.metrics.get('roc_auc')}")
    print(f"  pr_auc        : {winner.metrics.get('pr_auc')}")
    print(f"  f1            : {winner.metrics.get('f1')}")
    print(f"  recall        : {winner.metrics.get('recall')}")
    print(f"  Parent run ID : {parent_run_id}")

    model_name = os.environ.get("MODEL_NAME", "lead_intent_model")
    client = mlflow.MlflowClient()
    registered = mlflow.register_model(f"runs:/{winner.mlflow_run_id}/model", model_name)
    version = registered.version

    for v in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(model_name, v.version, "Archived")
        print(f"  Archived previous Production v{v.version}")

    client.transition_model_version_stage(model_name, version, "Production")
    sep()
    print(f"[DONE] Model v{version} ({winner.tier}) → Production")
    print(f"  MLflow UI : {os.environ['MLFLOW_TRACKING_URI']}")
    print(f"  Model     : {model_name}  version={version}")
    print()
    print("  Next steps:")
    print("    Start API : uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("    Dashboard : http://localhost:8000/ui")


if __name__ == "__main__":
    main()
