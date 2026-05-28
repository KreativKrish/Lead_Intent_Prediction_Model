"""Derive 7 composite scores from FEATURES_LEAD_INTENT and write LEAD_INTENT_SCORED."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from snowflake.connector.pandas_tools import write_pandas


class ScoredFeaturePipeline:
    SOURCE_TABLE = "FEATURES_LEAD_INTENT"
    TARGET_TABLE = "LEAD_INTENT_SCORED"

    def __init__(self, conn, ml_db: str, ml_schema: str):
        self.conn = conn
        self.ml_db = ml_db
        self.ml_schema = ml_schema

    def _load(self) -> pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT * FROM {self.ml_db}.{self.ml_schema}.{self.SOURCE_TABLE}"
        )
        rows = cur.fetchall()
        cols = [d[0].lower() for d in cur.description]
        cur.close()
        return pd.DataFrame(rows, columns=cols)

    def _p95(self, series: pd.Series) -> float:
        v = float(np.percentile(series.dropna(), 95))
        return v if v > 0 else 1.0

    def _score_eligibility(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["is_employed_professional"] * 25
            + df["has_ctc"] * 20
            + df["has_experience"] * 15
            + df["has_salary_increment_goal"] * 15
            + df["has_qualification"] * 15
            + df["is_domestic"] * 10
        ).astype(float)

    def _score_demographic(self, df: pd.DataFrame) -> pd.Series:
        return (
            (df["profile_completeness_score"] / 12) * 60
            + df["has_real_email"] * 15
            + df["has_alternate_mobile"] * 10
            + df["has_best_time_to_call"] * 10
            + df["has_pain_points"] * 5
        ).astype(float)

    def _score_quality(self, df: pd.DataFrame, p95_channel_cvr: float) -> pd.Series:
        channel_cvr_norm = df["channel_historical_cvr"].clip(upper=None).div(p95_channel_cvr).clip(0, 1)
        return (
            (1 - df["is_duplicate"]) * 30
            + df["has_real_email"] * 20
            + (1 - df["is_chatbot"]) * 15
            + channel_cvr_norm * 25
            + df["is_repeat_contact"] * 10
        ).astype(float)

    def _score_engagement(self, df: pd.DataFrame, p95_engagement: float) -> pd.Series:
        eng_norm = df["engagement_intensity_score"].div(p95_engagement).clip(0, 1)
        return (
            eng_norm * 40
            + df["call_answer_rate"] * 25
            + df["has_multi_channel_engagement"] * 20
            + df["has_positive_intent"] * 15
        ).astype(float)

    def _score_intent(self, df: pd.DataFrame) -> pd.Series:
        net_norm = (df["net_intent_score"].clip(-10, 15) + 10).div(25).clip(0, 1)
        return (
            net_norm * 35
            + df["contacted_within_1hr"] * 25
            + df["has_positive_intent"] * 20
            + (1 - df["has_negative_intent"]) * 20
        ).astype(float)

    def _score_campaign(
        self, df: pd.DataFrame, p95_channel_cvr: float, p95_campaign_cvr: float
    ) -> pd.Series:
        ch_cvr_norm   = df["channel_historical_cvr"].div(p95_channel_cvr).clip(0, 1)
        camp_cvr_norm = df["campaign_historical_cvr"].div(p95_campaign_cvr).clip(0, 1)
        return (
            ch_cvr_norm * 50
            + camp_cvr_norm * 40
            + df["has_campaign"] * 10
        ).astype(float)

    def _score_lead_aging(self, df: pd.DataFrame) -> pd.Series:
        return (100 * np.exp(-df["lead_age_days"] / 30)).round(2)

    def build(self) -> dict:
        df = self._load()

        # compute p95 normalization params
        p95_channel_cvr  = self._p95(df["channel_historical_cvr"])
        p95_campaign_cvr = self._p95(df["campaign_historical_cvr"])
        p95_engagement   = self._p95(df["engagement_intensity_score"])

        out = pd.DataFrame()
        out["lead_id"]            = df["lead_id"]
        out["eligibility_score"]  = self._score_eligibility(df)
        out["demographic_score"]  = self._score_demographic(df)
        out["quality_score"]      = self._score_quality(df, p95_channel_cvr)
        out["engagement_score"]   = self._score_engagement(df, p95_engagement)
        out["intent_score"]       = self._score_intent(df)
        out["campaign_score"]     = self._score_campaign(df, p95_channel_cvr, p95_campaign_cvr)
        out["lead_aging"]         = self._score_lead_aging(df)
        out["converted"]          = df["is_interested"].astype(int)

        # stratified split
        idx_train, idx_temp = train_test_split(
            out.index, test_size=0.30, stratify=out["converted"], random_state=42
        )
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=1 / 3, stratify=out.loc[idx_temp, "converted"], random_state=42
        )
        out["split"] = "train"
        out.loc[idx_val, "split"]  = "validation"
        out.loc[idx_test, "split"] = "test"
        out["scored_at"] = pd.Timestamp.now()

        write_pandas(
            self.conn,
            out,
            table_name=self.TARGET_TABLE,
            database=self.ml_db,
            schema=self.ml_schema,
            quote_identifiers=False,
            overwrite=True,
            auto_create_table=True,
        )

        score_cols = [
            "eligibility_score", "demographic_score", "quality_score",
            "engagement_score", "intent_score", "campaign_score", "lead_aging",
        ]
        return {
            "total_rows":  len(out),
            "train_rows":  int((out["split"] == "train").sum()),
            "val_rows":    int((out["split"] == "validation").sum()),
            "test_rows":   int((out["split"] == "test").sum()),
            "positive_rate":  float(out["converted"].mean()),
            "positive_class": int(out["converted"].sum()),
            "score_means": {c: round(float(out[c].mean()), 4) for c in score_cols},
            "normalization_params": {
                "p95_channel_cvr":  round(p95_channel_cvr, 6),
                "p95_campaign_cvr": round(p95_campaign_cvr, 6),
                "p95_engagement":   round(p95_engagement, 4),
            },
        }
