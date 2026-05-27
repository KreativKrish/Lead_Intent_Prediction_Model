"""CRM feature engineering pipeline.

Reads from PROD_DATABASE.CRM (read-only) via pandas, computes derived
features, and writes FEATURES_LEAD_INTENT to MARKETING_DATABASE.LEAD_INTENT_ML.

Why pandas instead of pure SQL CTEs:
    All columns in this Snowflake account were created with quoted lowercase
    identifiers.  Referencing them in SQL WHERE / GROUP BY without quotes
    resolves to uppercase and fails.  Using SELECT * into pandas sidesteps
    that entirely; aggregation-only SQL queries quote the few columns they need.

Feature groups:
    1.  Source quality           - channel/medium historical CVR
    2.  Campaign performance     - campaign-level CVR + volume
    3.  Follow-up velocity       - speed and cadence of follow-ups
    4.  Lead freshness           - age and recency of last update
    5.  Owner performance        - agent CVR, workload, tenure
    6.  Engagement intensity     - weighted multi-channel score
    7.  NLP intent score         - positive/negative keyword counts
    8.  Time-to-response         - minutes to first contact
    9.  Funnel progression speed - status transitions and time-to-interested
    10. Affordability indicators - employment, CTC, experience signals
    +   Profile completeness     - data quality score (0-12)
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

# ── SQL for aggregated tables (quoting needed only in GROUP BY / WHERE) ────────

_ACTIVITY_AGG_SQL = """
SELECT
    "lead_id"                                                               AS lead_id,
    COUNT(*)                                                                AS total_activities,
    COUNT(DISTINCT "activity_id")                                           AS unique_activity_types,
    COUNT(CASE WHEN "call_status" IS NOT NULL THEN 1 END)                  AS total_calls_logged,
    COUNT(CASE WHEN LOWER("call_status") IN ('answered','connected') THEN 1 END) AS answered_calls,
    COALESCE(SUM("duration"), 0)                                            AS total_call_duration_sec,
    AVG(CASE WHEN COALESCE("duration", 0) > 0 THEN "duration" END)        AS avg_call_duration_sec,
    MAX("duration")                                                         AS max_call_duration_sec,
    COUNT(CASE WHEN "call_recordings" IS NOT NULL THEN 1 END)              AS calls_with_recording,
    COUNT(DISTINCT "follow_up_id")                                          AS followup_count,
    MIN(CASE WHEN "follow_up_id" IS NOT NULL THEN "created_at" END)        AS first_followup_at,
    MAX(CASE WHEN "follow_up_id" IS NOT NULL THEN "created_at" END)        AS last_followup_at,
    MIN("created_at")                                                       AS first_activity_at,
    MAX("created_at")                                                       AS last_activity_at,
    SUM(CASE WHEN
            LOWER(COALESCE("activity_details",'')) LIKE '%interested%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%want to enroll%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%fee%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%admission%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%will join%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%joining%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%callback%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%ready to%'
         OR LOWER(COALESCE("description",''))      LIKE '%interested%'
         OR LOWER(COALESCE("description",''))      LIKE '%want to enroll%'
         OR LOWER(COALESCE("description",''))      LIKE '%fee%'
        THEN 1 ELSE 0 END)                                                  AS positive_intent_kw_count,
    SUM(CASE WHEN
            LOWER(COALESCE("activity_details",'')) LIKE '%not interested%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%wrong number%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%not responding%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%do not call%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%not received%'
         OR LOWER(COALESCE("activity_details",'')) LIKE '%switched off%'
         OR LOWER(COALESCE("description",''))      LIKE '%not interested%'
         OR LOWER(COALESCE("description",''))      LIKE '%not responding%'
        THEN 1 ELSE 0 END)                                                  AS negative_intent_kw_count
FROM {src_db}.{src_schema}.LEAD_ACTIVITIES
GROUP BY "lead_id"
"""

_FB_AGG_SQL = """
SELECT
    "lead_id"                                         AS lead_id,
    COUNT(*)                                          AS fb_events_count,
    SUM(CASE WHEN "success" THEN 1 ELSE 0 END)       AS fb_events_success,
    ROUND(AVG(COALESCE("retry_count", 0)), 2)         AS avg_fb_retry_count
FROM {src_db}.{src_schema}.FACEBOOK_CAPI_EVENTS
GROUP BY "lead_id"
"""

_VB_AGG_SQL = """
SELECT
    TRY_CAST(GET_PATH(TRY_PARSE_JSON("request"), 'lead_id')::STRING AS NUMBER) AS lead_id,
    COUNT(*)                                                                     AS voicebot_call_count,
    SUM(CASE WHEN "status_code" = '202' THEN 1 ELSE 0 END)                     AS voicebot_initiated_count
FROM {src_db}.{src_schema}.VOICE_BOT_API_LOGS
WHERE "action_type" = 'init_call_nurix_bot'
  AND TRY_CAST(GET_PATH(TRY_PARSE_JSON("request"), 'lead_id')::STRING AS NUMBER) IS NOT NULL
GROUP BY 1
"""

# ── helpers ────────────────────────────────────────────────────────────────────

def _days(ts_series: pd.Series, reference: date | None = None) -> pd.Series:
    """Days from a timestamp series to today (or reference date)."""
    ref = pd.Timestamp(reference or date.today())
    return (ref - pd.to_datetime(ts_series, errors="coerce").dt.normalize()).dt.days


def _hours_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """Hours from Series a to Series b."""
    return (
        pd.to_datetime(b, errors="coerce") - pd.to_datetime(a, errors="coerce")
    ).dt.total_seconds() / 3600


def _minutes_between(a: pd.Series, b: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(b, errors="coerce") - pd.to_datetime(a, errors="coerce")
    ).dt.total_seconds() / 60


def _flag(series: pd.Series) -> pd.Series:
    """Return 1 where series is non-null and non-empty-string, else 0."""
    return (series.notna() & (series.astype(str).str.strip() != "")).astype("int8")


# ── main pipeline class ────────────────────────────────────────────────────────

class CRMFeaturePipeline:
    """Builds FEATURES_LEAD_INTENT in the ML database."""

    def __init__(self, conn, src_db: str, src_schema: str, ml_db: str, ml_schema: str):
        self.conn = conn
        self.src_db = src_db
        self.src_schema = src_schema
        self.ml_db = ml_db
        self.ml_schema = ml_schema

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def _sql(self, template: str) -> str:
        return template.format(src_db=self.src_db, src_schema=self.src_schema,
                               ml_db=self.ml_db, ml_schema=self.ml_schema)

    def _read_sql(self, sql: str) -> pd.DataFrame:
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0].lower() for d in cur.description]
        return pd.DataFrame(rows, columns=cols)

    def _read_table(self, table: str) -> pd.DataFrame:
        return self._read_sql(
            f"SELECT * FROM {self.src_db}.{self.src_schema}.{table}"
        )

    def _write_to_snowflake(self, df: pd.DataFrame, table_name: str) -> None:
        from snowflake.connector.pandas_tools import write_pandas

        cur = self.conn.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.ml_db}.{self.ml_schema}")
        cur.execute(
            f"DROP TABLE IF EXISTS {self.ml_db}.{self.ml_schema}.{table_name}"
        )

        success, _, nrows, _ = write_pandas(
            self.conn, df,
            table_name=table_name,
            database=self.ml_db,
            schema=self.ml_schema,
            auto_create_table=True,
            overwrite=True,
            quote_identifiers=False,
        )
        if not success:
            raise RuntimeError(f"write_pandas failed for {table_name}")
        print(f"  Wrote {nrows:,} rows to {self.ml_db}.{self.ml_schema}.{table_name}")

    # ── feature computation ───────────────────────────────────────────────────

    def _build_features(
        self, leads: pd.DataFrame, act: pd.DataFrame, contacts: pd.DataFrame,
        fb: pd.DataFrame, vb: pd.DataFrame, users: pd.DataFrame,
        status_map: pd.DataFrame,
    ) -> pd.DataFrame:
        today = date.today()

        # --- normalise column names to lowercase ---
        for df in [leads, act, contacts, fb, vb, users, status_map]:
            df.columns = [c.lower() for c in df.columns]

        # --- filter soft-deleted leads ---
        if "deleted_at" in leads.columns:
            leads = leads[leads["deleted_at"].isna()].copy()

        # --- status name lookup ---
        status_lookup = status_map.set_index("id")[["name"]].rename(columns={"name": "status_name"})

        # --- contact lookup ---
        contacts_idx = contacts.set_index("id")

        # --- owner lookup ---
        users_idx = users.set_index("id")

        # --- source quality (channel historical CVR) ---
        ch_cvr = (
            leads.groupby("lead_channel", dropna=False)
            .agg(channel_cvr=("is_interested", "mean"),
                 channel_volume=("id", "count"))
            .reset_index()
        )

        # --- campaign performance ---
        camp_mask = leads["campaign_name"].notna() & (leads["campaign_name"].str.strip() != "")
        camp_cvr = (
            leads[camp_mask]
            .groupby("campaign_name", dropna=False)
            .agg(campaign_cvr=("is_interested", "mean"),
                 campaign_volume=("id", "count"))
            .reset_index()
        )

        # --- owner performance ---
        owner_perf = (
            leads.groupby("lead_owner", dropna=False)
            .agg(
                owner_cvr_overall=("is_interested", "mean"),
                owner_total_leads=("id", "count"),
                owner_tenure_days=("created_at", lambda s: (
                    pd.Timestamp(today) - pd.to_datetime(s, errors="coerce").min()
                ).days if not s.isna().all() else np.nan),
            )
            .reset_index()
        )
        # owner active leads (last 30d) - separate pass
        cutoff = pd.Timestamp(today) - pd.Timedelta(days=30)
        leads["_upd"] = pd.to_datetime(leads["updated_at"], errors="coerce")
        active_30 = (
            leads[leads["_upd"] >= cutoff]
            .groupby("lead_owner", dropna=False)["id"]
            .count()
            .rename("owner_active_leads_30d")
        )
        owner_perf = owner_perf.merge(active_30, on="lead_owner", how="left")
        owner_perf["owner_active_leads_30d"] = owner_perf["owner_active_leads_30d"].fillna(0)

        # --- merge source quality ---
        leads = leads.merge(ch_cvr, on="lead_channel", how="left")
        leads = leads.merge(camp_cvr, on="campaign_name", how="left")
        leads = leads.merge(owner_perf, on="lead_owner", how="left")

        # --- merge activity aggregates ---
        leads = leads.merge(act, left_on="id", right_on="lead_id", how="left", suffixes=("", "_act"))

        # --- merge contacts ---
        leads["contact_lead_count"] = leads["contact_id"].map(
            contacts_idx["lead_count"] if "lead_count" in contacts_idx.columns else pd.Series(dtype=float)
        ).fillna(0)
        leads["is_domestic"] = leads["contact_id"].map(
            contacts_idx["is_domestic"] if "is_domestic" in contacts_idx.columns else pd.Series(dtype=float)
        ).fillna(1)

        # --- merge facebook ---
        leads = leads.merge(fb, left_on="id", right_on="lead_id", how="left", suffixes=("", "_fb"))

        # --- merge voicebot ---
        leads = leads.merge(vb, left_on="id", right_on="lead_id", how="left", suffixes=("", "_vb"))

        # --- merge owner active status ---
        if "status" in users_idx.columns and "deleted_at" in users_idx.columns:
            users_idx["is_owner_active"] = (
                (users_idx["deleted_at"].isna()) & (users_idx["status"] == 1)
            ).astype("int8")
        elif "status" in users_idx.columns:
            users_idx["is_owner_active"] = (users_idx["status"] == 1).astype("int8")
        else:
            users_idx["is_owner_active"] = 0
        leads["is_owner_active"] = leads["lead_owner"].map(users_idx["is_owner_active"]).fillna(0).astype("int8")
        leads["owner_role_id"] = leads["lead_owner"].map(
            users_idx["role_id"] if "role_id" in users_idx.columns else pd.Series(dtype=float)
        )

        # --- status name mapping ---
        def map_status(col: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(col, errors="coerce")
            return numeric.map(status_lookup["status_name"])

        # ── now build the final feature DataFrame ──────────────────────────────
        f = pd.DataFrame()
        ld = leads  # shorthand

        # identifiers + target
        f["lead_id"]    = ld["id"]
        f["contact_id"] = ld["contact_id"]
        f["is_interested"] = ld["is_interested"].fillna(0).astype("int8")

        # 1. Source quality
        f["lead_channel"]         = ld["lead_channel"]
        f["source_medium"]        = ld["source_medium"]
        f["channel_historical_cvr"] = ld["channel_cvr"].fillna(0)
        f["channel_volume"]         = ld["channel_volume"].fillna(0)

        # 2. Campaign performance
        f["has_campaign"]           = _flag(ld["campaign_name"])
        f["campaign_historical_cvr"]= ld["campaign_cvr"].fillna(0)
        f["campaign_volume"]        = ld["campaign_volume"].fillna(0)

        # 3. Follow-up velocity
        f["followup_count"]           = ld["followup_count"].fillna(0).astype(int)
        f["hours_to_first_followup"]  = _hours_between(ld["lead_date"], ld["first_followup_at"])
        interval_hrs = _hours_between(ld["first_followup_at"], ld["last_followup_at"])
        denom = (ld["followup_count"].fillna(0) - 1).clip(lower=1)
        f["avg_followup_interval_hours"] = np.where(
            ld["followup_count"].fillna(0) > 1, interval_hrs / denom, np.nan
        )

        # 4. Lead freshness
        f["lead_age_days"]          = _days(ld["lead_date"])
        f["days_since_last_update"] = _days(ld["updated_at"])
        f["is_fresh_7d"]            = (f["lead_age_days"] <= 7).astype("int8")
        f["is_fresh_30d"]           = (f["lead_age_days"] <= 30).astype("int8")
        lead_dt = pd.to_datetime(ld["lead_date"], errors="coerce")
        f["lead_hour_of_day"]       = lead_dt.dt.hour
        f["lead_day_of_week"]       = lead_dt.dt.dayofweek
        f["lead_month"]             = lead_dt.dt.month
        f["is_weekend_lead"]        = lead_dt.dt.dayofweek.isin([5, 6]).astype("int8")
        f["is_business_hours_lead"] = lead_dt.dt.hour.between(9, 18).astype("int8")

        # 5. Owner performance
        f["is_owner_active"]      = ld["is_owner_active"].fillna(0).astype("int8")
        f["owner_role_id"]        = ld["owner_role_id"]
        f["owner_historical_cvr"] = ld["owner_cvr_overall"].fillna(0)
        f["owner_workload_30d"]   = ld["owner_active_leads_30d"].fillna(0)
        f["owner_tenure_days"]    = ld["owner_tenure_days"].fillna(0)
        f["was_reassigned"]       = (
            ld["previous_lead_owner"].notna()
            & (ld["previous_lead_owner"] != ld["lead_owner"])
        ).astype("int8")

        # 6. Engagement intensity
        f["call_count"]      = ld["call_count"].fillna(0)
        f["sms_count"]       = ld["sms_count"].fillna(0)
        f["email_count"]     = ld["email_count"].fillna(0)
        f["whatsapp_count"]  = ld["whatsapp_count"].fillna(0)
        f["engagement_intensity_score"] = (
            f["call_count"] * 3 + f["whatsapp_count"] * 2
            + f["email_count"] + f["sms_count"]
        )
        f["has_multi_channel_engagement"] = (
            (f["call_count"] > 0).astype(int)
            + (f["sms_count"] > 0).astype(int)
            + (f["email_count"] > 0).astype(int)
            + (f["whatsapp_count"] > 0).astype(int)
            >= 2
        ).astype("int8")
        f["total_activities"]        = ld["total_activities"].fillna(0)
        f["total_calls_logged"]      = ld["total_calls_logged"].fillna(0)
        f["answered_calls"]          = ld["answered_calls"].fillna(0)
        f["total_call_duration_sec"] = ld["total_call_duration_sec"].fillna(0)
        f["avg_call_duration_sec"]   = ld["avg_call_duration_sec"].fillna(0)
        f["max_call_duration_sec"]   = ld["max_call_duration_sec"].fillna(0)
        f["call_answer_rate"]        = np.where(
            f["total_calls_logged"] > 0,
            f["answered_calls"] / f["total_calls_logged"], 0
        ).round(3)
        f["calls_with_recording"]    = ld["calls_with_recording"].fillna(0)
        f["unique_activity_types"]   = ld["unique_activity_types"].fillna(0)
        f["days_since_last_activity"]= _days(ld["last_activity_at"])

        # 7. NLP intent score
        f["positive_intent_kw_count"] = ld["positive_intent_kw_count"].fillna(0)
        f["negative_intent_kw_count"] = ld["negative_intent_kw_count"].fillna(0)
        f["net_intent_score"]         = f["positive_intent_kw_count"] - f["negative_intent_kw_count"]
        f["has_positive_intent"]      = (f["positive_intent_kw_count"] > 0).astype("int8")
        f["has_negative_intent"]      = (f["negative_intent_kw_count"] > 0).astype("int8")

        # 8. Time-to-response
        f["minutes_to_first_contact"] = _minutes_between(ld["lead_date"], ld["first_activity_at"])
        f["contacted_within_1hr"]     = (
            f["minutes_to_first_contact"].notna() & (f["minutes_to_first_contact"] <= 60)
        ).astype("int8")
        f["contacted_within_24hrs"]   = (
            f["minutes_to_first_contact"].notna() & (f["minutes_to_first_contact"] <= 1440)
        ).astype("int8")
        f["never_contacted"]          = ld["first_activity_at"].isna().astype("int8")

        # 9. Funnel progression speed
        f["lead_status_id"]          = ld["lead_status"]
        f["lead_status_name"]        = map_status(ld["lead_status"])
        f["lead_sub_status_id"]      = ld["lead_sub_status"]
        f["lead_sub_status_name"]    = map_status(ld["lead_sub_status"])
        f["previous_lead_status_id"] = ld["previous_lead_status"]
        f["previous_lead_status_name"] = map_status(ld["previous_lead_status"])
        f["status_changed"]          = (
            ld["previous_lead_status"].notna()
            & (ld["previous_lead_status"] != ld["lead_status"])
        ).astype("int8")
        f["reached_interested_stage"]  = ld["interested_date_time"].notna().astype("int8")
        f["hours_to_interested_stage"] = _hours_between(ld["lead_date"], ld["interested_date_time"])
        f["days_in_current_status"]    = _days(ld["updated_at"])

        # 10. Affordability indicators
        f["has_ctc"]                  = _flag(ld["ctc_annual_package"])
        f["has_experience"]           = _flag(ld["experience"])
        f["has_company"]              = _flag(ld["company_name"])
        f["has_designation"]          = _flag(ld["designation"])
        f["is_employed_professional"] = (
            _flag(ld["company_name"]) & _flag(ld["designation"])
        ).astype("int8")
        f["has_salary_increment_goal"]= _flag(ld["salary_increment"])

        # Profile completeness (0-12 score)
        f["has_real_email"]        = (
            ld["email"].notna() & ~ld["email"].astype(str).str.contains("default@", na=False)
        ).astype("int8")
        f["has_gender"]            = _flag(ld["gender"])
        f["has_dob"]               = _flag(ld["dob"])
        f["has_state"]             = _flag(ld["state"])
        f["has_city"]              = _flag(ld["city"])
        f["has_alternate_mobile"]  = _flag(ld["alternate_mobile_number"])
        f["has_best_time_to_call"] = _flag(ld["best_time_to_call"])
        f["has_qualification"]     = _flag(ld["qualification"])
        f["has_pain_points"]       = _flag(ld["pain_points"])
        f["profile_completeness_score"] = (
            f["has_real_email"] + f["has_gender"] + f["has_dob"] + f["has_state"]
            + f["has_city"] + f["has_alternate_mobile"] + f["has_designation"]
            + f["has_company"] + f["has_ctc"] + f["has_experience"]
            + f["has_qualification"] + f["has_pain_points"]
        )

        # Misc flags
        f["is_chatbot"]   = ld["is_chatbot"].fillna(0).astype("int8")
        f["is_voicebot"]  = ld["is_voicebot"].fillna(0).astype("int8")
        f["is_duplicate"] = ld["duplicate_check"].fillna(0).astype("int8")
        f["lead_type"]    = ld["lead_type"]
        f["course_id"]    = ld["course"]
        f["university_id"]= ld["university_interested"]

        # Contact
        f["contact_lead_count"] = ld["contact_lead_count"].fillna(0)
        f["is_domestic"]        = ld["is_domestic"].fillna(1)
        f["is_repeat_contact"]  = (f["contact_lead_count"] > 1).astype("int8")

        # Facebook
        f["is_facebook_lead"]    = ld["fb_events_count"].notna().astype("int8")
        f["fb_events_count"]     = ld["fb_events_count"].fillna(0)
        f["fb_events_success"]   = ld["fb_events_success"].fillna(0)
        f["fb_event_success_rate"]= np.where(
            f["fb_events_count"] > 0,
            f["fb_events_success"] / f["fb_events_count"], 0
        ).round(3)

        # Voice bot
        f["has_voicebot_interaction"]  = ld["voicebot_call_count"].notna().astype("int8")
        f["voicebot_call_count"]       = ld["voicebot_call_count"].fillna(0)
        f["voicebot_initiated_count"]  = ld["voicebot_initiated_count"].fillna(0)

        # Timestamps
        f["lead_date"]           = pd.to_datetime(ld["lead_date"], errors="coerce")
        f["created_at"]          = pd.to_datetime(ld["created_at"], errors="coerce")
        f["updated_at"]          = pd.to_datetime(ld["updated_at"], errors="coerce")
        f["feature_computed_at"] = pd.Timestamp.now()

        return f

    # ── public entry point ────────────────────────────────────────────────────

    def build(self) -> dict:
        print("  Loading LEAD_MASTERS (1.3 M rows — ~2 min) ...")
        leads = self._read_table("LEAD_MASTERS")

        print("  Aggregating LEAD_ACTIVITIES in Snowflake ...")
        act = self._read_sql(self._sql(_ACTIVITY_AGG_SQL))

        print("  Loading CONTACT_MASTER ...")
        contacts = self._read_table("CONTACT_MASTER")

        print("  Aggregating FACEBOOK_CAPI_EVENTS ...")
        fb = self._read_sql(self._sql(_FB_AGG_SQL))

        print("  Aggregating VOICE_BOT_API_LOGS ...")
        vb = self._read_sql(self._sql(_VB_AGG_SQL))

        print("  Loading USERS ...")
        users = self._read_table("USERS")

        print("  Loading LEAD_STATUS_MASTERS ...")
        status_map = self._read_table("LEAD_STATUS_MASTERS")

        print("  Computing features ...")
        features = self._build_features(leads, act, contacts, fb, vb, users, status_map)

        print("  Writing to Snowflake ...")
        self._write_to_snowflake(features, "FEATURES_LEAD_INTENT")

        # stats
        n = len(features)
        pos = int(features["is_interested"].sum())
        stats = {
            "TOTAL_ROWS": n,
            "POSITIVE_CLASS": pos,
            "POSITIVE_RATE": round(pos / n, 4) if n else 0,
            "NEVER_CONTACTED_COUNT": int(features["never_contacted"].sum()),
            "FACEBOOK_LEAD_COUNT": int(features["is_facebook_lead"].sum()),
            "VOICEBOT_LEAD_COUNT": int(features["has_voicebot_interaction"].sum()),
            "NULL_RATE_FIRST_CONTACT": round(features["minutes_to_first_contact"].isna().mean(), 3),
            "NULL_RATE_FIRST_FOLLOWUP": round(features["hours_to_first_followup"].isna().mean(), 3),
            "NULL_RATE_INTERESTED_STAGE": round(features["hours_to_interested_stage"].isna().mean(), 3),
            "NULL_RATE_CHANNEL_CVR": round((features["channel_historical_cvr"] == 0).mean(), 3),
            "NULL_RATE_OWNER_CVR": round((features["owner_historical_cvr"] == 0).mean(), 3),
            "AVG_LEAD_AGE_DAYS": round(features["lead_age_days"].mean(), 1),
            "AVG_ENGAGEMENT_SCORE": round(features["engagement_intensity_score"].mean(), 2),
            "AVG_PROFILE_SCORE": round(features["profile_completeness_score"].mean(), 2),
            "AVG_NET_INTENT_SCORE": round(features["net_intent_score"].mean(), 3),
            "AVG_TOTAL_CALL_DURATION_SEC": round(features["total_call_duration_sec"].mean(), 0),
        }

        # correlations (numeric only)
        numeric_feats = [
            "channel_historical_cvr", "campaign_historical_cvr", "owner_historical_cvr",
            "engagement_intensity_score", "profile_completeness_score", "net_intent_score",
            "minutes_to_first_contact", "total_call_duration_sec", "call_answer_rate",
            "followup_count", "lead_age_days", "contacted_within_1hr",
            "is_employed_professional", "has_positive_intent", "is_repeat_contact",
            "was_reassigned", "is_facebook_lead", "is_owner_active",
            "has_voicebot_interaction", "is_fresh_7d",
        ]
        corr_rows = []
        for feat in numeric_feats:
            if feat in features.columns:
                c = features[feat].corr(features["is_interested"])
                if not np.isnan(c):
                    corr_rows.append((feat, round(c, 4)))
        corr_rows.sort(key=lambda x: abs(x[1]), reverse=True)

        return {"stats": stats, "correlations": corr_rows[:15]}
