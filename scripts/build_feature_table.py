"""Build the FEATURES_LEAD_INTENT table in MARKETING_DATABASE.LEAD_INTENT_ML."""

import os, re, sys, time
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        else:
            v = v.split("#")[0].strip()
        os.environ.setdefault(k.strip(), v)

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import snowflake.connector

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.crm_feature_pipeline import CRMFeaturePipeline

ACCOUNT   = os.environ["SNOWFLAKE_ACCOUNT"]
USER      = os.environ["SNOWFLAKE_USER"]
WAREHOUSE = os.environ["SNOWFLAKE_WAREHOUSE"]
ROLE      = os.environ["SNOWFLAKE_ROLE"]
SRC_DB    = os.environ["SNOWFLAKE_SOURCE_DATABASE"]
SRC_SCHEMA= os.environ["SNOWFLAKE_SOURCE_SCHEMA"]
ML_DB     = os.environ["SNOWFLAKE_ML_DATABASE"]
ML_SCHEMA = os.environ["SNOWFLAKE_ML_SCHEMA"]


def load_private_key() -> bytes:
    raw  = os.environ.get("SNOWFLAKE_PRIVATE_KEY", "")
    path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "")
    pw   = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
    if path and Path(path).exists():
        pem = Path(path).read_bytes()
    else:
        m = re.match(r"(-----BEGIN [^-]+-----)([A-Za-z0-9+/=\s]+)(-----END [^-]+-----)", raw)
        if not m:
            raise ValueError("Invalid SNOWFLAKE_PRIVATE_KEY format")
        h, b, f = m.groups()
        b = b.replace(" ", "").replace("\n", "")
        pem = (f"{h}\n" + "\n".join(b[i:i+64] for i in range(0, len(b), 64)) + f"\n{f}").encode()
    p = serialization.load_pem_private_key(pem, password=pw.encode() if pw else None, backend=default_backend())
    return p.private_bytes(serialization.Encoding.DER, 
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption()
    )


def sep(title: str = "", w: int = 65) -> None:
    if title:
        print(f"\n── {title} {'─' * max(0, w - len(title) - 4)}")
    else:
        print("─" * w)


def main() -> None:
    sep("Lead Intent Feature Pipeline")
    print(f"  Source : {SRC_DB}.{SRC_SCHEMA}")
    print(f"  Target : {ML_DB}.{ML_SCHEMA}.FEATURES_LEAD_INTENT")

    pkb = load_private_key()
    conn = snowflake.connector.connect(
        account=ACCOUNT, 
        user=USER, 
        private_key=pkb,
        warehouse=WAREHOUSE, 
        role=ROLE,
    )
    print("\n[OK] Connected to Snowflake\n")

    t0 = time.time()
    pipeline = CRMFeaturePipeline(conn, SRC_DB, SRC_SCHEMA, ML_DB, ML_SCHEMA)

    try:
        result = pipeline.build()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    elapsed = time.time() - t0

    # ── Stats ──────────────────────────────────────────────────────────────────
    s = result["stats"]
    sep("Dataset Statistics")
    print(f"  Total rows           : {int(s['TOTAL_ROWS']):>12,}")
    print(f"  Positive class (=1)  : {int(s['POSITIVE_CLASS']):>12,}")
    print(f"  Positive rate        : {float(s['POSITIVE_RATE']):>12.2%}")
    print(f"  Never contacted      : {int(s['NEVER_CONTACTED_COUNT']):>12,}")
    print(f"  Facebook leads       : {int(s['FACEBOOK_LEAD_COUNT']):>12,}")
    print(f"  Voicebot leads       : {int(s['VOICEBOT_LEAD_COUNT']):>12,}")

    sep("Feature Quality (null rates)")
    null_keys = [k for k in s if k.startswith("NULL_RATE_")]
    for k in null_keys:
        label = k.replace("NULL_RATE_", "").replace("_", " ").lower()
        print(f"  {label:<30} {float(s[k]):.1%}")

    sep("Feature Means")
    print(f"  avg lead age (days)       : {float(s['AVG_LEAD_AGE_DAYS']):.1f}")
    print(f"  avg engagement score      : {float(s['AVG_ENGAGEMENT_SCORE']):.2f}")
    print(f"  avg profile completeness  : {float(s['AVG_PROFILE_SCORE']):.2f}/12")
    print(f"  avg net intent score      : {float(s['AVG_NET_INTENT_SCORE']):.3f}")
    print(f"  avg call duration (sec)   : {float(s['AVG_TOTAL_CALL_DURATION_SEC']):.0f}")

    sep("Top Features by |Correlation| with is_interested")
    print(f"  {'Feature':<35} {'Correlation':>12}")
    print(f"  {'-------':<35} {'-----------':>12}")
    for row in result["correlations"]:
        feat, corr = row[0], row[1]
        bar = "█" * int(abs(float(corr)) * 20)
        print(f"  {feat:<35} {float(corr):>+.4f}  {bar}")

    sep()
    print(f"[DONE] Table FEATURES_LEAD_INTENT built in {elapsed:.0f}s")
    print(f"       {ML_DB}.{ML_SCHEMA}.FEATURES_LEAD_INTENT")
    conn.close()


if __name__ == "__main__":
    main()
