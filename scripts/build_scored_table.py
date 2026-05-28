"""Build the LEAD_INTENT_SCORED table in MARKETING_DATABASE.LEAD_INTENT_ML.

Usage:
    python scripts/build_scored_table.py
"""

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

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "http://localhost:5000")

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import snowflake.connector

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.scored_feature_pipeline import ScoredFeaturePipeline

ACCOUNT   = os.environ["SNOWFLAKE_ACCOUNT"]
USER      = os.environ["SNOWFLAKE_USER"]
WAREHOUSE = os.environ["SNOWFLAKE_WAREHOUSE"]
ROLE      = os.environ["SNOWFLAKE_ROLE"]
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
    return p.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def sep(title: str = "", w: int = 65) -> None:
    if title:
        print(f"\n── {title} {'─' * max(0, w - len(title) - 4)}")
    else:
        print("─" * w)


def main() -> None:
    sep("Lead Intent Scored Feature Pipeline")
    print(f"  Source : {ML_DB}.{ML_SCHEMA}.{ScoredFeaturePipeline.SOURCE_TABLE}")
    print(f"  Target : {ML_DB}.{ML_SCHEMA}.{ScoredFeaturePipeline.TARGET_TABLE}")

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
    pipeline = ScoredFeaturePipeline(conn, ML_DB, ML_SCHEMA)

    try:
        result = pipeline.build()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)

    elapsed = time.time() - t0

    sep("Dataset Statistics")
    print(f"  Total rows           : {result['total_rows']:>12,}")
    print(f"  Positive class (=1)  : {result['positive_class']:>12,}")
    print(f"  Positive rate        : {result['positive_rate']:>12.2%}")
    print(f"  Train rows           : {result['train_rows']:>12,}")
    print(f"  Validation rows      : {result['val_rows']:>12,}")
    print(f"  Test rows            : {result['test_rows']:>12,}")

    sep("Score Means")
    for name, mean in result["score_means"].items():
        print(f"  {name:<25} : {mean:>8.2f}")

    sep("Normalization Parameters")
    for name, val in result["normalization_params"].items():
        print(f"  {name:<25} : {val:>10.6f}")

    sep()
    print(f"[DONE] Table {ScoredFeaturePipeline.TARGET_TABLE} built in {elapsed:.0f}s")
    print(f"       {ML_DB}.{ML_SCHEMA}.{ScoredFeaturePipeline.TARGET_TABLE}")
    conn.close()


if __name__ == "__main__":
    main()
