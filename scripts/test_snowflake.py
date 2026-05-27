"""Test Snowflake connectivity, preview LEAD_MASTERS, and provision LEAD_INTENT_ML schema."""

import os
import re
import sys
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        else:
            value = value.split("#")[0].strip()
        os.environ.setdefault(key.strip(), value)

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import snowflake.connector

ACCOUNT   = os.environ["SNOWFLAKE_ACCOUNT"]
USER      = os.environ["SNOWFLAKE_USER"]
WAREHOUSE = os.environ["SNOWFLAKE_WAREHOUSE"]
ROLE      = os.environ["SNOWFLAKE_ROLE"]

# Source (read-only)
SRC_DB     = os.environ["SNOWFLAKE_SOURCE_DATABASE"]   # PROD_DATABASE
SRC_SCHEMA = os.environ["SNOWFLAKE_SOURCE_SCHEMA"]     # CRM
SRC_TABLE  = "LEAD_MASTERS"
PREVIEW_ROWS = 10

# ML target (read/write)
ML_DB     = os.environ["SNOWFLAKE_ML_DATABASE"]        # MARKETING_DATABASE
ML_SCHEMA = os.environ["SNOWFLAKE_ML_SCHEMA"]          # LEAD_INTENT_ML


def load_private_key() -> bytes:
    raw  = os.environ.get("SNOWFLAKE_PRIVATE_KEY", "")
    path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "")
    pw   = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")

    if path and Path(path).exists():
        pem_bytes = Path(path).read_bytes()
    elif raw:
        m = re.match(r"(-----BEGIN [^-]+-----)([A-Za-z0-9+/=\s]+)(-----END [^-]+-----)", raw)
        if not m:
            raise ValueError("SNOWFLAKE_PRIVATE_KEY is not a valid PEM block.")
        header, b64, footer = m.groups()
        b64 = b64.replace(" ", "").replace("\n", "")
        wrapped = "\n".join(b64[i : i + 64] for i in range(0, len(b64), 64))
        pem_bytes = f"{header}\n{wrapped}\n{footer}".encode()
    else:
        raise ValueError("No private key configured.")

    p_key = serialization.load_pem_private_key(
        pem_bytes, password=pw.encode() if pw else None, backend=default_backend()
    )
    return p_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def separator(title: str = "", width: int = 72) -> None:
    if title:
        print(f"\n{'─' * 3} {title} {'─' * max(0, width - len(title) - 5)}")
    else:
        print("─" * width)


def print_resultset(cursor, rows: list) -> None:
    cols = [d[0] for d in cursor.description]
    widths = [
        min(28, max(len(c), max((len(str(r[i])) for r in rows), default=0)))
        for i, c in enumerate(cols)
    ]
    line = "  ".join(str(v)[:w].ljust(w) for v, w in zip(cols, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(v)[:w].ljust(w) for v, w in zip(row, widths)))


def main() -> None:
    print(f"Account  : {ACCOUNT}")
    print(f"User     : {USER}  |  Warehouse: {WAREHOUSE}  |  Role: {ROLE}")
    print(f"Source   : {SRC_DB}.{SRC_SCHEMA}  (read-only)")
    print(f"ML Target: {ML_DB}.{ML_SCHEMA}  (read/write)")

    pkb = load_private_key()

    try:
        conn = snowflake.connector.connect(
            account=ACCOUNT,
            user=USER,
            private_key=pkb,
            warehouse=WAREHOUSE,
            role=ROLE,
        )
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK] Connected successfully!\n")

    with conn:
        cur = conn.cursor()

        # ── 1. Preview LEAD_MASTERS (read-only) ──────────────────────────────
        separator(f"Source: {SRC_DB}.{SRC_SCHEMA}.{SRC_TABLE}  (top {PREVIEW_ROWS} rows)")
        try:
            cur.execute(f"SELECT * FROM {SRC_DB}.{SRC_SCHEMA}.{SRC_TABLE} LIMIT {PREVIEW_ROWS}")
            rows = cur.fetchall()
            if rows:
                print_resultset(cur, rows)
            else:
                print("  (table is empty)")

            cur.execute(f"SELECT COUNT(*) FROM {SRC_DB}.{SRC_SCHEMA}.{SRC_TABLE}")
            total = cur.fetchone()[0]
            print(f"\n  Total rows: {total:,}")

            cur.execute(f"DESCRIBE TABLE {SRC_DB}.{SRC_SCHEMA}.{SRC_TABLE}")
            cols = cur.fetchall()
            separator(f"Schema of {SRC_TABLE}")
            print(f"  {'COLUMN':<35} {'TYPE':<25} NULLABLE")
            print(f"  {'------':<35} {'----':<25} --------")
            for col in cols:
                print(f"  {col[0]:<35} {col[1]:<25} {col[3]}")
        except Exception as e:
            print(f"  [ERROR] Could not read {SRC_TABLE}: {e}", file=sys.stderr)

        # ── 2. Create LEAD_INTENT_ML schema in MARKETING_DATABASE ───────────
        separator(f"ML Target: {ML_DB}.{ML_SCHEMA}")
        try:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {ML_DB}.{ML_SCHEMA}")
            print(f"  [OK] Schema {ML_DB}.{ML_SCHEMA} exists (created if new).")

            cur.execute(f"SHOW TABLES IN SCHEMA {ML_DB}.{ML_SCHEMA}")
            tables = cur.fetchall()
            if tables:
                print(f"\n  Existing tables in {ML_SCHEMA}:")
                for t in tables:
                    print(f"    • {t[1]}")
            else:
                print(f"  No tables yet in {ML_SCHEMA} — ready for ML pipeline.")
        except Exception as e:
            print(f"  [ERROR] Could not provision {ML_SCHEMA}: {e}", file=sys.stderr)

    separator()
    print("Done.")


if __name__ == "__main__":
    main()
