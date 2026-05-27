"""Explore all tables in PROD_DATABASE.CRM — schemas, row counts, sample data."""

import os, re, sys
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

ACCOUNT   = os.environ["SNOWFLAKE_ACCOUNT"]
USER      = os.environ["SNOWFLAKE_USER"]
WAREHOUSE = os.environ["SNOWFLAKE_WAREHOUSE"]
ROLE      = os.environ["SNOWFLAKE_ROLE"]
SRC_DB    = os.environ["SNOWFLAKE_SOURCE_DATABASE"]
SRC_SCHEMA= os.environ["SNOWFLAKE_SOURCE_SCHEMA"]

def load_private_key():
    raw  = os.environ.get("SNOWFLAKE_PRIVATE_KEY","")
    path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH","")
    pw   = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE","")
    if path and Path(path).exists():
        pem = Path(path).read_bytes()
    else:
        m = re.match(r"(-----BEGIN [^-]+-----)([A-Za-z0-9+/=\s]+)(-----END [^-]+-----)", raw)
        h,b,f = m.groups(); b=b.replace(" ","").replace("\n","")
        pem = f"{h}\n"+"\n".join(b[i:i+64] for i in range(0,len(b),64))+f"\n{f}"
        pem = pem.encode()
    p = serialization.load_pem_private_key(pem, password=pw.encode() if pw else None, backend=default_backend())
    return p.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())

conn = snowflake.connector.connect(account=ACCOUNT, user=USER, private_key=load_private_key(), warehouse=WAREHOUSE, role=ROLE)
cur  = conn.cursor()

print(f"\n{'='*60}")
print(f"  Exploring {SRC_DB}.{SRC_SCHEMA}")
print(f"{'='*60}")

cur.execute(f"SHOW TABLES IN SCHEMA {SRC_DB}.{SRC_SCHEMA}")
tables = [t[1] for t in cur.fetchall()]
print(f"\nFound {len(tables)} tables: {', '.join(tables)}\n")

for tbl in tables:
    full = f"{SRC_DB}.{SRC_SCHEMA}.{tbl}"
    print(f"\n{'─'*60}")
    print(f"TABLE: {tbl}")
    print(f"{'─'*60}")

    # row count
    cur.execute(f"SELECT COUNT(*) FROM {full}")
    print(f"  Rows: {cur.fetchone()[0]:,}")

    # schema
    cur.execute(f"DESCRIBE TABLE {full}")
    cols = cur.fetchall()
    print(f"  Columns ({len(cols)}):")
    for c in cols:
        print(f"    {c[0]:<40} {c[1]}")

    # sample (3 rows)
    cur.execute(f"SELECT * FROM {full} LIMIT 3")
    rows = cur.fetchall()
    if rows:
        headers = [d[0] for d in cur.description]
        print(f"\n  Sample (3 rows):")
        for r in rows:
            print("    " + " | ".join(f"{h}={str(v)[:30]}" for h,v in zip(headers,r)))

conn.close()
print("\nDone.")
