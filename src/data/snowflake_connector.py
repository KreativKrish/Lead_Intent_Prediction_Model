"""Snowflake data warehouse connector."""

import os
import re
from contextlib import contextmanager
from pathlib import Path

import snowflake.connector

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _load_private_key_bytes() -> bytes | None:
    """Load DER-encoded private key from env (inline PEM or file path)."""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    raw = os.getenv("SNOWFLAKE_PRIVATE_KEY", "")
    path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "")
    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")

    if path and Path(path).exists():
        pem_bytes = Path(path).read_bytes()
    elif raw:
        match = re.match(
            r"(-----BEGIN [^-]+-----)([A-Za-z0-9+/=\s]+)(-----END [^-]+-----)",
            raw,
        )
        if not match:
            raise ValueError("SNOWFLAKE_PRIVATE_KEY is not a valid PEM block.")
        header, b64, footer = match.groups()
        b64 = b64.replace(" ", "").replace("\n", "")
        wrapped = "\n".join(b64[i : i + 64] for i in range(0, len(b64), 64))
        pem_bytes = f"{header}\n{wrapped}\n{footer}".encode()
    else:
        return None

    pw = passphrase.encode() if passphrase else None
    p_key = serialization.load_pem_private_key(pem_bytes, password=pw, backend=default_backend())
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


class SnowflakeConnector:
    """Manager for Snowflake connections.

    Source (read-only) : SNOWFLAKE_SOURCE_DATABASE / SNOWFLAKE_SOURCE_SCHEMA
    ML target (write)  : SNOWFLAKE_ML_DATABASE     / SNOWFLAKE_ML_SCHEMA
    Pass `target="ml"` to write to the ML database.
    """

    def __init__(self, target: str = "source"):
        self.account   = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user      = os.getenv("SNOWFLAKE_USER")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.role      = os.getenv("SNOWFLAKE_ROLE", "ANALYST")

        if target == "ml":
            self.database = os.getenv("SNOWFLAKE_ML_DATABASE", "MARKETING_DATABASE")
            self.schema   = os.getenv("SNOWFLAKE_ML_SCHEMA", "LEAD_INTENT_ML")
        else:
            self.database = os.getenv("SNOWFLAKE_SOURCE_DATABASE", "PROD_DATABASE")
            self.schema   = os.getenv("SNOWFLAKE_SOURCE_SCHEMA", "CRM")

        # prefer private key; fall back to password
        self._private_key = _load_private_key_bytes()
        self._password = os.getenv("SNOWFLAKE_PASSWORD") if not self._private_key else None

    def _connect_kwargs(self) -> dict:
        kwargs = dict(
            account=self.account,
            user=self.user,
            database=self.database,
            schema=self.schema,
            warehouse=self.warehouse,
            role=self.role,
        )
        if self._private_key:
            kwargs["private_key"] = self._private_key
        else:
            kwargs["password"] = self._password
        return kwargs

    @contextmanager
    def get_connection(self):
        """Context manager for Snowflake connections.

        Yields:
            Snowflake connection object.
        """
        conn = None
        try:
            conn = snowflake.connector.connect(**self._connect_kwargs())
            logger.info(f"Connected to Snowflake: {self.account}/{self.database}/{self.schema}")
            yield conn
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str):
        """Execute a query and return results.

        Args:
            query: SQL query string.

        Returns:
            Query results.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()

    def fetch_table(self, table_name: str, limit: int | None = None) -> list:
        """Fetch data from a table.

        Args:
            table_name: Name of the table.
            limit: Optional limit on rows.

        Returns:
            List of rows.
        """
        query = f"SELECT * FROM {self.schema}.{table_name}"
        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Fetching data from {table_name}")
        return self.execute_query(query)

    def fetch_dataframe(self, query: str):
        """Fetch query result as pandas DataFrame.

        Args:
            query: SQL query string.

        Returns:
            Pandas DataFrame.
        """
        import pandas as pd

        with self.get_connection() as conn:
            logger.info("Fetching data as DataFrame")
            return pd.read_sql(query, conn)
