"""Snowflake data warehouse connector."""

import os
from contextlib import contextmanager

import snowflake.connector

from ..utils.logger import get_logger

logger = get_logger(__name__)


class SnowflakeConnector:
    """Manager for Snowflake connections."""

    def __init__(self):
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.database = os.getenv("SNOWFLAKE_DATABASE", "LEAD_INTENT_DB")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", "FEATURES")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.role = os.getenv("SNOWFLAKE_ROLE", "ANALYST")

    @contextmanager
    def get_connection(self):
        """Context manager for Snowflake connections.

        Yields:
            Snowflake connection object.
        """
        try:
            conn = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                database=self.database,
                schema=self.schema,
                warehouse=self.warehouse,
                role=self.role,
            )
            logger.info(f"Connected to Snowflake: {self.account}/{self.database}/{self.schema}")
            yield conn
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
        finally:
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
