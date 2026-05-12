"""Initialize Snowflake with tables and sample data."""

import os

import numpy as np
import pandas as pd


def create_tables():
    """Create necessary tables in Snowflake."""
    from src.data.snowflake_connector import SnowflakeConnector

    connector = SnowflakeConnector()

    # Create TRAINING_DATA table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS TRAINING_DATA (
        ID INT PRIMARY KEY,
        LEAD_SCORE FLOAT,
        COMPANY_SIZE INT,
        ENGAGEMENT_SCORE FLOAT,
        RESPONSE_TIME_HOURS FLOAT,
        EMAIL_OPEN_RATE FLOAT,
        EMAIL_CLICK_RATE FLOAT,
        PAGE_VIEWS INT,
        TIME_SINCE_SIGNUP_DAYS INT,
        INDUSTRY VARCHAR,
        COMPANY_TYPE VARCHAR,
        LOCATION VARCHAR,
        PRODUCT_INTEREST VARCHAR,
        SOURCE VARCHAR,
        SALES_STAGE VARCHAR,
        INTENT_LABEL INT
    )
    """

    try:
        connector.execute_query(create_table_query)
        print("Created TRAINING_DATA table")
    except Exception as e:
        print(f"Table creation error (may already exist): {e}")


def generate_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate sample training data."""
    data = {
        "ID": range(1, n_rows + 1),
        "LEAD_SCORE": np.random.uniform(0, 100, n_rows),
        "COMPANY_SIZE": np.random.randint(1, 5000, n_rows),
        "ENGAGEMENT_SCORE": np.random.uniform(0, 100, n_rows),
        "RESPONSE_TIME_HOURS": np.random.uniform(0, 48, n_rows),
        "EMAIL_OPEN_RATE": np.random.uniform(0, 1, n_rows),
        "EMAIL_CLICK_RATE": np.random.uniform(0, 1, n_rows),
        "PAGE_VIEWS": np.random.randint(0, 200, n_rows),
        "TIME_SINCE_SIGNUP_DAYS": np.random.randint(0, 365, n_rows),
        "INDUSTRY": np.random.choice(
            ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"],
            n_rows
        ),
        "COMPANY_TYPE": np.random.choice(
            ["SaaS", "Enterprise", "Startup", "SMB", "Fortune500"],
            n_rows
        ),
        "LOCATION": np.random.choice(
            ["US", "EU", "APAC", "LATAM", "Canada"],
            n_rows
        ),
        "PRODUCT_INTEREST": np.random.choice(
            ["Enterprise", "SMB", "Startup", "Premium"],
            n_rows
        ),
        "SOURCE": np.random.choice(
            ["LinkedIn", "Direct", "Partner", "Event", "Webinar"],
            n_rows
        ),
        "SALES_STAGE": np.random.choice(
            ["Awareness", "Consideration", "Decision", "Qualification", "Nurture"],
            n_rows
        ),
        "INTENT_LABEL": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),  # 30% intent
    }
    return pd.DataFrame(data)


def main():
    """Main execution."""
    print("Initializing Snowflake with sample data...")

    # Check if Snowflake credentials are configured
    if not all([
        os.getenv("SNOWFLAKE_ACCOUNT"),
        os.getenv("SNOWFLAKE_USER"),
        os.getenv("SNOWFLAKE_PASSWORD"),
    ]):
        print("Snowflake credentials not configured. Skipping data seeding.")
        print("Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD env vars.")
        return

    try:
        create_tables()

        # Generate and insert sample data
        df = generate_sample_data(n_rows=1000)
        print(f"Generated {len(df)} sample rows")
        print("Sample data generated. Ready to load into Snowflake.")

    except Exception as e:
        print(f"Error during seeding: {e}")


if __name__ == "__main__":
    main()
