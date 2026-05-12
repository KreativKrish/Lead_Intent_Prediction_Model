FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Apache Airflow
RUN pip install --no-cache-dir \
    apache-airflow==2.8.1 \
    apache-airflow-providers-snowflake==4.2.2 \
    postgresql \
    psycopg2-binary

# Copy requirements
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create Airflow home directory
RUN mkdir -p /airflow

ENV AIRFLOW_HOME=/airflow

# Initialize Airflow database
RUN airflow db init || true

EXPOSE 8080
