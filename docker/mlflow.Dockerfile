FROM python:3.10-slim

WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow[tracking]=2.10.2

# Create directories for MLflow
RUN mkdir -p /mlflow/artifacts /mlflow/db

# Expose port
EXPOSE 5000

# Run MLflow tracking server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
