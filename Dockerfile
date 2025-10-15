# ================================
# Base image
# ================================
FROM python:3.12-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy files
COPY . /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install "zenml" mlflow scikit-learn pandas numpy joblib sqlmodel sqlalchemy passlib sqlalchemy_utils pymysql pydantic

# Install integrations (sem iniciar o daemon)
RUN zenml integration install sklearn -y && \
    zenml integration install mlflow -y

ENTRYPOINT ["/app/entrypoint.sh"]
