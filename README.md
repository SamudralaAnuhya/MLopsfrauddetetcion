# Real-Time Fraud Detection with MLOps

A scalable, real-time fraud detection system built from scratch—designed to mirror real-world banking use cases. This project demonstrates industry best practices for machine learning model creation, retraining, and deployment while leveraging powerful tools like Apache Kafka, Airflow, MLflow, MinIO, and Apache Spark Structured Streaming.

## Features

- End-to-End fraud detection pipeline architecture
- High-throughput Kafka clusters for reliable message passing
- Real-time inference with Spark Structured Streaming
- Model versioning and experiment tracking with MLflow
- Advanced feature engineering and F2-score optimization
- Continuous feedback loop for model retraining
- Fully containerized setup with Docker Compose

## System Architecture

The system follows a modern microservices architecture to ensure scalability and maintainability:

1. **Data Generation & Ingestion**: Synthetic transaction data flows into Kafka
2. **Training Pipeline**: Airflow orchestrates daily model training
3. **Model Registry**: MLflow tracks experiments and serves models
4. **Real-time Inference**: Spark Streaming processes transactions and makes predictions
5. **Feedback Loop**: Predictions are routed back for continuous improvement

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Data Ingestion | Python, Faker, Apache Kafka |
| Workflow Orchestration | Apache Airflow (Celery + Redis + PostgreSQL) |
| Model Training | XGBoost, SMOTE, RandomizedSearchCV |
| Experiment Tracking | MLflow + Model Registry |
| Artifact Storage | MinIO (S3-compatible) |
| Metadata Storage | PostgreSQL |
| Real-time Processing | Apache Spark (Structured Streaming) |
| Feedback Loop | Kafka → Airflow → Retraining DAG |
| Containerization | Docker Compose |

## Workflow Details

### 1. Transaction Simulation & Ingestion
- Realistic synthetic data generated with Python `faker`
- JSON schema validation ensures data quality
- Transactions published to Kafka topic: `transactions`

### 2. Airflow-Powered Daily Model Training
- DAG scheduled at `03:00 UTC`
- Handles data preparation, model training, and evaluation:
  - SMOTE class balancing for imbalanced fraud data
  - XGBoost with hyperparameter tuning via `RandomizedSearchCV`
  - Optimal threshold selection using `F2-score (β=2)`
  - Comprehensive model metrics logging
- Average runtime: ~15 minutes for 1M transactions

### 3. MLflow Logging & Model Registry
- Tracks all experiment parameters, metrics, and artifacts
- Logs precision, recall, F2 score, confusion matrix, and ROC AUC
- Stores model artifacts in MinIO (S3-compatible storage)
- Maintains model metadata in PostgreSQL
- Manages model lifecycle: `None → Staging → Production`

### 4. Real-Time Inference via Spark Streaming
- Consumes transactions from Kafka in real-time
- Loads the latest production model from MLflow
- Applies feature engineering and inference
- Publishes predictions to Kafka topic: `fraud_predictions`

Example prediction output:
```json
{
  "transaction_id": "1234-uuid",
  "amount": 899.0,
  "rolling_avg_7d": 400.5,
  "amount_to_avg_ratio": 2.24,
  "prediction": 1
}
```

### 5. Feedback Loop for Retraining
- Prediction stream feeds back into the training pipeline
- Enables continuous model improvement
- Adjusts to changing fraud patterns over time

## Installation & Setup

### Prerequisites
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended

### Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-fraud-detection.git
cd mlops-fraud-detection
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env file to customize settings if needed
```

3. Start all services:
```bash
docker compose --profile flower up -d --build
```

4. To stop all services:
```bash
docker compose --profile flower down
```

### Accessing Components

| Service | URL |
|---------|-----|
| Airflow | http://localhost:8080 | 
| MLflow | http://localhost:5000 | 
| MinIO | http://localhost:9001 | 
| Spark UI | http://localhost:4040 | 

## What You'll Learn

This project demonstrates:
- Building production-grade ML training pipelines
- Tuning models for fraud-specific metrics (prioritizing F2-score)
- Deploying real-time ML inference at scale with Spark
- Versioning and serving models using MLflow and MinIO
- Structuring feedback loops for continuous model improvement

## Roadmap

- [ ] Add Prometheus + Grafana for comprehensive monitoring
- [ ] Implement A/B testing for multiple model comparison
- [ ] Add CI/CD integration with GitHub Actions
- [ ] Implement drift detection and automatic retraining triggers
- [ ] Add advanced anomaly detection techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Built with 💡 and ☕ by Anuhysamudrala

---

If this project helped you, please consider giving it a star ⭐ and sharing it with others!
