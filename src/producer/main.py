import json
import logging
import os
import random
import time
import signal
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from random import randint

from confluent_kafka import Producer
from dotenv import load_dotenv
from faker import Faker
from jsonschema import validate, ValidationError, FormatChecker

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path="../.env")

# Faker instance for generating random data
fake = Faker()


class TransactionProducer():
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_username = os.getenv("KAFKA_USERNAME")
        self.kafka_password = os.getenv("KAFKA_PASSWORD")
        self.topic = os.getenv("KAFKA_TOPIC", "transactions")
        self.running = False

        # Producer configuration for Confluent Kafka
        self.producer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "transaction-producer",
            "compression.type": "gzip",
            "linger.ms": 5,
            "batch.size": 16384,
        }

        if self.kafka_username and self.kafka_password:
            self.producer_config.update({
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": self.kafka_username,
                "sasl.password": self.kafka_password,
            })
        else:
            self.producer_config["security.protocol"] = "PLAINTEXT"
        try:
            self.producer = Producer(self.producer_config)
            logger.info("Confluent Kafka Producer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Confluent Kafka Producer: {str(e)}")
            raise e

if __name__ == '__main__':
    producer = TransactionProducer()
    producer.run_continuous_production()
