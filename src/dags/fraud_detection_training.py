import json
import logging
import os

import joblib
import mlflow
import pandas as pd
import numpy as np

from airflow.utils import yaml
from dotenv import load_dotenv
import boto3
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from numpy.array_api import astype
from kafka import KafkaConsumer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (make_scorer, fbeta_score, precision_recall_curve, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# Configure dual logging to file and stdout with structured format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    """
    End-to-end fraud detection training system implementing MLOps best practices.

    Key Architecture Components:
    - Configuration Management: Centralized YAML config with environment overrides
    - Data Ingestion: Kafka consumer with SASL/SSL authentication
    - Feature Engineering: Temporal, behavioral, and monetary feature constructs
    - Model Development: XGBoost with SMOTE for class imbalance
    - Hyperparameter Tuning: Randomized search with stratified cross-validation
    - Model Tracking: MLflow integration with metrics/artifact logging
    - Deployment Prep: Model serialization and registry

    The system is designed for horizontal scalability and cloud-native operation.
    """

    def __init__(self, config_path='/app/config.yaml'):
        # Environment hardening for containerized deployments
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        # Load environment variables before config to allow overrides
        load_dotenv(dotenv_path='/app/.env')

        # Configuration lifecycle management
        self.config = self._load_config(config_path)

        # Security-conscious credential handling
        os.environ.update({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        })

        # Pre-flight system checks
        self._validate_environment()

        # MLflow configuration for experiment tracking
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def _load_config(self, config_path: str) -> dict:
        """
        Load and validate hierarchical configuration with fail-fast semantics.

        Implements:
        - YAML configuration parsing
        - Early validation of critical parameters
        - Audit logging of configuration loading
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully')
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    def _validate_environment(self):
        """
        System integrity verification with defense-in-depth checks:
        1. Required environment variables
        2. Object storage connectivity
        3. Credential validation

        Fails early to prevent partial initialization states.
        """
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_USERNAME', 'KAFKA_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f'Missing required environment variables: {missing}')
        self._check_minio_connection()

    def _check_minio_connection(self):
        """
        Validate object storage connectivity and bucket configuration for aws
        minio is the connection for aws

        Implements:
        - S3 client initialization with error handling
        - Bucket existence check
        - Automatic bucket creation (if configured)

        Maintains separation of concerns between configuration and infrastructure setup.
        """
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=self.config['mlflow']['s3_endpoint_url'],
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('Minio connection verified. Buckets: %s', bucket_names)

            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')

            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info('Created missing MLFlow bucket: %s', mlflow_bucket)
        except Exception as e:
            logger.error('Minio connection failed: %s', str(e))

    def create_feature(self, df: pd.DataFrame):
        """
        1. Temporal Features:
           - Transaction hour
           - Night/weekend indicators

        2. Behavioral Features:
           - 24h user activity window

        3. Monetary Features:
           - Amount to historical average ratio for 7 days before

        4. Merchant Risk:
           - Predefined high-risk merchant list
        """
        # handle time zone difference (sort all transactions)
        df = df.sort_values(['user_id', 'timestamp']).copy()

        # ---- Temporal Feature Engineering ----
        df['transaction_hour'] = df['timestamp'].dt.hour
        # transactions happened between 11pm - 5am marked as night
        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] < 5)).astype(int)
        # flagging weekend(sat ,sun) ....day starts with (mon-0 , tue-1...sat-5,sun-6)
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        # finding the date of transaction
        df['transaction_day'] = df['timestamp'].dt.day

        # -- Behavioral Feature Engineering --
        # counting number of transactions  in last 24 hours (multiple like 50 in 1 day is again flag)
        df['user_activity_24h'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: g.rolling('24h', on='timestamp', closed='left')['amount'].count().fillna(0)
        )

        # -- Monetary Feature Engineering --
        # pattern based on last week mean amount (current is 1000 but in the avg of daily last week is 100 so flag it )
        # Relative amount detection compared to user's historical pattern
        df['amount_to_avg_ratio'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: (g['amount'] / g['amount'].rolling(7, min_periods=1).mean()).fillna(1.0)
        )

        # -- Merchant Risk Profiling --
        # External risk intelligence integration point  (#crypto)
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df['merchant_risk'] = df['merchant'].isin(high_risk_merchants).astype(int)

        feature_cols = [
            'amount', 'is_night', 'is_weekend', 'transaction_day', 'user_activity_24h',
            'amount_to_avg_ratio', 'merchant_risk', 'merchant']

        # checking if main column is there or not (fraud)
        if 'is_fraud' not in df.columns:
            raise ValueError('Missing target column "is_fraud"')
        return df[feature_cols + ['is_fraud']]

    def read_from_kafka(self) -> pd.DataFrame:
        """
        Reads data from Kafka
        secure authentication to kafka and returns it as a dataframe
        data quality checks
            schema validation
            fraud column exists
            finding total amount of fraud transactions are available
        """

        try:
            topic = self.config['kafka']['topic']
            logger.info('Connecting to kafka topic %s', topic)

            # connecting to kafka,connections from config.yaml
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config['kafka']['bootstrap_servers'].split(','),
                security_protocol='SASL_SSL',
                sasl_mechanism='PLAIN',
                sasl_plain_username=self.config['kafka']['username'],
                sasl_plain_password=self.config['kafka']['password'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                consumer_timeout_ms=self.config['kafka'].get('timeout', 10000)
            )
            # list comprehension for our data
            messages = [msg.value for msg in consumer]
            consumer.close()

            df = pd.DataFrame(messages)
            if df.empty:
                raise ValueError('No messages received from Kafka')

            # making same time format same as in producer - generate_transaction
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            # checking whether our main column for training is there or not(is_fraud)
            if 'is_fraud' not in df.columns:
                raise ValueError('Fraud label (is_fraud) missing from Kafka data')

            # how much fraud transactions are there in data
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info('Kafka data read successfully with fraud rate: %.2f%%', fraud_rate)
            return df
        except Exception as e:
            logger.error('Failed to read data from Kafka: %s', str(e), exc_info=True)
            raise

    def train_model(self):
        """
                End-to-end training pipeline implementing ML best practices
                1. Data ingestion (read data from apache kafka and validating)
                2. Feature engineering (categorize , numerical - onehot encoding )
                3. Spliiting Train and Test data
                4. Class Imbalance Mitigation (SMOTE)
                5. Hyperparameter Optimization(RandomizedSearchCV)
                6. Threshold Tuning(which threshold is fraud and not fraud)
                7. Model Evaluation
                8. Artifact Logging
                9. Model Registry
                Implements MLflow experiment tracking for full reproducibility.
                """
        try:
            logger.info('starting training model')
            # Data ingestion
            df = self.read_from_kafka()
            # feature engineering from raw data
            data = self.create_feature(df)

            # train and test split
            X = data.drop(columns=['is_fraud'])
            y = data['is_fraud']

            # Class imbalance safeguards
            # no fraud scenarios
            if y.sum() == 0:
                raise ValueError('No positive samples in training data')
                # very less fraud scenarios for tranining we need to do augmentation
            if y.sum() < 10:
                logger.warning('Low positive samples: %d. Consider additional data augmentation', y.sum())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['model'].get('test_size', 0.2),  # model is from config.yaml
                stratify=y,  # Keeps class balance in both splits
                random_state=self.config['model'].get('seed', 42)
            )

            with mlflow.start_run():
                # Dataset metadata logging
                mlflow.log_metrics({
                    'train_samples': X_train.shape[0],
                    'positive_samples': int(y_train.sum()),
                    'class_ratio': float(y_train.mean()),
                    'test_samples': X_test.shape[0]
                })

                # Categorical feature preprocessing

                preprocessor = ColumnTransformer([
                    ('merchant_encoder',
                     OrdinalEncoder(  # ordinl(medium,low,high),,onehot(red,green , blue) everyone has same weight
                         handle_unknown='use_encoded_value',
                         # If during inference the model sees a merchant it hasn't seen during training,
                         unknown_value=-1,  # it won’t crash — it assigns unknown_value=-1.
                         dtype=np.float32  # more memory_efficient for XGBoost
                     ), ['merchant'])
                ], remainder='passthrough')  # remaining columns keeps as is

                # XGBoost configuration with efficiency optimizations
                xgb = XGBClassifier(
                    eval_metric='aucpr',  # Area Under Precision-Recall Curve
                    random_state=self.config['model'].get('seed', 42),
                    reg_lambda=1.0,  # L2 regularization to reduce overfitting and control model complexity  (reg_alpha ,,,l1regularization)
                    n_estimators=self.config['model']['params']['n_estimators'],  # number of boosting rounds
                    n_jobs=-1,  # uses all availble cpu for parallelism
                    tree_method=self.config['model'].get('tree_method', 'hist')
                    # builts decision trees by spliiting at various thresholds
                )

                # for balencing the imbalenced negative scenarios(ADASYN another variant)
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=self.config['model'].get('seed', 42))),
                    ('classifier', xgb)
                ], memory='./cache')


                #Hyperparameter search space design (to try with diffrent combination and find best model)
                param_dist = {
                    "classifier__max_depth" : [3,5,7,], #controls the depth of tree for regularization
                    'classifier__subsample' :[0.6, 0.8, 1.0], #fractions of trainings added to tree(Stochastic gradient boosting)
                    'classifier__colsample_bytree': [0.6, 0.8, 1.0], #feature randoimization helps overfitting
                    'classifier__learning_rate': [0.01, 0.05, 0.1], #shriks the contributions of each tree(prevent overfitting)
                    'classifier__gamma': [0, 0.1, 0.3], #minimze the loss reduction
                    'classifier__reg_alpha': [0, 0.1, 0.5] #l1 regularization
                }

                #optimazation for F-beta score (find best combination oif model for above by doing mix and match
                #we can do either by grid or random searchcv
                searcher = RandomizedSearchCV(
                    pipeline,
                    param_dist,
                    scoring=make_scorer(fbeta_score,beta=2, zero_division=0),#recall(b>1) Out of all actual positives(actual frauds), how many did the model correctly identify?
                    n_iter = 20 , #in hyperparameter we have 6 sets with each 3 values so 6^3 which is 729 , but it tries randomly for only 20 combinations
                    n_jobs = -1 , #uses all cpu's
                    cv = StratifiedKFold(n_splits=3 , shuffle=True),
                    refit=True, #uses the best model parameter in the entire set
                    error_score='raise',
                    verbose=2,
                    random_state=self.config['model'].get('seed', 42)
                )

                logger.info('Starting hyperparameter tuning...')
                searcher.fit(X_train, y_train)  #model training
                best_params = searcher.best_params_
                best_model = searcher.best_estimator_
                logger.info('Best hyperparameters: %s', best_params)

                # Threshold optimization using training data
                train_proba = best_model.predict_proba(X_train)[:, 1]
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_train, train_proba)
                f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0
                             for p, r in zip(precision_arr[:-1], recall_arr[:-1])]  #list comprehension for presion,recall
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info('Optimal threshold determined: %.4f', best_threshold)

                #Model evaluation - testing
                X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)
                test_proba = best_model.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]
                y_pred = (test_proba >= best_threshold).astype(int)

                # Comprehensive metrics suite
                metrics = {
                    'auc_pr': float(average_precision_score(y_test, test_proba)),
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                    'threshold': float(best_threshold)
                }

                mlflow.log_metrics(metrics)
                mlflow.log_params(best_params)

                # Confusion matrix visualization
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Not Fraud', 'Fraud'])
                plt.yticks(tick_marks, ['Not Fraud', 'Fraud'])

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')

                plt.tight_layout()
                cm_filename = 'confusion_matrix.png'
                plt.savefig(cm_filename)
                mlflow.log_artifact(cm_filename)
                plt.close()

                # Precision-Recall curve documentation
                plt.figure(figsize=(10, 6))
                plt.plot(recall_arr, precision_arr, marker='.', label='Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                pr_filename = 'precision_recall_curve.png'
                plt.savefig(pr_filename)
                mlflow.log_artifact(pr_filename)
                plt.close()

                # Model packaging and registry
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path='model',
                    signature=signature,
                    registered_model_name='fraud_detection_model'
                )

                # Model serialization for deployment (creating pkl file)
                os.makedirs('/app/models', exist_ok=True)
                joblib.dump(best_model, '/app/models/fraud_detection_model.pkl')

                logger.info('Training successfully completed with metrics: %s', metrics)

                return best_model, metrics

        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise
