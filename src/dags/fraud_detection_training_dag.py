from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import  AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'datamasterylab.com',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 20),
    # 'execution_timeout': timedelta(minutes=120),   #only for testing to stop training if its taking long
    'max_active_runs': 1
}

def train_model(**context):
    """Airflow wrapper for training task"""
    from fraud_detection_training import FraudDetectionTraining
    try :
        logger.info('Initializing fraud detection training')
        trainer = FraudDetectionTraining()
        model,precision = trainer.train_model()
        return {'status': 'success', 'precision': precision}

    except AirflowException as e:
        logger.error('Training failed: %s', str(e), exc_info=True)
        raise AirflowException(f'Model training failed: {str(e)}')


with DAG(
    dag_id='fraud_detection_training',
    default_args=default_args,
    description='Fraud detetcion model training pipleine',
    schedule_interval='0 3 * * *',# chrone job at every day 3am
    catchup=False,
    tags=['fraud','ML']
) as dag:
    validate_environment = BashOperator(
        task_id='validate_environment',
        bash_command='''
        echo "validating environment
        test -f /app/config.yaml &&
        test -f /app/.env &&
        echo "Environment is valid"
        '''
    )
    training_task = PythonOperator(
        task_id='execute_training',
        python_callable= train_model,
        provide_context=True
    )
    cleanup_task = BashOperator(
        task_id='cleanup_task',
        bash_command='rm-rf /app/.env &&',
        trigger_rule='all_done'
    )
    validate_environment >> training_task >> cleanup_task

    #Documentation for Airflow
    dag.doc_md = """
    #Financial fraud detection pipeline
    Daily training fraud detection model using:
        - Transaction from Kafka
        - XGBOOST classifier with precision optimization 
        - MLFlow for experiment tracking
    """


