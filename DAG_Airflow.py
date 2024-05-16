from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'gender_detection_pipeline',
    default_args=default_args,
    description='Gender Detection Pipeline',
    schedule_interval=timedelta(days=1),
)

def check_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} does not exist")

check_folder_task = PythonOperator(
    task_id='check_folder_exists',
    python_callable=check_folder_exists,
    op_kwargs={'folder_path': 'include/dataset/{female,male}'},
    dag=dag,
)

preprocess_images_task = SparkSubmitOperator(
    task_id='preprocess_images',
    application='preprocess_images.py',
    conn_id='spark_default',
    dag=dag,
)

check_folder_task >> preprocess_images_task
