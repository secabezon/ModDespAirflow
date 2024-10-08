from airflow import DAG
import sys
from datetime import datetime, timedelta
from airflow.operators.python import PythonVirtualenvOperator
from airflow.utils.email import send_email
from airflow.operators.bash_operator import BashOperator
from airflow.decorators import task
from airflow.operators.python_operator import PythonOperator
PATH_COMMON='../'
sys.path.append(PATH_COMMON)
from common.add_task import task_preprocess,task_prediction

def failure_email(context):
    task_instance = context['task_instance']
    task_status = 'Failed'

    subject = f'Airflow task {task_instance.task_id} {task_status}'
    body = f'The task {task_instance.task_id} completed with status {task_status}.  \n\n'\
    f' The task execution date is: {context['execution_date']}\n'
    to_email = 'secabezon21@gmail.com'
    send_email(to= to_email, subject=subject, html_content=body)

def success_email(context):
    task_instance = context['task_instance']
    task_status = 'Success'

    subject = f'Airflow task {task_instance.task_id} {task_status}'
    body = f'The task {task_instance.task_id} completed with status {task_status}.  \n\n'\
    f' The task execution date is: {context['execution_date']}\n'
    to_email = 'secabezon21@gmail.com'
    send_email(to= to_email, subject=subject, html_content=body)

default_args={
    'owner':'airflow',
    'start_date':datetime(2024,3,14),
    'scheduler_interval':'None',
    'email_on_failure': True,
    'email_on_success': True,
    'email_one_retry': False
    #'retries': 1,
    #'retry_delay':timedelta(seconds=5)
}
with DAG(
    'preprocess_predict',
    default_args=default_args,
    description='demo de notificacion via email',
    schedule=None,
    on_failure_callback=lambda context: failure_email(context),
    on_success_callback=lambda context: success_email(context)
    ) as dag:
    preprocess_data = task_preprocess()
    predict_task =  task_prediction(preprocess_data)
    preprocess_data >> predict_task
