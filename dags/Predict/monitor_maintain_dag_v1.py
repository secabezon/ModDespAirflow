from airflow import DAG
from datetime import datetime, timedelta
from airflow.utils.email import send_email
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

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


def python_command1():
    print('How are you=')

def python_command2():
    raise ValueError('Forzando un fallo en la tarea')


default_args={
    'owner':'airflow',
    'start_date':datetime(2024,3,14),
    'scheduler_interval':'None',
    'email_on_failure': True,
    'email_on_success': True,
    'email_one_retry': False,
    'retries': 1,
    'retry_delay':timedelta(seconds=5)
}
with DAG(
    'monitoring_v1',
    default_args=default_args,
    description='demo de notificacion via email',
    schedule=None,
    on_failure_callback=lambda context: failure_email(context),
    on_success_callback=lambda context: success_email(context)
    ) as dag:
    task=PythonOperator(
        task_id='execute_python_command',
        python_callable=python_command2
    )
    task2=BashOperator(
        task_id='execute_bash_command',
        bash_command = 'hello world'
    )
    task >> task2