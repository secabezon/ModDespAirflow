from airflow import DAG
from datetime import datetime, timedelta
import sys
from airflow.utils.email import send_email
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.decorators import dag

PATH_COMMON='../'#lo que hace es ir hacia afuera un escalon para poder ir a la carpeta common
sys.path.append(PATH_COMMON)#ejecuta lo explicado en path common
from common.add_task import task_train


@dag(
    dag_id='training',
    start_date=datetime(2022,1,1),
    schedule=None
)
def mydag():
    task_train()

dagtrain=mydag()