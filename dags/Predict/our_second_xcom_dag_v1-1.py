import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Argumentos por defecto para las tareas del DAG
default_args = {
    'owner': 'datapath',
    'depends_on_past': False,
    'start_date': datetime(2023, 6, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Función para enviar datos a XCom
def push_xcom(ti):
    value_to_push = 'Este es un valor enviado a XCom'
    ti.xcom_push(key='sample_xcom_key', value=value_to_push)
    print(f'Valor enviado a XCom: {value_to_push}')

# Función para recuperar datos de XCom
def pull_xcom(ti):
    pulled_value = ti.xcom_pull(key='sample_xcom_key', task_ids='push_xcom_task')
    print(f'Valor recuperado de XCom: {pulled_value}')

# Definición del DAG 
with DAG(
    dag_id='our_second_dag_v1.1',
    default_args=default_args,
    description='Un DAG simple de ejemplo usando XCom',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    # Definición de la primera tarea que enviará datos a XCom
    push_xcom_task = PythonOperator(
        task_id='push_xcom_task',
        python_callable=push_xcom,
    )

    # Definición de la segunda tarea que recuperará datos de XCom
    pull_xcom_task = PythonOperator(
        task_id='pull_xcom_task',
        python_callable=pull_xcom,
    )

    # Establecer la dependencia entre las tareas
    push_xcom_task >> pull_xcom_task
