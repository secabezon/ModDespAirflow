FROM python:3.8-slim

ENV PYTHONUNBUFFERED True
ENV PATH="/root/.local/bin:${PATH}"

# Crear el directorio en el contenedor
RUN mkdir -p /opt/airflow/dags/data

# Cambiar el directorio de trabajo
WORKDIR /opt/airflow/dags/data

# Copiar toda la carpeta data al contenedor
COPY ./data /opt/airflow/dags/data

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install joblib
