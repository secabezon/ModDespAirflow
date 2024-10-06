from airflow.decorators import task
from datetime import datetime, timedelta


@task.virtualenv(#task que tiene un ambiente virtual, como en nuestro pc. Si se usa llibreria tal cual, va a fallar porque faltarian las librerias necesarias para correr el codigo. Importante se usa virtual env porque se requieren instalar librerias especificas que no estan en la biblioteca principal de python
    task_id='trainingmodel',#Se define ID
    requirements=[
        'pandas','scikit-learn','joblib','numpy'
    ],#parametros requirements se define las librerias que necesitan instalarse en el ambiente virtual, además de las que tiene airflow por defecto
    system_site_packages=False
)
def task_train():#Dentro del ambiente trabajo tal cual un codigo python
    import pandas as pd
    import sys
    from datetime import datetime
    from sklearn.ensemble._forest import RandomForestRegressor
    import joblib
    import numpy as np

    PATH_COMMON='../'#lo que hace es ir hacia afuera un escalon para poder ir a la carpeta common
    sys.path.append(PATH_COMMON)#ejecuta lo explicado en path common

    X_train = pd.read_csv('/opt/airflow/dags/data/input/xtrain.csv')
    X_test = pd.read_csv('/opt/airflow/dags/data/input/xtest.csv')
    Y_train = pd.read_csv('/opt/airflow/dags/data/input/ytrain.csv')
    Y_test = pd.read_csv('/opt/airflow/dags/data/input/ytest.csv')
    features = pd.read_csv('/opt/airflow/dags/data/input/selected_features.csv')

    common_columns = np.intersect1d(features['0'].values, X_train.columns)

    X_train = X_train[common_columns]

    common_columns = np.intersect1d(features['0'].values, X_test.columns)

    X_test = X_test[common_columns]

    Rdm_frst=RandomForestRegressor(n_estimators=100,random_state=123)
    Rdm_frst.fit(X_train,y=Y_train)

    joblib.dump(Rdm_frst,'/opt/airflow/dags/data/model/RandomForest.joblib')

    print('Modelo Guardado')

@task.virtualenv(#task que tiene un ambiente virtual, como en nuestro pc. Si se usa llibreria tal cual, va a fallar porque faltarian las librerias necesarias para correr el codigo. Importante se usa virtual env porque se requieren instalar librerias especificas que no estan en la biblioteca principal de python
    task_id='predictionmodel',#Se define ID
    requirements=[
        'pandas','scikit-learn','joblib','numpy'
    ],#parametros requirements se define las librerias que necesitan instalarse en el ambiente virtual, además de las que tiene airflow por defecto
    system_site_packages=False
)
def task_prediction():#Dentro del ambiente trabajo tal cual un codigo python
    import pandas as pd
    import sys
    from datetime import datetime
    from joblib import load
    import numpy as np

    PATH_COMMON='../'#lo que hace es ir hacia afuera un escalon para poder ir a la carpeta common
    sys.path.append(PATH_COMMON)#ejecuta lo explicado en path common

    regressor=load('/opt/airflow/dags/data/model/RandomForest.joblib')

    X_train = pd.read_csv('/opt/airflow/dags/data/input/xtrain.csv')
    X_test = pd.read_csv('/opt/airflow/dags/data/input/xtest.csv')
    features = pd.read_csv('/opt/airflow/dags/data/input/selected_features.csv')

    common_columns = np.intersect1d(features['0'].values, X_train.columns)

    X_train = X_train[common_columns]

    common_columns = np.intersect1d(features['0'].values, X_test.columns)

    X_test = X_test[common_columns]

    prediction=regressor.predict(X_test)
    prediction=pd.DataFrame(prediction,columns=['prediction'])

    prediction.to_csv('/opt/airflow/dags/data/output/prediction.csv')

    print('prediction Guardado')

@task.virtualenv(#task que tiene un ambiente virtual, como en nuestro pc. Si se usa llibreria tal cual, va a fallar porque faltarian las librerias necesarias para correr el codigo. Importante se usa virtual env porque se requieren instalar librerias especificas que no estan en la biblioteca principal de python
    task_id='monitoringmodel',#Se define ID
    requirements=[
        'pandas','scikit-learn','joblib','numpy'
    ],#parametros requirements se define las librerias que necesitan instalarse en el ambiente virtual, además de las que tiene airflow por defecto
    system_site_packages=False
)
def task_monitoringmodel():#Dentro del ambiente trabajo tal cual un codigo python
    import pandas as pd
    import sys
    from datetime import datetime
    from joblib import load
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    import os

    PATH_COMMON='../'#lo que hace es ir hacia afuera un escalon para poder ir a la carpeta common
    sys.path.append(PATH_COMMON)#ejecuta lo explicado en path common

    regressor=load('/opt/airflow/dags/data/model/RandomForest.joblib')

    y_test = pd.read_csv('/opt/airflow/dags/data/input/ytest.csv')
    X_test = pd.read_csv('/opt/airflow/dags/data/input/xtest.csv')
    features = pd.read_csv('/opt/airflow/dags/data/input/selected_features.csv')

    common_columns = np.intersect1d(features['0'].values, X_test.columns)

    X_test = X_test[common_columns]

    prediction=regressor.predict(X_test)
    test_mse=mean_squared_error(prediction,y_test)
    test_rmse=mean_squared_error(prediction,y_test,squared=False)
    test_r2=r2_score(prediction,y_test)
    avg_car_price=np.mean(prediction)

    importance = pd.Series(np.abs(regressor.feature_importances_))
    importance.index =common_columns
    importance.sort_values(inplace=True, ascending=False)

    processing_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    
    metrics_results = pd.DataFrame({
        'Proccesing_date': [processing_date],
        'test_MSE': [test_mse],
        'test_RMSE': [test_rmse],
        'test_R2': [test_r2],
        'Average_car_price': [avg_car_price]
    })

    importance=pd.DataFrame({
        'Process_date':[processing_date]*len(common_columns),
        'Feature':importance.index,
        'Importance':importance.values
    })
    matrics_path='/opt/airflow/dags/data/output/metrics_result.csv'
    if os.path.isfile(matrics_path):
        metrics_results.to_csv(matrics_path, mode='a',header=False,index=False)
    else:
        metrics_results.to_csv(matrics_path, mode='w',index=False)
    feature_path='/opt/airflow/dags/data/output/feature_importance.csv'
    if os.path.isfile(feature_path):
        importance.to_csv(feature_path, mode='a',header=False,index=False)
    else:
        importance.to_csv(feature_path, mode='w',index=False)

    print('Metricas Guardadas')