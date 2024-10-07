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
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble._forest import RandomForestRegressor
    import joblib
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.linear_model import Lasso
    from sklearn.feature_selection import SelectFromModel

    PATH_COMMON='../'
    sys.path.append(PATH_COMMON)
    df = pd.read_csv('/opt/airflow/dags/data/input/train.csv')
    df['running']=df['running'].apply(lambda x: float(x.replace('km','')) if x=='km' else float(x.replace('miles',''))*1.609344)
    df=df.drop('wheel',axis=1)

    qual_mappings = {'excellent': 3, 'good':2, 'crashed': 0, 'normal': 1, 'new': 4}
    df['status'] = df['status'].map(qual_mappings)
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[['model', 'motor_type','color','type']])
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(['model', 'motor_type','color','type']))
    df = pd.concat([df[['year','running','status','motor_volume','price']].reset_index(drop=True), encoded_df], axis=1)
    df['motor_type_gas']=df.apply(lambda x: 1 if x['motor_type_petrol and gas']==1 else x['motor_type_gas'],axis=1)
    df['motor_type_petrol']=df.apply(lambda x: 1 if x['motor_type_petrol and gas']==1 else x['motor_type_petrol'],axis=1)
    df=df.drop(['motor_type_petrol and gas'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(['price'], axis=1),
        df['price'],
        test_size=0.1,
        random_state=0,
    )
    scaler = MinMaxScaler()

    scaler.fit(X_train)


    X_train = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns
    )

    sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=0))

    sel_.fit(X_train, y_train)

    selected_feats = X_train.columns[(sel_.get_support())]

    X_train[selected_feats]
    selected_feats.to_csv('/opt/airflow/dags/data/output/selected_features.csv')

    Rdm_frst=RandomForestRegressor(n_estimators=100,random_state=123)
    Rdm_frst.fit(X_train,y=y_train)
    joblib.dump(Rdm_frst,'/opt/airflow/dags/data/model/RandomForest.joblib')

    print('Modelo Entrenado y Guardado')

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