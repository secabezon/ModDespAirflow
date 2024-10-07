from airflow.decorators import task
from datetime import datetime, timedelta


@task.virtualenv(
    task_id='trainingmodel',#Se define ID
    requirements=[
        'pandas','scikit-learn','joblib','numpy'
    ],
    system_site_packages=False
)
def task_train():
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
    df['running']=df['running'].apply(lambda x: float(x.replace('km','')) if x[-2:]=='km' else float(x.replace('miles',''))*1.609344)
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

    y_train=np.log(y_train)
    y_test=np.log(y_test)
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
    pd.Series(selected_feats).to_csv('/opt/airflow/dags/data/output/selected_features.csv', index=False)

    Rdm_frst=RandomForestRegressor(n_estimators=100,random_state=123)
    Rdm_frst.fit(X_train,y=y_train)
    joblib.dump(Rdm_frst,'/opt/airflow/dags/data/model/RandomForest.joblib')

    print('Modelo Entrenado y Guardado')

@task.virtualenv(
    task_id='preprocessvalues',
    requirements=[
        'pandas','scikit-learn','joblib','numpy'
    ],
    system_site_packages=False
)
def task_preprocess():
    import pandas as pd
    import sys
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    PATH_COMMON='../'
    sys.path.append(PATH_COMMON)
    df = pd.read_csv('/opt/airflow/dags/data/input/test.csv')
    df['running']=df['running'].apply(lambda x: float(x.replace('km','')) if x[-2:]=='km' else float(x.replace('miles',''))*1.609344)
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

    features = pd.read_csv('/opt/airflow/dags/data/output/selected_features.csv')

    common_columns = np.intersect1d(features['0'].values, df.columns)

    X_test = df[common_columns]

    print('Datos Preprocesados')
    return X_test

@task.virtualenv(
    task_id='predictionmodel',
    requirements=[
        'pandas','scikit-learn','joblib','numpy'
    ],
    system_site_packages=False
)
def task_prediction(X_test):
    import pandas as pd
    import sys
    import joblib
    import numpy as np

    PATH_COMMON='../'
    sys.path.append(PATH_COMMON)

    regressor=joblib.load('/opt/airflow/dags/data/model/RandomForest.joblib')


    prediction=regressor.predict(X_test)
    prediction=np.exp(prediction)
    prediction=pd.DataFrame(prediction,columns=['prediction'])

    prediction.to_csv('/opt/airflow/dags/data/output/prediction.csv')

    print('prediction Guardada')