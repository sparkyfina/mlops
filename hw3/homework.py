import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect import get_run_logger 

import datetime
from datetime import datetime
from datetime import date

import pickle

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    
    logger = get_run_logger()
    
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")

    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger = get_run_logger()
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)

    logger = get_run_logger()
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_path(date_in):
    #str = './data/fhv_tripdata_2021-01.parquet', 
    #val_path: str = './data/fhv_tripdata_2021-02.parquet')

    #If date is None, use the current day. 
    # Use the data from 2 months back as the training data and the data from the previous month as validation data.
    if (date_in == None):
        #process_date = datetime.date.today()
        process_date = date.today()
    else:
        process_date = datetime.strptime(date_in, "%Y-%m-%d")
    
    train = "./data/fhv_tripdata_2021-%02d.parquet" % (process_date.month -2)
    val = "./data/fhv_tripdata_2021-%02d.parquet" % (process_date.month - 1)    
    logger = get_run_logger()
    logger.info(f"Training data path = {train}")
    logger.info(f"Validation data path = {val}")

    return train, val


from prefect.deployments import DeploymentSpec
#from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

@flow(task_runner=SequentialTaskRunner())
#def main(train_path: str = './data/fhv_tripdata_2021-01.parquet', 
#           val_path: str = './data/fhv_tripdata_2021-02.parquet'):
def main(date=None):
    
    train_path, val_path = get_path(date).result() 
    #str = './data/fhv_tripdata_2021-01.parquet', 
    #val_path: str = './data/fhv_tripdata_2021-02.parquet'):

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

  
    model_name = f"models/model-{date}.bin"
    dv_name =  f"models/dv-{date}.b"

    with open(model_name, 'wb') as f_out:
        pickle.dump((lr), f_out)

    with open(dv_name, "wb") as f_out:
        pickle.dump(dv, f_out)
    

#no need to call main from scheduled deployment
#main()
#main("2021-03-15")
#main(date="2021-08-15")

DeploymentSpec(
    flow=main,
    #flow_location="./homework.py",
    name="hw3_model_training",
    schedule=CronSchedule(
        #cron="5 * * * *",  #test version every 5 minutes
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    #tells to run locally and not in container or k8
    tags=["ml-cron"]
)
