import argparse
import os
import pickle

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models-new"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

mr_uri = mlflow.get_registry_uri()
print("Current registry uri: {}".format(mr_uri))
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))




SPACE = {
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    'random_state': 42
}


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        params = space_eval(SPACE, params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # evaluate model on the validation and test sets
        valid_rmse = mean_squared_error(y_valid, rf.predict(X_valid), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


def run(data_path, log_top):

    client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"]
    )
    print("starting train_and_log_model loop")
 
    for run in runs:
        #print("run data parms = ", data_path)
        print("run data parms = ", run.data.params)
        train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the lowest test RMSE
    print("selectin best model to register")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    print("Experiment = ", experiment)
    print("Experiment ID = ", experiment.experiment_id)
    
    # best_run = client.search_runs( ...  )[0]
    best_runs = client.search_runs(
    experiment_ids=experiment.experiment_id,
    filter_string="metrics.test_rmse < 7",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
    )

    print("ready to show best runs")
    for run in best_runs:
        print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['test_rmse']:.4f}")

    print("Best run = ",best_runs[0] , " run id = ", best_runs[0].info.run_id )
    run_id = best_runs[0].info.run_id

    # register the best model
  
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="nyc-random-forest")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote."
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)
