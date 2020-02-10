import xgboost as xgb
import mlflow

from hyperopt import fmin, tpe, hp
from model.eval import kfold_eval


def optimise(train: xgb.DMatrix, max_evals: int):
    # build the objective function for optimisation
    def objective(params):
        result = kfold_eval(
            train,
            **params
        )

        # log
        experiment = mlflow.get_experiment_by_name('hyperparameter optimisation')
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment('hyperparameter optimisation')

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_metrics(result)
            mlflow.log_params(params)

        # convert metrics to loss
        loss = 1 - result['test-auprc']

        return loss

    # define the search space
    space = {
        'max_depth': hp.quniform('max_depth', 1, 100, 10),
        'num_boost_round': hp.quniform('num_boost_round', 10, 100, 10)
    }

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals
    )

    return best



