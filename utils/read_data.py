from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb


def load_data() -> tuple:
    # load scikit-learn built-in data
    data = load_breast_cancer()

    # split train, valid, test
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.3,
        random_state=1

    )

    # convert to xgboost DMatrix
    train = xgb.DMatrix(
        data=X_train,
        label=y_train,
        feature_names=data.feature_names
    )

    test = xgb.DMatrix(
        data=X_test,
        label=y_test,
        feature_names=data.feature_names
    )

    return train, test
