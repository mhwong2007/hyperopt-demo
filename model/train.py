import xgboost as xgb


def train_model(train: xgb.DMatrix, **model_params) -> xgb.Booster:
    # create parameters
    params = {
        'booster': model_params.get('booster', 'gbtree'),
        'eta': model_params.get('eta', 0.3),
        'gamma': model_params.get('gamma', 0),
        'max_depth': int(model_params.get('max_depth', 6)),
        'min_child_weight': model_params.get('min_child_weight', 1),
        'max_delta_step': model_params.get('max_delta_step', 0),
        'subsample': model_params.get('subsample', 1),
        'colsample_bytree': model_params.get('colsample_bytree', 1),
        'colsample_bylevel': model_params.get('colsample_bylevel', 1),
        'colsample_bynode': model_params.get('colsample_bynode', 1),
        'lambda': model_params.get('lambda', 1),
        'alpha': model_params.get('alpha', 0),
        'tree_method': model_params.get('tree_method', 'auto'),
        'scale_pos_weight': model_params.get('scale_pos_weight', 1),
        'refresh_leaf': model_params.get('refresh_leaf', 1),
        'process_type': model_params.get('process_type', 'default'),
        'num_parallel_tree': model_params.get('num_parallel_tree', 1)
    }

    num_boost_round = int(model_params.get('num_boost_round', 10))

    bst = xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=num_boost_round,
    )

    return bst
