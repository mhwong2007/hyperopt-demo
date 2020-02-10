import xgboost as xgb
from sklearn import metrics


def calc_tp(y: list, y_pred: list) -> float:
    val = 0.0
    for i in range(len(y)):
        if int(y[i]) == 1 and y_pred[i] > 0.5:
            val += 1.0

    return val


def calc_tn(y: list, y_pred: list) -> float:
    val = 0.0
    for i in range(len(y)):
        if int(y[i]) == 0 and y_pred[i] <= 0.5:
            val += 1.0

    return val


def calc_fp(y: list, y_pred: list) -> float:
    val = 0.0
    for i in range(len(y)):
        if int(y[i]) == 0 and y_pred[i] > 0.5:
            val += 1.0

    return val


def calc_fn(y: list, y_pred: list) -> float:
    val = 0.0
    for i in range(len(y)):
        if int(y[i]) == 1 and y_pred[i] <= 0.5:
            val += 1.0

    return val


def calc_recall(tp, tn, fp, fn):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0


def calc_precision(tp, tn, fp, fn):
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0.0


def calc_fpr(tp, tn, fp, fn):
    try:
        return fp / (fp + tn)
    except ZeroDivisionError:
        return 0.0


def calc_auprc(y: list, y_pred: list) -> float:
    return metrics.average_precision_score(y, y_pred)


def calc_auc(y: list, y_pred: list) -> float:
    _fpr, _tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
    return metrics.auc(_fpr, _tpr)


def kfold_eval(train: xgb.DMatrix, **model_params) -> dict:
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
    nfold = model_params.get('nfold', 5)

    def custom_metrics(y_pred: list, data: xgb.DMatrix) -> list:
        y = data.get_label()
        auprc = calc_auprc(y, y_pred)
        tp = calc_tp(y, y_pred)
        tn = calc_tn(y, y_pred)
        fp = calc_fp(y, y_pred)
        fn = calc_fn(y, y_pred)
        precision = calc_precision(tp, tn, fp, fn)
        recall = calc_recall(tp, tn, fp, fn)
        fpr = calc_fpr(tp, tn, fp, fn)

        return [
            ('precision', precision),
            ('recall', recall),
            ('fpr', fpr),
            ('auprc', auprc)
        ]

    history = xgb.cv(
        params=params,
        dtrain=train,
        num_boost_round=num_boost_round,
        nfold=nfold,
        stratified=True,
        feval=custom_metrics,
        seed=1
    )

    # get last row -> the result of the final boost round
    last_boost_result = history.tail(n=1)
    result = {
        'train-auprc': float(last_boost_result['train-auprc-mean']),
        'train-precision': float(last_boost_result['train-precision-mean']),
        'train-recall': float(last_boost_result['train-recall-mean']),
        'train-fpr': float(last_boost_result['train-fpr-mean']),
        'test-auprc': float(last_boost_result['test-auprc-mean']),
        'test-precision': float(last_boost_result['test-precision-mean']),
        'test-recall': float(last_boost_result['test-recall-mean']),
        'test-fpr': float(last_boost_result['test-fpr-mean']),
    }

    return result
