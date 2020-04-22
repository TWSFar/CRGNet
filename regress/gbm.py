"""
"""
import joblib
import numpy as np
import lightgbm as lgb
from utils import get_log, get_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (mean_absolute_error as MAE,
                             mean_squared_error as MSE,
                             explained_variance_score as EVS,
                             r2_score as R2)
hyp = {
    'train_dataset': 'regress/train_dataset.csv',
    'test_dataset': 'regress/val_dataset.csv',
    'resume': 'mlp.pkl',
    'aim': 0.0066}


def main():
    ls = [0, 1, 2, 3]
    log = get_log()
    # Load datasets
    feature_train, target_train = get_dataset(hyp['train_dataset'], hyp['aim'], transform=True, scaler=False)
    feature_test, target_test = get_dataset(hyp['test_dataset'], hyp['aim'], transform=True, scaler=False)
    lgb_train = lgb.Dataset(feature_train[:, ls], target_train)
    lgb_eval = lgb.Dataset(feature_test[:, ls], target_test, reference=lgb_train)

    # Define model
    param_grid = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0}

    # Train
    gbm = lgb.train(param_grid,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=lgb_eval)
                    # early_stopping_rounds=5)

    # save model to file
    gbm.save_model('model.txt')

    # Predict
    predict_results = gbm.predict(feature_test[:, ls], num_iteration=gbm.best_iteration)
    for metric in [R2, MAE, MSE, EVS]:
        score = metric(target_test, predict_results)
        log.info(metric.__name__+': '+str(score))


if __name__ == '__main__':
    main()
