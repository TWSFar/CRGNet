"""
"""
import joblib
import numpy as np
import lightgbm as lgb
from utils import get_log, get_dataset
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (mean_absolute_error as MAE,
                             mean_squared_error as MSE,
                             explained_variance_score as EVS,
                             r2_score as R2)
hyp = {
    'train_dataset1': 'density_tools/statistic_results/DOTA_train1.csv',
    'test_dataset1': 'density_tools/statistic_results/DOTA_val1.csv',
    'train_dataset2': 'density_tools/statistic_results/DOTA_train2.csv',
    'test_dataset2': 'density_tools/statistic_results/DOTA_train2.csv',
    'aim': 100}


def main():
    log = get_log()
    # Load datasets
    feature_train1, target_train1 = get_dataset(hyp['train_dataset1'], hyp['aim'], transform=False, scaler=False)
    feature_test1, target_test1 = get_dataset(hyp['test_dataset1'], hyp['aim'], transform=False, scaler=False)
    feature_train2, target_train2 = get_dataset(hyp['train_dataset2'], hyp['aim'], transform=False, scaler=False)
    feature_test2, target_test2 = get_dataset(hyp['test_dataset2'], hyp['aim'], transform=False, scaler=False)
    feature_train_val = feature_train1 + feature_test1 + feature_train2 + feature_test2
    target_train_val = target_train1 + target_test1 + target_train2 + target_test2
    feature_train_val2 = feature_train2 + feature_test2
    target_train_val2 = target_train2 + target_test2

    # Define model
    param_grid = {
        'learning_rate': 0.1,
        'num_boost_round': 150,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'max_depth': 6,
        'num_leaves': 45,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        # 'reg_alpha': 0.04,
        'reg_lambda': 0.12,
        'metric': 'rmse'}

    # Model
    model = lgb.LGBMRegressor(**param_grid)
    # model = tree.ExtraTreeRegressor()

    # Train
    model.fit(feature_train_val, target_train_val)

    # Predict
    predict_results = model.predict(feature_test2)
    log.info(((predict_results > 0) * (predict_results < 0.6)).sum())
    for metric in [R2, MAE, MSE, EVS]:
        score = metric(target_test2, predict_results)
        log.info(metric.__name__+': '+str(score))

    # Save
    joblib.dump(model, 'gbm_dota_{}.pkl'.format(hyp['aim']))


if __name__ == '__main__':
    main()
