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


dataset = "DOTA"
hyp = {
    'train_dataset1': '/home/twsf/work/CRGNet/density_tools/statistic_results/{}_train_1.csv'.format(dataset),
    'test_dataset1': '/home/twsf/work/CRGNet/density_tools/statistic_results/{}_val_1.csv'.format(dataset),
    'train_dataset2': '/home/twsf/work/CRGNet/density_tools/statistic_results/{}_train_2.csv'.format(dataset),
    'test_dataset2': '/home/twsf/work/CRGNet/density_tools/statistic_results/{}_val_2.csv'.format(dataset),
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
    joblib.dump(model, '/home/twsf/work/CRGNet/density_tools/weights/gbm_{}_{}.pkl'.format(dataset.lower(), hyp['aim']))


if __name__ == '__main__':
    main()
