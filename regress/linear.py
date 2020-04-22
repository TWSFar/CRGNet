"""
    r2_score: 0.6606250386866317
    mean_absolute_error: 5.919616106926392
    mean_squared_error: 137.72707634715482
    explained_variance_score: 0.666945130281111
ard

"""
import joblib
import numpy as np
from utils import get_log, get_dataset
from sklearn.linear_model import (
    LinearRegression, ARDRegression, BayesianRidge, TheilSenRegressor, RANSACRegressor)
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
    log = get_log()
    # Load datasets
    feature_train, target_train = get_dataset(hyp['train_dataset'], hyp['aim'])
    feature_test, target_test = get_dataset(hyp['test_dataset'], hyp['aim'])

    # Define model
    model = ARDRegression()

    # Train
    model.fit(feature_train, target_train)

    # joblib.dump(model, 'mlp.pkl')
    # model = joblib.load(hyp['resume'])

    # Predict
    predict_results = model.predict(feature_test)

    for metric in [R2, MAE, MSE, EVS]:
        score = metric(target_test, predict_results)
        log.info(metric.__name__+': '+str(score))


if __name__ == '__main__':
    main()
