
import joblib
import numpy as np
from utils import get_log, get_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
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
    feature_train, target_train = get_dataset(hyp['train_dataset'], hyp['aim'], transform=False)
    feature_test, target_test = get_dataset(hyp['test_dataset'], hyp['aim'], transform=False)

    # Define model
    pls = PLSRegression()
    param_grid = {'n_components': [1, 2, 3, 4],
                  'scale': [True, False]}
    model = GridSearchCV(pls, param_grid=param_grid, scoring='explained_variance', cv=5)

    # Train
    model.fit(feature_train, target_train)

    # joblib.dump(model, 'mlp.pkl')
    # model = joblib.load(hyp['resume'])

    log.info(model.best_params_)
    log.info(model.best_score_)
    # Predict
    predict_results = model.predict(feature_test)

    for metric in [R2, MAE, MSE, EVS]:
        score = metric(target_test, predict_results)
        log.info(metric.__name__+': '+str(score))


if __name__ == '__main__':
    main()
