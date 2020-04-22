import csv
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_log():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(message)s')
    logger = logging.getLogger(__name__)
    # f_handler = logging.FileHandler('', mode='a')
    # logger.addHandler(f_handler)
    return logger


def get_dataset(file, aim, transform=True, scaler=True):
    # load dataset
    feature = []
    target = []
    csv_file = csv.reader(open(file))
    for content in csv_file:
        content = list(map(float, content))
        if len(content) != 0:
            feature.append(content[0:-1])
            target.append(content[-1]/aim)

    if transform:
        for i in range(len(feature)):
            feature[i][0] = 1 / feature[i][0]
            feature[i][3] = 1 / feature[i][3]

    # dataset transform
    if scaler:
        scaler = StandardScaler().fit(feature)
        feature = scaler.transform(feature)

    return np.array(feature), target
