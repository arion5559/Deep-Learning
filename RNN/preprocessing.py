import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load


def import_set(set_path):
    dataset = pd.read_csv(set_path)
    set = dataset.iloc[:, 1:2].values
    return set


def scale_set(set, fit=False):
    if fit:
        sc = MinMaxScaler(feature_range=(0, 1), copy=True)
        set_scaled = sc.fit_transform(set)
        dump(sc, "std_scaler.bin", compress=True)
    else:
        sc = load("std_scaler.bin")
        set_scaled = sc.transform(set)
    return set_scaled


def create_data_structure(set):
    x = []
    y = []
    for i in range(80, len(set)):
        x.append(set[i-80:i, 0])
        y.append(set[i, 0])
    x_train, y_train = np.array(x), np.array(y)
    return x_train, y_train


def reshape_set(x):
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x


def inverse_transformation(set):
    sc = load("std_scaler.bin")
    return sc.inverse_transform(set)