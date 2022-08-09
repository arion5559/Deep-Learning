import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump, load


def import_values(path):
    dataset = pd.read_csv(path)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return x, y


def scale_minmax(x, fit=False):
    if fit:
        sc = MinMaxScaler(feature_range=(0, 1), copy=True)
        x_scaled = sc.fit_transform(x)
        dump(sc, "minMax_scaler.bin", compress=True)
    else:
        sc = load("minMax_scaler.bin")
        x_scaled = sc.transform(x)
    return x_scaled


def inverse_scale(x, scaler_path):
    sc = load(scaler_path)
    return sc.inverse_transform(x)


def scale_standard(x, fit=False):
    if fit:
        sc = StandardScaler()
        x_scaled = sc.fit_transform(x)
        dump(sc, "std_scaler.bin", compress=True)
    else:
        sc = load("std_scaler.bin")
        x_scaled = sc.transform(x)
    return x_scaled
