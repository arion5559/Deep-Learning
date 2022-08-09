from minisom import MiniSom
import numpy as np
import pandas as pd
import preprocessing
from joblib import dump, load


def train_som(path, save_path):
    x, y = preprocessing.import_values(path)
    x = preprocessing.scale_minmax(x, fit=True)
    som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.3)
    som.random_weights_init(x)
    som.train_random(data=x, num_iteration=300)
    dump(som, save_path)


def load_som(path):
    return load(path)


def find_frauds(path, x, printing=True):
    x = preprocessing.scale_minmax(x)
    som = load_som(path)
    mappings = som.win_map(x)
    top_cells = 4
    helper = np.concatenate((som.distance_map().reshape(100, 1), np.arange(100).reshape(100, 1)), axis=1)
    helper = helper[helper[:, 0].argsort()][::-1]
    idx = helper[:top_cells, 1]
    result_map = []
    for i in range(10):
        for j in range(10):
            if (i * 10 + j) in idx:
                if len(result_map) == 0:
                    result_map = mappings[(i, j)]
                else:
                    result_map = np.concatenate((result_map, mappings[(i, j)]), axis=0)
    frauds = preprocessing.inverse_scale(result_map, "minMax_scaler.bin")
    if printing:
        print("Fraud customer ID's")
        for i in frauds[:, 0]:
            print(int(i))
    return frauds

