from minisom import MiniSom
import numpy as np
import pandas as pd
import preprocessing
from joblib import dump, load
from pylab import bone, pcolor, colorbar, plot, show


def train_som(path, save_path):
    x, y = preprocessing.import_values(path)
    x = preprocessing.scale(x, fit=True)
    som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.3)
    som.random_weights_init(x)
    som.train_random(data=x, num_iteration=300)
    dump(som, save_path)


def visualize_som(path, x, y):
    x = preprocessing.scale(x)
    som = load_som(path)
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    markers = ["o", "s"]
    colors = ["r", "g"]
    for i, x in enumerate(x):
        w = som.winner(x)
        plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor=colors[y[i]],
             markerfacecolor="None",
             markersize=10,
             markeredgewidth=2)
    show()


def load_som(path):
    return load(path)


def find_frauds(path, x):
    x = preprocessing.scale(x)
    som = load_som(path)
    mappings = som.win_map(x)
    frauds = np.concatenate((mappings[(1, 1)], mappings[(4, 1)]), axis=0)
    frauds = preprocessing.inverse_scale(frauds)
    print("Fraud customer ID's")
    for i in frauds[:, 0]:
        print(int(i))

