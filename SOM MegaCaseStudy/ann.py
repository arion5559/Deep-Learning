import tensorflow as tf
import numpy as np
import pandas as pd
import keras

import preprocessing
import training
from keras.models import model_from_json


def build_ann():
    ann = keras.models.Sequential()
    ann.add(keras.layers.Dense(units=6, activation="relu"))
    ann.add(keras.layers.Dense(units=6, activation="relu"))
    ann.add(keras.layers.Dense(units=12, activation="relu"))
    ann.add(keras.layers.Dense(units=12, activation="relu"))
    ann.add(keras.layers.Dense(units=24, activation="relu"))
    ann.add(keras.layers.Dense(units=24, activation="relu"))
    ann.add(keras.layers.Dense(units=1, activation="sigmoid"))
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    ann_model_json = ann.to_json()
    with open(r"ann_model.json", "w") as json_file:
        json_file.write(ann_model_json)

    print("Model saved")


def train_ann(file_path):
    dataset = pd.read_csv(file_path)
    x = dataset.iloc[:, 1:].values
    y = get_dependent_variable(dataset)
    x = preprocessing.scale_standard(x, fit=True)
    ann = import_ann()
    ann.fit(x, y, batch_size=10, epochs=200)
    ann.save_weights(r"ann_model.h5")
    print("Weights saved")


def get_dependent_variable(dataset):
    is_fraud = np.zeros(len(dataset))
    x = dataset.iloc[:, :-1].values
    frauds = training.find_frauds(r"Self_organizing_map.bin", x)
    for i in range(len(dataset)):
        if dataset.iloc[i, 0] in frauds:
            is_fraud[i] = 1
    return is_fraud


def import_ann():
    json_file = open(r"ann_model.json", "r")
    loaded_nn_json = json_file.read()
    json_file.close()
    loaded_nn = model_from_json(loaded_nn_json)
    loaded_nn.load_weights(r"ann_model.h5")
    return loaded_nn


def make_prediction(pred_input):
    ann = import_ann()
    return ann.predict(preprocessing.inverse_scale(pred_input, "std_scaler.bin"))
