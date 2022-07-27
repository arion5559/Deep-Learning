import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ann_visualizer.visualize import ann_viz

dataset = pd.read_excel("Folds5x2_pp.xlsx")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=8, activation="relu"))
ann.add(tf.keras.layers.Dense(units=8, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=4, activation="relu"))
ann.add(tf.keras.layers.Dense(units=4, activation="relu"))

ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer="adam", loss="mean_squared_error")

history = ann.fit(x_train, y_train, batch_size=32, epochs=50)
print(history.history)

results = ann.evaluate(x_test, y_test)

print(results)

y_pred = ann.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

ann_viz(ann)
