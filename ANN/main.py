import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping


def see_med_acc_ann():
    classifier = KerasClassifier(build_fn=build_ann, batch_size=32, epochs=200)
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=3)
    mean = accuracies.mean()
    variance = accuracies.std()


def adjust_ann(x_train, y_train):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.25)

    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    sess = tf.compat.v1.Session(config=config)

    tf.compat.v1.keras.backend.set_session(session=sess)

    classifier = KerasClassifier(build_fn=build_ann, callbacks=EarlyStopping(patience=3, monitor='loss'))
    parameters = {
        "batch_size": [10, 25, 32, 64],
        "epochs": [50, 100],
        "optimizer": ["adam", "rmsprop"],
        "neurons1": [6, 10, 14],
        "neurons2": [12, 16, 20],
        "neurons3": [24, 28, 32],
    }
    grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy",
                               cv=10)
    grid_search = grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(best_params)
    print(best_accuracy)


def build_ann(neurons1, neurons2, neurons3, optimizer):
    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.Dense(units=neurons1, activation="relu"))

    ann.add(Dropout(rate=0.1))

    ann.add(tf.keras.layers.Dense(units=neurons1, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=neurons2, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=neurons2, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=neurons3, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=neurons3, activation="relu"))

    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    ann.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return ann


def train_ann(ann, x_train, y_train):

    history = ann.fit(x_train, y_train, batch_size=32, epochs=50)

    ann_model_json = ann.to_json()

    with open(r"ann_model.json", "w") as json_file:
        json_file.write(ann_model_json)

    ann.save_weights(r"ann_model.h5")

    tf.keras.utils.plot_model(ann, show_shapes=True, show_layer_names=True)

    return ann


def load_ann():
    json_file = open(r"ann_model.json", "r")
    loaded_nn_json = json_file.read()
    json_file.close()
    loaded_nn = model_from_json(loaded_nn_json)
    loaded_nn.load_weights(r"ann_model.h5")
    return loaded_nn


dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.025)
print(tf.config.list_physical_devices)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = 0.025

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# ann = build_ann(6, 12, 24, optimizer="adam")
# see_med_acc_ann()
adjust_ann(x_train, y_train)
# ann = train_ann(ann, x_train, y_train)
ann = load_ann()

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print(accuracy_score(y_test, y_pred))
