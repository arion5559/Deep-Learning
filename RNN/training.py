import preprocessing
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def adjust_ann():
    set = preprocessing.import_set("Google_Stock_Price_Train.csv")
    set_scaled = preprocessing.scale_set(set, fit=True)
    x_train, y_train = preprocessing.create_data_structure(set_scaled)
    x_train = preprocessing.reshape_set(x_train)
    classifier = KerasClassifier(build_fn=create_rnn)
    neurons = []
    for i in range(5, 6):
        neurons.append(2**i)

    print(neurons)
    parameters = {
        "neurons": neurons,
        "epochs": [100]
    }
    grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="neg_mean_squared_error",
                               cv=10)
    grid_search = grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(best_params)
    print(best_accuracy)


def create_rnn(shape=(80, 1), neurons=128):
    rnn = Sequential()
    rnn.add(LSTM(units=neurons, return_sequences=True, input_shape=shape))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=neurons, return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=neurons, return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=neurons, return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=neurons))
    rnn.add(Dropout(0.2))
    rnn.add(Dense(units=1))
    rnn.compile(optimizer="adam", loss="mean_squared_error")
    return rnn


def train_rnn(set_path, **kwargs):
    set = preprocessing.import_set(set_path)
    set_scaled = preprocessing.scale_set(set, fit=True)
    x_train, y_train = preprocessing.create_data_structure(set_scaled)
    x_train = preprocessing.reshape_set(x_train)
    regressor = create_rnn((x_train.shape[1], 1), **kwargs)
    regressor.fit(x_train, y_train, epochs=200, batch_size=40)
    return regressor


def save_rnn(rnn):
    rnn_model_json = rnn.to_json()
    with open(r"rnn_model.json", "w") as json_file:
        json_file.write(rnn_model_json)

    rnn.save_weights(r"rnn_model.h5")
    print("Model saved")


def load_rnn():
    json_file = open(r"rnn_model.json", "r")
    loaded_nn_json = json_file.read()
    json_file.close()
    loaded_nn = model_from_json(loaded_nn_json)
    loaded_nn.load_weights(r"rnn_model.h5")
    return loaded_nn
