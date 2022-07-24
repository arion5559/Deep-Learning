import preprocessing
import pandas as pd
import numpy as np
import training
import keras


def make_prediction(train_set_path, test_set_path):
    rnn = training.load_rnn()
    train_set = pd.read_csv(train_set_path)
    test_set = pd.read_csv(test_set_path)
    dataset_total = pd.concat((train_set["Open"], test_set["Open"]), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_set) - 80:].values
    inputs = inputs.reshape(-1, 1)
    inputs = preprocessing.scale_set(inputs)
    x_test, y_test = preprocessing.create_data_structure(inputs)
    x_test = preprocessing.reshape_set(x_test)
    predicted_stock_price = rnn.predict(x_test)
    y_pred = preprocessing.inverse_transformation(predicted_stock_price)
    return y_pred


def predict_future(num_of_pred, train_set_path):
    rnn = training.load_rnn()
    train_set = preprocessing.import_set(train_set_path)
    inputs = train_set[len(train_set) - 80:].values
    inputs.reshape(-1, 1)
    inputs = preprocessing.scale_set(inputs)
    x_test = preprocessing.create_data_structure(inputs)
    x_test = np.array(x_test)
    x_test = preprocessing.reshape_set(x_test)
    predictions = np.array()
    for i in range(num_of_pred + 1):
        predictions = np.append(predictions, rnn.predict(x_test))
        x_test = np.append(x_test, predictions[-1])



