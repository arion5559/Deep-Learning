import numpy as np
import tensorflow as tf
import tensorflow.core.protobuf.config_pb2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from ann_visualizer.visualize import ann_viz
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import plot_model

MEMORY_PROCESS = 0.2
SIZE = 300


def optimize_cnn(x_train, y_train):
    classifier = KerasClassifier(build_fn=build_cnn, callbacks=EarlyStopping(patience=6, monitor="val_loss"))

    parameters = {
        "dense_neurons": [32, 64, 128],
        "optimizer": ["adam", "rmsprop"],
        "input_shape": [[SIZE, SIZE, 3]],
        "number_of_conv2D": [2, 3, 4],
        "number_of_dense": [2, 3, 4],
        "epochs": [250]
    }
    grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10)
    grid_search.fit(train_set, validation_data=test_set, )
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    print(best_params)
    print(best_accuracy)
    return best_params


def build_cnn(dense_neurons, optimizer, input_shape):
    cnn = tf.keras.models.Sequential()

    cnn.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape))
    cnn.add(MaxPool2D(pool_size=2, strides=None))

    cnn.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    cnn.add(MaxPool2D(pool_size=2, strides=None))

    cnn.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
    cnn.add(MaxPool2D(pool_size=2, strides=None))

    cnn.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
    cnn.add(MaxPool2D(pool_size=2, strides=None))

    cnn.add(Flatten())

    cnn.add(Dense(units=dense_neurons, activation=tf.nn.relu))

    cnn.add(Dropout(rate=0.1))

    cnn.add(Dense(units=dense_neurons, activation=tf.nn.relu))
    cnn.add(Dense(units=dense_neurons, activation=tf.nn.relu))

    cnn.add(Dense(units=1, activation=tf.nn.sigmoid))

    cnn.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return cnn


def train_cnn(cnn, train_set, test_set, epochs):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_PROCESS)

    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.per_process_gpu_memory_fraction = MEMORY_PROCESS

    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    cnn.fit(train_set, validation_data=test_set, epochs=epochs,
            callbacks=EarlyStopping(patience=6, monitor="val_loss"))

    plot_model(cnn, show_shapes=True, show_layer_names=True, expand_nested=True)

    return cnn


def convert_to_test(path):
    test_image = image.load_img(path, target_size=(SIZE, SIZE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image


def print_result(result):
    if result[0][0] == 1:
        prediction = "It's a dog"
    else:
        prediction = "It's a cat"
    return prediction


def save_model(model, name):
    ann_model_json = model.to_json()

    with open(f"{name}.json", "w") as json_file:
        json_file.write(ann_model_json)

    model.save_weights(f"{name}.h5")


def load_model(name):
    json_file = open(f"{name}.json", "r")
    loaded_nn_json = json_file.read()
    json_file.close()
    loaded_nn = model_from_json(loaded_nn_json)
    loaded_nn.load_weights(f"{name}.h5")
    return loaded_nn


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_PROCESS)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.per_process_gpu_memory_fraction = MEMORY_PROCESS

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.3, zoom_range=0.3, horizontal_flip=True)

train_set = train_datagen.flow_from_directory("dataset/training_set", target_size=(SIZE, SIZE),
                                              batch_size=32, class_mode="binary")

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = train_datagen.flow_from_directory("dataset/test_set", target_size=(SIZE, SIZE),
                                             batch_size=32, class_mode="binary")

cnn = build_cnn(dense_neurons=256, optimizer="adam", input_shape=[SIZE, SIZE, 3])

cnn = train_cnn(cnn=cnn, train_set=train_set, test_set=test_set, epochs=250)

save_model(cnn, "cnn_model")

cnn = load_model("cnn_model")

test_image_1 = convert_to_test("dataset/single_prediction/cat_or_dog_1.jpg")
test_image_2 = convert_to_test("dataset/single_prediction/cat_or_dog_2.jpg")
test_cuca = convert_to_test("dataset/single_prediction/cat_or_dog_cuca.jpeg")
test_tsuki = convert_to_test("dataset/single_prediction/cat_or_dog_tsuki.jpeg")

result_1 = cnn.predict(test_image_1)
result_2 = cnn.predict(test_image_2)
result_cuca = cnn.predict(test_cuca)
result_tsuki = cnn.predict(test_tsuki)

print(train_set.class_indices)

print(f"Test 1: {print_result(result_1)}")
print(f"Test 2: {print_result(result_2)}")
print(f"Test cuca: {print_result(result_cuca)}")
print(f"Test tsuki: {print_result(result_tsuki)}")
