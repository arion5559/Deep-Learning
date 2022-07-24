import training
import matplotlib.pyplot as plt
import predict
import preprocessing
import tensorflow as tf
from ann_visualizer.visualize import ann_viz


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)

config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# rnn = training.train_rnn("Google_Stock_Price_Train.csv")

# training.save_rnn(rnn)

rnn = training.load_rnn()

real_stock_prices = preprocessing.import_set("Google_Stock_Price_Test.csv")
y_pred = predict.make_prediction("Google_Stock_Price_Train.csv", "Google_Stock_Price_Test.csv")

plt.plot(real_stock_prices, color="red", label="real")
plt.plot(y_pred, color="blue", label="predicted")
plt.title("Google stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()
