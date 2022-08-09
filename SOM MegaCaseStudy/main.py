import preprocessing
import training
import ann

training.train_som("Credit_Card_Applications.csv", "Self_organizing_map.bin")

x, y = preprocessing.import_values("Credit_Card_Applications.csv")

# training.visualize_som(r"Self_organizing_map.bin", x=x, y=y)

training.find_frauds(r"Self_organizing_map.bin", x)

ann.build_ann()

ann.train_ann("Credit_Card_Applications.csv")

y_pred = ann.make_prediction(x)

print(y_pred)
