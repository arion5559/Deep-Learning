import preprocessing
import auto_encoder

training_set = preprocessing.import_dataset("ml-100k/u1.base", sep="\t", header=None)
test_set = preprocessing.import_dataset("ml-100k/u1.test", sep="\t", header=None)
training_set, test_set, nb_id, nb_data_per_id = preprocessing.import_information_given_sets(training_set, test_set)

# auto_encoder.train_encoder(training_set, test_set, nb_id, nb_data_per_id)

auto_encoder.test_encoder(training_set, test_set, nb_id, nb_data_per_id)
