import boltzman_machine_training
import preprocessing

training_set = preprocessing.import_dataset("ml-100k/u1.base", sep="\t", header=None)
test_set = preprocessing.import_dataset("ml-100k/u1.test", sep="\t", header=None)
training_set, test_set, nb_id = preprocessing.import_information_given_sets(training_set, test_set)

# boltzman_machine_training.train_machine(training_set, nb_id)

boltzman_machine_training.test_machine(training_set, test_set, nb_id)
