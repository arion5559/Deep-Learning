import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def import_dataset(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def prepare_set(set_prepare):
    return np.array(set_prepare, dtype="int")


def get_max(set1, set2, column):
    return int(max(max(set1[:, column]), max(set2[:, column])))


def convert(data, max_range, list_range):
    new_data = []
    for id_1 in range(1, max_range + 1):
        id_2 = data[:, 1][data[:, 0] == id_1]
        id_3 = data[:, 2][data[:, 0] == id_1]
        data_list = np.zeros(list_range)
        data_list[id_2 - 1] = id_3
        new_data.append(data_list)
    return new_data


def import_information_given_sets(training_set, test_set):
    training_set = prepare_set(training_set)
    test_set = prepare_set(test_set)
    nb_users = get_max(training_set, test_set, 0)
    nb_movies = get_max(training_set, test_set, 1)
    training_set = convert(training_set, nb_users, nb_movies)
    test_set = convert(test_set, nb_users, nb_movies)
    training_set = torch.FloatTensor(training_set)
    test_set = torch.FloatTensor(test_set)
    return training_set, test_set, nb_users, nb_movies


def part_data(dataset):
    return train_test_split(dataset, test_size=0.2)


def import_information_given_path(path, **kwargs):
    training_set, test_set = part_data(import_dataset(path, **kwargs))
    nb_users = get_max(training_set, test_set, 0)
    nb_movies = get_max(training_set, test_set, 1)
    training_set = convert(training_set, nb_users, nb_movies)
    test_set = convert(test_set, nb_users, nb_movies)
    training_set = torch.FloatTensor(training_set)
    test_set = torch.FloatTensor(test_set)
    training_set = convert_valorations(training_set)
    test_set = convert_valorations(test_set)
    return training_set, test_set
