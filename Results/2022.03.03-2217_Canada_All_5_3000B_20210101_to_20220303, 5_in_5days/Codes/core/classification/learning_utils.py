import numpy as np


def encode_categorical_labels_to_numerical(labels):
    unique_labels = np.unique(labels)
    unique_targets = np.arange(0, len(unique_labels))
    labels_in_dict = dict(zip(unique_labels, unique_targets))

    numerical_labels = [labels_in_dict.get(x) for x in labels]

    labels_list = labels_in_dict.keys()
    class_list = labels_in_dict.values()

    numerical_labels = np.array(numerical_labels)  # y

    return numerical_labels, labels_list, class_list
