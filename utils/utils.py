import math
import random


def shuffle_dataset(data):
    '''
    In-place shuffling of the dataset.
    :param graphs: a list of tuples (X_values, Y_values, adjacency_lists, size), each representing a graph
    :param sizes: a list of sizes, one for each graph
    :return:
    '''
    random.shuffle(data)


def inductive_split_dataset(data, train_perc, valid_perc, test_perc, shuffle=False):
    '''
    Needed for inductive settings where I have a dataset of graphs (graph classification).
    Shuffles and splits a dataset in train, validation and test graphs.
    :param data: a list of tuples (X_values, Y_values, adjacency_lists, size), each representing a graph
    :param sizes: a list of sizes, one for each graph
    :param train_perc: the percentage of examples for the training set, from 0.0 to 1.0
    :param valid_perc: the percentage of examples for the validation set, from 0.0 to 1.0
    :param test_perc: the percentage of examples for the test set, from 0.0 to 1.0
    :param shuffle: if True, shuffles the list of graphs before splitting
    :return: a tuple made of 3 datasets: (data_train, data_valid, graphs_tests), which are identical
    in structure to the dataset passed as input
    '''
    # print(train_perc, valid_perc, test_perc, train_perc + valid_perc + test_perc)
    assert train_perc + valid_perc + test_perc >= 0.99

    if shuffle:
        shuffle_dataset(data)

    total_examples = len(data)

    train_dim = int(total_examples*train_perc)  # python converts to the nearest lower integer
    valid_dim = int(total_examples*valid_perc)

    data_train = data[0:train_dim]
    data_valid = data[train_dim:train_dim + valid_dim]
    data_test = data[train_dim + valid_dim:]

    return data_train, data_valid, data_test


def transductive_split_dataset(total_no_nodes, no_classes, Y,
                               no_train_nodes, no_valid_nodes, no_test_nodes, shuffle=False):
    '''
    Needed for transductive settings where I have a graph (node classification).
    Nodes have their features and a target label attached.
    Shuffles and splits a dataset in train, validation and test nodes.
    :param no_test_nodes:
    :param no_valid_nodes:
    :param no_train_nodes:     :param sizes: a list of sizes, one for each graph
    :param shuffle: if True, shuffles the list of graphs before splitting
    :return: a tuple made of 3 datasets: (data_train, data_valid, graphs_tests), which are identical
    in structure to the dataset passed as input
    '''
    assert total_no_nodes >= no_train_nodes + no_valid_nodes + no_test_nodes, "Too many nodes asked"

    # Shuffle should just select indices at random! It MUST NOT shuffle (avoid inconsistencies between X, Y and adj_list
    idx_list = [i for i in range(total_no_nodes)]
    if shuffle:
        random.shuffle(idx_list)

    class_idxs = [[] for _ in range(no_classes)]

    train_idxs = []
    train_class_examples = [0. for _ in range(no_classes)]
    valid_idxs = []
    valid_class_examples = [0. for _ in range(no_classes)]
    test_idxs = []
    test_class_examples = [0. for _ in range(no_classes)]

    for i in range(total_no_nodes):
        # Assign an index to a class
        next_idx = idx_list[i]
        class_idxs[Y[next_idx, 0]].append(next_idx)

    for cl in range(no_classes):
        for cl_idx in class_idxs[cl]:
            if train_class_examples[cl] < no_train_nodes//no_classes:
                train_idxs.append(cl_idx)
                train_class_examples[cl] = train_class_examples[cl] + 1
            elif valid_class_examples[cl] < no_valid_nodes//no_classes:
                valid_idxs.append(cl_idx)
                valid_class_examples[cl] = valid_class_examples[cl] + 1
            elif test_class_examples[cl] < no_test_nodes // no_classes:
                test_idxs.append(cl_idx)
                test_class_examples[cl] = test_class_examples[cl] + 1
            else:
                pass
    '''
    print(train_idxs)
    print(train_class_examples)
    print(valid_idxs)
    print(valid_class_examples)
    print(test_idxs)
    print(test_class_examples)
    '''
    return train_idxs, valid_idxs, test_idxs


def build_folds(folds, data):
    '''
    Returns a list of tuples (train_fold, valid/test fold), with valid/test splits that do not overlap
    :param folds:
    :param data: list of examples
    :return:
    '''

    set_dim = len(data)
    fold_dim = math.floor(set_dim / folds)

    splits = []

    for k in range(folds):

        if k == 0:
            train_split = data[(k + 1) * fold_dim:]
            valid_split = data[:fold_dim]

        elif k == folds - 1:
            train_split = data[0:k * fold_dim]
            valid_split = data[k * fold_dim:]
        else:
            train_split = data[0:k * fold_dim] + data[(k + 1) * fold_dim:]
            valid_split = data[k * fold_dim:(k + 1) * fold_dim]

        splits.append((train_split, valid_split))

        # print(k, train_split, valid_split)

    return splits


def generate_grid(configs):
    '''
    Takes a dictionary of key:list pairs and computes all possible permutations.
    :param configs:
    :return: A dictionary generator
    '''

    keys = configs.keys()
    result = {}

    if configs == {}:
        yield {}
    else:
        configs_copy = dict(configs)  # create a copy to remove keys

        # get the "first" key
        param = list(keys)[0]
        del configs_copy[param]

        first_key_values = configs[param]
        for value in first_key_values:
            result[param] = value

            for nested_config in generate_grid(configs_copy):
                result.update(nested_config)
                yield result
