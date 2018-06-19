import random
import numpy as np


def shuffle_dataset(graphs):
    '''
    In-place shuffling of the dataset.
    :param graphs: a list of tuples (X_values, Y_values, adjacency_lists, size), each representing a graph
    :param sizes: a list of sizes, one for each graph
    :return:
    '''
    random.shuffle(graphs)


def get_max_ariety(adjacency_lists, A):
    ariety = np.zeros(A)

    for l in adjacency_lists:
        tmp = np.zeros(A)
        for (el, a) in l:
            tmp[a] += 1

        ariety = np.maximum(ariety, tmp)

    return ariety


def split_dataset(graphs, train_perc, valid_perc, test_perc, shuffle=False):
    '''
    Shuffles and splits a dataset in train, validation and test. The sum of the last parameters should be 1.0
    :param graphs: a list of tuples (X_values, Y_values, adjacency_lists, size), each representing a graph
    :param sizes: a list of sizes, one for each graph
    :param train_perc: the percentage of examples for the training set, from 0.0 to 1.0
    :param valid_perc: the percentage of examples for the validation set, from 0.0 to 1.0
    :param test_perc: the percentage of examples for the test set, from 0.0 to 1.0
    :param shuffle: if True, shuffles the list of graphs before splitting
    :return: a tuple made of 3 datasets: (graphs_train, graphs_valid, graphs_tests), which are identical
    in structure to the dataset passed as input
    '''
    assert train_perc + valid_perc + test_perc == 1.0

    if shuffle:
        shuffle_dataset(graphs)

    total_examples = len(graphs)

    train_dim = int(total_examples*train_perc)  # python converts to the nearest lower integer
    valid_dim = int(total_examples*valid_perc)
    test_dim = int(total_examples*test_perc)

    graphs_train = graphs[0:train_dim]
    graphs_valid = graphs[train_dim:train_dim + valid_dim]
    graphs_test = graphs[train_dim + valid_dim:train_dim + valid_dim + test_dim]

    return graphs_train, graphs_valid, graphs_test


def unravel(graphs, one_target_per_graph=True):
    '''
    Takes "graphs" and unravels them into a single big one, using the size of each graph to shift indices
    in the adjacency lists.
    This is required to give the dataset to the model, and it should be done after having split between
    training, validation and test.
    The method returns a new copy of adjacency_lists, so that the original graph is not modified.
    :param graphs: a list of tuples (X_values, Y_values, adjacency_lists, size), each representing a graph
    :param one_target_per_graph: if true, Y will have one target per graph; if false,Y will have a target for each node
    :return: a single graph tuple: (X, Y, adjacency_lists, sizes). "sizes" is a list of all graphs' sizes
    '''

    X_final = []
    Y_final = []
    adjacency_lists_final = []
    sizes = []

    start_id = 0
    for (X, Y, adjacency_lists, size) in graphs:

        X_final.extend(X)
        sizes.append(size)

        if one_target_per_graph:
            Y_final.append(Y[0])  # The class is equal for all nodes of the graph
        else:
            Y_final.extend(Y)

        adjacency_lists_shifted = [[(u + start_id, a) for (u, a) in l] for l in adjacency_lists]
        adjacency_lists_final.extend(adjacency_lists_shifted)

        start_id += size

    return np.array(X_final), np.array(Y_final), adjacency_lists_final, sizes

