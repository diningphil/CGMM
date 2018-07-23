from __future__ import absolute_import, division, print_function
import random
import numpy as np
import tensorflow as tf

import os

stats_folder = 'saved_statistics'
unigrams_folder = 'saved_unigrams'


def save_unigrams(unigrams, unigrams_filename, layer_no):
    '''
    :param unigrams:
    :param unigrams_filename:
    :param layer:
    :return:
    '''
    if not os.path.exists(unigrams_folder):
        os.makedirs(unigrams_folder)

    if not os.path.exists(os.path.join(unigrams_folder, unigrams_filename)):
        os.makedirs(os.path.join(unigrams_folder, unigrams_filename))

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(os.path.join(unigrams_folder, unigrams_filename, unigrams_filename)
                                         + '_' + str(layer_no))

    for i in range(0, unigrams.shape[0]):
        # Create a feature
        feature = {'inference/unigram': tf.train.Feature(bytes_list=tf.train.BytesList(value=[unigrams[i,:].tostring()]))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def recover_unigrams(unigrams_filename, layers, C, concatenate=True, return_numpy=True):
    '''
    This function creates the dataset by reading from different files according to "layers".
    :param unigrams_filename:
    :param layers:
    :return: A tf.data.Dataset, where each element has shape [?, L*C] if concatenate=True, [?, L, C] otherwise
    '''
    L = len(layers)

    def parse_example(*examples):

        unigrams = None

        for l in range(0, L):
            example = examples[l]
            feature = {'inference/unigram': tf.FixedLenFeature([], tf.string)}

            # Decode the record read by the reader
            features = tf.parse_single_example(example, features=feature)

            unigram = tf.decode_raw(features['inference/unigram'], tf.float64)

            # Reshape data into the original shape
            if not concatenate:
                unigram = tf.reshape(unigram, [1, C])  # add dimension relative to L

                if unigrams is None:
                    unigrams = unigram
                else:
                    unigrams = tf.concat([unigrams, unigram], axis=0)  # L is the middle axis
            else:
                unigram = tf.reshape(unigram, [C])

                if unigrams is None:
                    unigrams = unigram
                else:
                    unigrams = tf.concat([unigrams, unigram], axis=0)  # here we want LxC

        return unigrams

    layers_stats = []
    for layer in layers:
        # print("Loading", filename + '_' + str(layer))

        layer_stats = tf.data.TFRecordDataset([os.path.join(unigrams_folder, unigrams_filename, unigrams_filename)
                                               + '_' + str(layer)])
        layers_stats.append(layer_stats)

    unigrams_dataset = tf.data.Dataset.zip(tuple(layers_stats))
    unigrams_dataset = unigrams_dataset.map(parse_example)

    if not return_numpy:
        return unigrams_dataset
    else:
        it = unigrams_dataset.batch(100000000).make_one_shot_iterator()
        n = it.get_next()
        with tf.Session().as_default():
            ds = n.eval()
            return ds


def save_statistics(adjacency_lists, inferred_states, A, C2, filename, layer_no):
    '''
    :param adjacency_lists:
    :param inferred_states:
    :param A:
    :param C2:
    :param filename:
    :param layer_no:
    :return:
    '''
    C2 += 1  # always need to consider the bottom state

    if not os.path.exists(stats_folder):
        os.makedirs(stats_folder)

    if not os.path.exists(os.path.join(stats_folder, filename)):
        os.makedirs(os.path.join(stats_folder, filename))

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(os.path.join(stats_folder, filename, filename) + '_' + str(layer_no))

    for u in range(0, len(adjacency_lists)):
        # Compute statistics
        statistics = np.zeros((A, C2))

        incident_nodes = adjacency_lists[u]
        for u2, a in incident_nodes:
            node_state = inferred_states[u2]
            statistics[a, node_state] += 1

        # Create a feature
        feature = {'train/stats': tf.train.Feature(bytes_list=tf.train.BytesList(value=[statistics.tostring()]))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Append to file (can you do it?)
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def recover_statistics(filename, layers, A, C2):
    '''
    This function creates the dataset by reading from different files according to "layers".
    :param filename:
    :param layers:
    :return: A tf.data.Dataset, where each element has shape [L, A, C2]. The first dimension is obtained by merging
    the L required files specified by the "layers" argument
    '''

    C2 += 1  # always need to consider the bottom state
    L = len(layers)

    def parse_example(*examples):

        stats = None

        for l in range(0, L):
            example = examples[l]
            feature = {'train/stats': tf.FixedLenFeature([], tf.string)}


            # Decode the record read by the reader
            features = tf.parse_single_example(example, features=feature)

            l_stats = tf.decode_raw(features['train/stats'], tf.float64)

            # Reshape image data into the original shape
            l_stats = tf.reshape(l_stats, [1, A, C2])  # add dimension relative to L

            if stats is None:
                stats = l_stats
            else:
                stats = tf.concat([stats, l_stats], axis=0)

        return stats

    layers_stats = []
    for layer in layers:
        # print("Loading", filename + '_' + str(layer))

        layer_stats = tf.data.TFRecordDataset([os.path.join(stats_folder, filename, filename) + '_' + str(layer)])
        layers_stats.append(layer_stats)

    stats_dataset = tf.data.Dataset.zip(tuple(layers_stats))
    stats_dataset = stats_dataset.map(parse_example)

    return stats_dataset


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

