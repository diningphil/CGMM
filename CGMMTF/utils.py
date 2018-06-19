from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf


def save_statistics(adjacency_lists, inferred_states, target, A, C2, filename, layer_no):
    """
    :param last_states: the last array of states
    :param prev_statistics: the statistics needed: list of numpy matrices UxAxC2
    :return: the statistics needed for this model, according to the Lprec parameter
    """

    C2 += 1  # always need to consider the bottom state

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(filename + '_' + str(layer_no))

    for u in range(0, len(adjacency_lists)):
        # Compute statistics
        statistics = np.zeros((A, C2))

        incident_nodes = adjacency_lists[u]
        for u2, a in incident_nodes:
            node_state = inferred_states[u2]
            statistics[a, node_state] += 1

        # Create a new Example
        label = target[u]
        # Create a feature
        feature = {'train/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                   'train/stats': tf.train.Feature(bytes_list=tf.train.BytesList(value=[statistics.tostring()]))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Append to file (can you do it?)
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def recover_statistics(filename, layer, A, C2):
    '''
    This function creates the dataset by reading from different files according to "layers".
    :param filename:
    :param layers:
    :return: A tf.data.Dataset, where each element has shape [L, A, C2]. The first dimension is obtained by merging
    the L required files specified by the "layers" argument
    '''

    C2 += 1  # always need to consider the bottom state

    def parse_example(example):

        feature = {'train/label': tf.FixedLenFeature([], tf.int64),
                   'train/stats': tf.FixedLenFeature([], tf.string)}


        # Decode the record read by the reader
        features = tf.parse_single_example(example, features=feature)

        label = tf.cast(features['train/label'], tf.int64)

        stats = tf.decode_raw(features['train/stats'], tf.float64)


        # Reshape image data into the original shape
        stats = tf.reshape(stats, [1, A, C2])  # add dimension relative to L

        return stats

    # FIRST VERSION: reads a single file and produces a dataset
    stats_dataset = tf.data.TFRecordDataset([filename + '_' + str(layer)])
    stats_dataset = stats_dataset.map(parse_example)

    return stats_dataset