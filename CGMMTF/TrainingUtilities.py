from __future__ import absolute_import, division, print_function
import time
from CGMMTF.MultinomialMixtureTF import MultinomialMixture
from CGMMTF.VStructureTF import VStructure
from CGMMTF.DatasetUtilities import *


def current_milli_time():
    return int(round(time.time() * 1000))


def _aggregate_states(C, sizes, states):
    '''
    Aggregates the states into a frequency vector
    :param C: the size of the hidden states' alphabet
    :param sizes: a list of integers, each of them representing the size of a graph
    :param states: the array of states produced for all the graphs
    :return: The frequency state vector of dimension no_graphsxC
    '''
    freq_aggregated = []

    last_states = np.array(states, dtype='int')

    curr_size = 0
    for size in sizes:
        freq = np.zeros(C)
        np.add.at(freq, last_states[curr_size:(curr_size + size)], 1)
        freq /= size  # Normalize
        freq_aggregated.append(freq)
        curr_size += size

    return np.array(freq_aggregated)


def compute_input_matrix(architecture, C, X, adjacency_lists, sizes, up_to_layer=-1, concatenate=False,
                         return_all_states=False):
    '''
    Computes the input matrix (state frequency vectors) of dimension Cx1 if concatenate is False, Cxup_to_layer
    otherwise.
    :param architecture: the trained architecture
    :param C: the size of the hidden states' alphabet
    :param X: the list of all vertexes labels of the dataset. See unravel in DatasetUtilities.
    :param adjacency_lists:
    :param sizes: a list of integers, each of them representing the size of a graph. See unravel in DatasetUtilities.
    :param up_to_layer: by default, up to the end of the architecture. Alternatively one can specify when to stop
    :param concatenate: if True, concatenate the layers
    :param return_all_states: if true, return the numpy matrix of all states computed at all layers
    :return: the input matrix if return_all_states is False, (input_matrix, all_states) otherwise
    '''

    if up_to_layer == -1:
        up_to_layer = len(architecture)

    all_states = incremental_inference(architecture, X, adjacency_lists, up_to_layer=up_to_layer)
    last_states = all_states[-1, :]  # all states has shape (up_to_layer, len(X))

    if not concatenate:
        input_matrix = _aggregate_states(C, sizes, last_states)
    else:
        for layer in range(0, up_to_layer):
            new_input_matrix = _aggregate_states(C, sizes, last_states)
            if layer == 0:
                input_matrix = new_input_matrix
            else:
                input_matrix = np.hstack((input_matrix, new_input_matrix))

    if return_all_states:
        return input_matrix, all_states
    else:
        return input_matrix


def incremental_inference(model_name, A, C2, use_statistics, target_dataset, adjacency_lists, sizes,
                          statistics_filename, batch_size=2000, up_to_layer=-1):
    '''
    Performs inference throughout the architecture
    :param architecture: the trained architecture
    :param A:
    :param C2:
    :param use_statistics: the subset of preceeding layers to use
    :param target_dataset: a Dataset.
    :param adjacency_lists: a list of lists (one for each vertex) representing the connections between vertexes
    :param sizes: needed to determine which vertexes belong to which graph and build the output
    :param batch_size: batch inference for efficiency purposes
    :param up_to_layer: by default, up to the end of the architecture. Alternatively one can specify when to stop
    :return: [no_graphs, L, C] tensor of frequency counts vectors for each layer and vertex
    '''
    # TODO LOAD THE ARCHITECTURE!

    if up_to_layer != -1:
        max_depth = up_to_layer
    else:
        max_depth = len(architecture)

    with tf.Session() as sess:
        # build minibatches from dataset
        batch_dataset = target_dataset.batch(batch_size=batch_size)

        for layer in range(0, max_depth):
            model = architecture[layer]

            if layer == 0:
                print("INFERENCE LAYER 0")
                inferred_states = model.perform_inference(batch_dataset, sess)
            else:
                print("INFERENCE LAYER", layer)

                layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

                stats_dataset = recover_statistics(statistics_filename, layer_wise_statistics, A, C2)
                batch_statistics = stats_dataset.batch(batch_size=batch_size)

                inferred_states = model.perform_inference(batch_dataset, batch_statistics, sess)

            save_statistics(adjacency_lists, inferred_states, A, C2, statistics_filename, layer)

    # TODO NOW CHOOSE WHAT TO RETURN!
    # IT MAY BE THE A MATRIX [?, C] from which to compute the vector of frequency counts
    # return: [no_graphs, L, C] tensor of frequency counts vectors for each layer and vertex


def incremental_training(C, K, A, use_statistics, adjacency_lists, target_dataset, layers, statistics_filename,
                         threshold=0, max_epochs=100, batch_size=2000, save_name=None):
    '''
    Build an architecture. Assumes C is equal to C2, as it is often the case
    :param C: the size of the hidden states' alphabet
    :param K: the size of the emission alphabet (discrete)
    :param A: the size of the edges' alphabet
    :param use_statistics: an array with a subset of preceding layers to consider. For example, np.array([1,2,3])
    considers the immediate 3 preceding layers, whenever possible.
    :param adjacency_lists: a list of lists (one for each vertex) representing the connections between vertexes
    :param target_dataset: a Dataset.
    :param layers: the depth of the architecture
    :param threshold: the threshold parameter for the EM algorithm
    :param max_epochs: the maximum number of epochs per training
    :return: an architecture)
    '''
    variables_to_save = []

    with tf.Session() as sess:
        # build minibatches from dataset
        batch_dataset = target_dataset.batch(batch_size=batch_size)

        print("LAYER 0")
        with tf.variable_scope("base_layer"):
            mm = MultinomialMixture(C, K)
            mm.train(batch_dataset, sess, max_epochs=max_epochs, threshold=threshold)

            if save_name is not None:
                # Add ops to save and restore the variables ('uses the variables' names')
                variables_to_save.extend([mm.prior, mm.emission])

        inferred_states = mm.perform_inference(batch_dataset, sess)
        save_statistics(adjacency_lists, inferred_states, A, C, statistics_filename, 0)

        for layer in range(1, layers):
            print("LAYER", layer)

            # e.g 1 - [1, 3] = [0, -2] --> [0]
            # e.g 5 - [1, 3] = [4, 2]  --> [4, 2]
            layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

            L = len(layer_wise_statistics)

            # print(layer_wise_statistics)

            stats_dataset = recover_statistics(statistics_filename, layer_wise_statistics, A, C)
            batch_statistics = stats_dataset.batch(batch_size=batch_size)

            with tf.variable_scope("general_layer"):
                vs = VStructure(C, C, K, L, A, current_layer=layer)

                vs.train(batch_dataset, batch_statistics, sess, max_epochs=max_epochs, threshold=threshold)

                if save_name is not None:
                    # Add ops to save and restore the variables ('uses the variables' names')
                    variables_to_save.extend([vs.emission, vs.arcS, vs.layerS, vs.transition])

                inferred_states = vs.perform_inference(batch_dataset, batch_statistics, sess)
                save_statistics(adjacency_lists, inferred_states, A, C, statistics_filename, layer)

            if save_name is not None:
                saver = tf.train.Saver(variables_to_save)
                saver.save(sess, './checkpoints/' + save_name + '/model.ckpt')
