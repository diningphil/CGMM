from __future__ import absolute_import, division, print_function
import time
from CGMMTF.MultinomialMixtureTF import MultinomialMixture
from CGMMTF.VStructureTF import VStructure
from CGMMTF.DatasetUtilities import *

from sklearn import svm

checkpoint_folder = './checkpoints'


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


def build_architecture(K, A, C, use_statistics, layers):
    architecture = []
    for layer in range(0, layers):
        if layer == 0:
            architecture.append(MultinomialMixture(C, K))
        else:
            layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]
            L = len(layer_wise_statistics)
            architecture.append(VStructure(C, C, K, L, A, layer))

    return architecture


def incremental_inference(model_name, K, A, C, layers, use_statistics, batch_dataset, adjacency_lists, sizes,
                        unigram_filename, statistics_filename, batch_size=2000):
    '''
    Performs inference throughout the architecture. Assumes C = C2
    :param architecture:
    :param A:
    :param C2:
    :param use_statistics: the subset of preceeding layers to use
    :param target_dataset: a Dataset.
    :param adjacency_lists: a list of lists (one for each vertex) representing the connections between vertexes
    :param sizes: needed to determine which vertexes belong to which graph and build the output
    :param batch_size: batch inference for efficiency purposes
    :return: [no_graphs, L, C] tensor of frequency counts vectors for each layer and vertex
    '''

    architecture = build_architecture(K, A, C, use_statistics, layers)

    with tf.Session() as sess:
        variables_to_restore = []
        max_depth = len(architecture)

        sess.run(tf.global_variables_initializer())

        for layer in range(0, max_depth):
            model = architecture[layer]
            if layer == 0:
                variables_to_restore.extend([model.prior, model.emission])
            else:
                variables_to_restore.extend([model.transition, model.emission, model.arcS, model.layerS])

        restore = tf.train.Saver(variables_to_restore)
        restore.restore(sess, os.path.join(checkpoint_folder, model_name, 'model.ckpt'))
        print("Restored all parameters")

        for layer in range(0, max_depth):
            model = architecture[layer]

            if layer == 0:
                print("INFERENCE LAYER 0")

                # TODO you should find an elegant way to store vectors batch by batch. It is not so simple
                # because of the sizes argument
                inferred_states = model.perform_inference(batch_dataset, sess)
                save_unigrams(_aggregate_states(C, sizes, inferred_states), unigram_filename, layer)

            else:
                print("INFERENCE LAYER", layer)

                layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

                stats_dataset = recover_statistics(statistics_filename, layer_wise_statistics, A, C)
                batch_statistics = stats_dataset.batch(batch_size=batch_size)

                # TODO you should find an elegant way to store vectors batch by batch. It is not so simple
                # because of the sizes argument
                inferred_states = model.perform_inference(batch_dataset, batch_statistics, sess)
                save_unigrams(_aggregate_states(C, sizes, inferred_states), unigram_filename, layer)

            save_statistics(adjacency_lists, inferred_states, A, C, statistics_filename, layer)

    print("Resetting the default graph")
    tf.reset_default_graph()

    # Clear the statistics files (no longer necessary)
    for layer_no in range(0, max_depth):
        os.remove(os.path.join(stats_folder, statistics_filename, statistics_filename + '_' + str(layer_no)))


def incremental_training(C, K, A, use_statistics, adjacency_lists, batch_dataset, layers, statistics_filename,
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
        print("LAYER 0")

        mm = MultinomialMixture(C, K)
        mm.train(batch_dataset, sess, max_epochs=max_epochs, threshold=threshold, debug=False)

        if save_name is not None:
            # Add ops to save and restore the variables ('uses the variables' names')
            variables_to_save.extend([mm.prior, mm.emission])

        # TODO you should find an elegant way to get inferred_states as a dataset, to limit memory consumption
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

            vs = VStructure(C, C, K, L, A, current_layer=layer)

            vs.train(batch_dataset, batch_statistics, sess, max_epochs=max_epochs, threshold=threshold)

            if save_name is not None:
                # Add ops to save and restore the variables ('uses the variables' names')
                variables_to_save.extend([vs.emission, vs.arcS, vs.layerS, vs.transition])

            # TODO you should find an elegant way to get inferred_states as a dataset, to limit memory consumption
            inferred_states = vs.perform_inference(batch_dataset, batch_statistics, sess)
            save_statistics(adjacency_lists, inferred_states, A, C, statistics_filename, layer)

        if save_name is not None:

            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            if not os.path.exists(os.path.join(checkpoint_folder, save_name)):
                os.makedirs(os.path.join(checkpoint_folder, save_name))

            saver = tf.train.Saver(variables_to_save)
            print("Model saved in", saver.save(sess, os.path.join(checkpoint_folder, save_name, 'model.ckpt')))

    if save_name is not None:
        print("Resetting the default graph")
        tf.reset_default_graph()

    # Clear the statistics files (no longer necessary)
    for layer_no in range(0, layers):
        os.remove(os.path.join(stats_folder, statistics_filename, statistics_filename + '_' + str(layer_no)))


def compute_accuracy(predictions, ground_truth):
    assert len(predictions) == len(ground_truth)
    return 100*(np.sum(ground_truth == predictions) / len(predictions))


def compute_svm_accuracy(unigrams, Y, svmC, gamma, unigrams_valid=None, Y_valid=None):
    '''
    '''
    # SVC performs a AVA approach for multiclass classification
    if gamma is None:
        clf = svm.SVC(C=svmC, kernel='linear', shrinking=False)
    else:
        clf = svm.SVC(C=svmC, kernel='rbf', gamma=gamma, shrinking=False)

    # Train on train set
    clf.fit(unigrams, Y)

    # Compute train accuracy
    tr_acc = compute_accuracy(clf.predict(unigrams), Y)

    vl_acc = -1
    if unigrams_valid is not None and Y_valid is not None:
        vl_acc = compute_accuracy(clf.predict(unigrams_valid), Y_valid)

    return tr_acc, vl_acc
