import concurrent.futures
from datetime import datetime
import time
import pickle

import sklearn.metrics
from sklearn import svm
import logging
import math

from CGMM.MultinomialMixture import MultinomialMixture as BaseCase
from CGMM.VStructure import VStructure as GeneralCase
from CGMM.DatasetUtilities import *

from multiprocessing import Lock


def current_milli_time():
    return int(round(time.time() * 1000))


# SHARED AMONG PROCESSES
lock = Lock()


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


def incremental_inference(architecture, X, adjacency_lists, up_to_layer=-1):
    '''
    Performs inference throughout the architecture
    :param architecture: the trained architecture
    :param X: the list of all vertexes labels of the dataset. See unravel in DatasetUtilities.
    :param adjacency_lists: a list of lists (one for each vertex) representing the connections between vertexes
    :param up_to_layer: by default, up to the end of the architecture. Alternatively one can specify when to stop
    :return:a tuple (last states inferred, all states inferred)
    '''

    layer = 0
    last_states = None
    prev_statistics = None
    last_statistics = None

    if up_to_layer == -1:
        layers = len(architecture)
    else:
        architecture = architecture[:up_to_layer]
        layers = len(architecture)

    inferred_states = np.empty((len(architecture), len(X)), dtype='int')

    for model in architecture:
        if layer == 0:
            last_states = model.inference(X)
        elif layer < layers:
            last_states, last_statistics = model.inference(X, adjacency_lists, last_states, prev_statistics)

        inferred_states[layer, :] = last_states

        # Append statistics of the new layer after having computed them
        if prev_statistics is None and last_statistics is not None:
            prev_statistics = last_statistics
        elif prev_statistics is not None and last_statistics is not None:
            # Concat current statistics with previous one
            prev_statistics = np.concatenate([last_statistics, prev_statistics], axis=1)
        else:
            assert prev_statistics is None
            assert last_statistics is None
        
        layer += 1

    return inferred_states


def incremental_training(C, K, A, Lprec, adjacency_lists, X, layers, threshold=0, max_epochs=100, architecture=None,
                         prev_statistics=None, last_states=None):
    '''
    Build an architecture
    :param C: the size of the hidden states' alphabet
    :param K: the size of the emission alphabet (discrete)
    :param A: the size of the edges' alphabet
    :param Lprec: an array with a subset of preceding layers to consider. For example, np.array([1,2,3])
    considers the immediate 3 preceding layers, whenever possible.
    :param adjacency_lists: a list of lists (one for each vertex) representing the connections between vertexes
    :param X: the list of all vertexes labels of the dataset. See unravel in DatasetUtilities.
    :param layers: the depth of the architecture
    :param threshold: the threshold parameter for the EM algorithm
    :param max_epochs: the maximum number of epochs per training
    :param architecture: if provided together with prev_statistics and last_states, resumes training.
    :param prev_statistics: see architecture description
    :param last_states: see architecture description
    :return: a tuple (architecture, prev_statistics, last_states)
    '''
    # Assume hidden states alphabet does not vary between layers --> easy to extend, CGMM already considers a
    # different parameter C2.

    if architecture is not None:
        print("Resuming training...")
    else:
        architecture = []

        print("Training base layer...")
        base_model = BaseCase(C, K)
        base_model.train(X, threshold, max_epochs)
        last_states = base_model.inference(X)
        architecture.append(base_model)

    i = len(architecture)
    while i < layers:

        print("Training new layer", (i+1))
        model = GeneralCase(C, C, K, Lprec, A)
        if i == 1:
            model.train(X, threshold, max_epochs, adjacency_lists, last_states, None, layer=(i + 1))
            last_states, last_statistics = model.inference(X, adjacency_lists, last_states, None)
        elif i < layers:
            model.train(X, threshold, max_epochs, adjacency_lists, last_states, prev_statistics, layer=(i + 1))
            last_states, last_statistics = model.inference(X, adjacency_lists, last_states, prev_statistics)


        # TODO to limit memory usage you should consider reading from disk the statistics relative to the minibatch only
        # I still do not know how to do this, but it is the way.

        # Append statistics of the new layer after having computed them
        if prev_statistics is None:
            prev_statistics = last_statistics
        else:
            # Append last statistics to the TOP.
            prev_statistics = np.concatenate([last_statistics, prev_statistics], axis=1)

        architecture.append(model)

        i += 1

    return architecture, prev_statistics, last_states


def incremental_collective_inference(architecture, prediction_set, adjacency_lists, chosen, upToLayer=-1):
    layer = 0
    last_states = None
    prev_statistics = None

    if upToLayer == -1:
        layers = len(architecture)
    else:
        architecture = architecture[:upToLayer]
        layers = len(architecture)

    inferred_states = np.empty((len(architecture), len(prediction_set)), dtype='int')
    inferred_labels = np.empty((len(architecture), len(prediction_set)), dtype='int')
    predicted = prediction_set

    for model in architecture:
        predicted = np.array(predicted, copy=True)

        if layer == 0:
            # Invoke the method of the MultinomialMixture
            predicted, last_states = model.collective_inference(predicted, chosen)

        elif layer < layers:
            # Invoke the method of the VStructure
            predicted, last_states, last_statistics = \
                model.collective_inference(predicted, adjacency_lists, last_states, chosen, prev_statistics)

        inferred_states[layer, :] = last_states
        inferred_labels[layer, :] = predicted

        # Append statistics of the new layer after having computed them
        if prev_statistics is None and layer > 0:
            # Append first bunch of statistics
            prev_statistics = last_statistics
        elif layer > 0:
            # Concat current statistics with previous one
            prev_statistics = np.concatenate([last_statistics, prev_statistics], axis=1)

        layer += 1

    last_states = inferred_states[-1, :]
    return last_states, inferred_states, inferred_labels


def compute_accuracy(predictions, ground_truth):
    assert len(predictions) == len(ground_truth)
    return 100*(np.sum(ground_truth == predictions) / len(predictions))


def confusion_matrix(predictions, ground_truth):
    assert len(predictions) == len(ground_truth)
    return sklearn.metrics.confusion_matrix(ground_truth, predictions)


def jaccard_gram_matrix(feature_matrix_1, feature_matrix_2 = None):
    '''
    Computes the gram matrix given two feature matrices
    :param feature_matrix_1: m x F matrix
    :param feature_matrix_2: n x F matrix
    :return: m x n gram matrix
    '''
    if feature_matrix_2 is None:
        feature_matrix_2 = feature_matrix_1

    assert feature_matrix_1.shape[1] == feature_matrix_2.shape[1]
    m = feature_matrix_1.shape[0]
    n = feature_matrix_2.shape[0]
    gram_matrix = np.zeros((m, n))
    '''
    # Here to understand what is coded next in a more compact way
    for i in range(0, no_ex):
        for j in range(0, no_ex):
            min = np.sum(np.minimum(feature_matrix[i, :], feature_matrix[j, :]))
            max = np.sum(np.maximum(feature_matrix[i, :], feature_matrix[j, :]))
            gram_matrix[i, j] = min/max
    '''
    f = np.vectorize(lambda x: 1.0 if np.equal(x, 0.0) else x)

    for j in range(0, n):
        min = np.sum(np.minimum(feature_matrix_1[:, :], feature_matrix_2[j, :]), axis=1)
        max = f(np.sum(np.maximum(feature_matrix_1[:, :], feature_matrix_2[j, :]), axis=1))
        np.divide(min, max, out=gram_matrix[:, j])

    return gram_matrix


def bigram(all_states, adjacency_lists, sizes, A, C, concatenate):
    '''
    Computes the bigram vector of dimension CxC. If concatenate is True, concatenates the bigrams of all layers
    :param all_states: matrix (layer,state) comprising of the states of all vertexes for each layer
    :param adjacency_lists: a list of lists (one for each vertex) representing the connections between vertexes
    :param sizes: a list of integers, each of them representing the size of a graph
    :param A: the size of the edges' alphabet
    :param C: the size of the hidden states' alphabet
    :param concatenate: if true, concatenate the bigram vectors of all layers
    :return: The bigram vector
    '''
    no_examples = len(sizes)

    dim = C*C

    # Take the last computed states and compute new_statistics.
    if not concatenate:
        feature_matrix = np.zeros((no_examples, dim), dtype='int')

        last_states = all_states[-1, :]

        # Compute statistics
        new_statistics = np.zeros((len(adjacency_lists), A, C))
        for u in range(0, len(adjacency_lists)):
            incident_nodes = adjacency_lists[u]
            for u2, a in incident_nodes:
                node_state = last_states[u2]
                new_statistics[u, a, node_state] += 1

        # Update the feature matrix for each structure
        start_u = 0
        structure_no = 0
        for structure_len in sizes:
            for u in range(start_u, start_u+structure_len):
                for j in range(0, C):
                    feature_matrix[structure_no, last_states[u]*C + j] += \
                        np.sum(new_statistics[u, :, j], dtype='int')
                        # note: info about A is not exploited

            start_u += structure_len
            structure_no += 1

    else:
        feature_matrix = None

        for layer in range(0, all_states.shape[0]):
            new_feature_matrix = np.zeros((no_examples, dim), dtype='int')

            last_states = all_states[layer, :]

            # Compute statistics
            new_statistics = np.zeros((len(adjacency_lists), A, C))
            for u in range(0, len(adjacency_lists)):
                incident_nodes = adjacency_lists[u]
                for u2, a in incident_nodes:
                    node_state = last_states[u2]
                    new_statistics[u, a, node_state] += 1

            # Update the feature matrix for each structure
            start_u = 0
            structure_no = 0
            for structure_len in sizes:
                for u in range(start_u, start_u + structure_len):
                    for j in range(0, C):
                        new_feature_matrix[structure_no, last_states[u] * C + j] += \
                            np.sum(new_statistics[u, :, j], dtype='int')
                        # info about A is not exploited

                start_u += structure_len
                structure_no += 1

            if feature_matrix is None:
                feature_matrix = new_feature_matrix
            else:
                feature_matrix = np.hstack((feature_matrix, new_feature_matrix))

    return feature_matrix


def _compute_svm_accuracy(architecture, C, svmC, gamma, X_train, Y_train, adjacency_lists_train, sizes_train, X_valid, Y_valid,
                          adjacency_lists_valid, sizes_valid, concatenate_fingerprints=False, up_to_layer=-1):
    '''
    Trains an SVM on top of a CGMM and computes training and validation accuracy 
    :param architecture: the trained architecture
    :param C: the size of the hidden states' alphabet
    :param svmC: SVM's C parammeter
    :param gamma: RBF gamma parameter
    :param X_train: the list of all vertexes labels of the training dataset. See unravel in DatasetUtilities.
    :param Y_train: the list of all vertexes targets of the training dataset. See unravel in DatasetUtilities.
    :param adjacency_lists_train: a list of lists (one for each vertex) representing the connections between vertexes
    :param sizes_train: a list of integers, each of them representing the size of a training graph
    :param X_valid: the list of all vertexes labels of the training dataset. See unravel in DatasetUtilities.
    :param Y_valid: the list of all vertexes targets of the training dataset. See unravel in DatasetUtilities.
    :param adjacency_lists_valid: a list of lists (one for each vertex) representing the connections between vertexes
    :param sizes_valid: a list of integers, each of them representing the size of a validation graph
    :param concatenate_fingerprints:  if true, concatenate fingerprints produced at each layer
    :param up_to_layer: default is all the architecture, otherwise indicate when to stop
    :return: (training accuracy, validation accuracy)
    '''
    # Compress each structure in a vector of states' frequences
    input_matrix_train = compute_input_matrix(architecture, C, X_train, adjacency_lists_train, sizes_train,
                                              concatenate=concatenate_fingerprints, up_to_layer=up_to_layer)
    # Get the label of each structure using the root label

    # SVC performs a AVA approach for multiclass classification
    if gamma is None:
        clf = svm.SVC(C=svmC, kernel='linear', shrinking=False)
    else:
        clf = svm.SVC(C=svmC, kernel='rbf', gamma=gamma, shrinking=False)

    # Train on train set
    clf.fit(input_matrix_train, Y_train)

    # Compute train accuracy
    tr_acc = compute_accuracy(clf.predict(input_matrix_train), Y_train)

    # Now test on the valid set
    input_matrix_valid = compute_input_matrix(architecture, C, X_valid, adjacency_lists_valid, sizes_valid,
                                              concatenate=concatenate_fingerprints, up_to_layer=up_to_layer)

    # Compute valid accuracy
    predictions_valid = clf.predict(input_matrix_valid)
    vl_acc = compute_accuracy(predictions_valid, Y_valid)

    return tr_acc, vl_acc


def _compute_jaccard_accuracy(architecture, A, C, unibigram, X_train, Y_train,
                              adjacency_lists_train,
                              sizes_train, X_valid, Y_valid, adjacency_lists_valid,
                              sizes_valid,
                              concatenate_fingerprints,
                              unigram_train=None, unigram_valid=None,
                              allStates_train=None, allStates_valid=None):
    '''
    Trains a SVM with Jaccard kernel on top of a CGMM and computes training and validation accuracy
    :param architecture: the trained architecture
    :param A: the size of the edges' alphabet
    :param C: the size of the hidden states' alphabet
    :param unibigram: if true, concatenate the unigram with the bigram vector.
    :param X_train: the list of all vertexes labels of the training dataset. See unravel in DatasetUtilities.
    :param Y_train: the list of all vertexes targets of the training dataset. See unravel in DatasetUtilities.
    :param adjacency_lists_train: a list of lists (one for each vertex) representing the connections between vertexes
    :param sizes_train: a list of integers, each of them representing the size of a training graph
    :param X_valid: the list of all vertexes labels of the training dataset. See unravel in DatasetUtilities.
    :param Y_valid: the list of all vertexes targets of the training dataset. See unravel in DatasetUtilities.
    :param adjacency_lists_valid: a list of lists (one for each vertex) representing the connections between vertexes
    :param sizes_valid: a list of integers, each of them representing the size of a validation graph
    :param concatenate_fingerprints: if true, concatenate fingerprints produced at each layer
    :param up_to_layer: default is all the architecture, otherwise indicate when to stop
    :param unigram_train: if provided, avoid its computation, default is None
    :param unigram_valid: if provided, avoid its computation, default is None
    :param allStates_train: if provided, avoid its computation, default is None
    :param allStates_valid:if provided, avoid its computation, default is None
    :return: (training accuracy, validation accuracy)
    '''

    if not (unigram_train is not None and unigram_valid is not None
            and allStates_train is not None and allStates_valid is not None):

        unigram_train, allStates_train = compute_input_matrix(architecture, C, X_train, adjacency_lists_train,
                                                              sizes_train, concatenate=concatenate_fingerprints,
                                                              return_all_states=True)
        unigram_valid, allStates_valid = compute_input_matrix(architecture, C, X_valid, adjacency_lists_valid,
                                                              sizes_valid, concatenate=concatenate_fingerprints,
                                                              return_all_states=True)

    if unibigram:

        # The kernel matrix at test time needs to be the kernel between the test data and training
        # data ---> m x C and n x C --> m x n kernel matrix (the important thing is the feature dim "n")

        bigram_feat_mat = bigram(allStates_train, adjacency_lists_train, sizes_train, A, C, concatenate_fingerprints)
        features_train = np.hstack((unigram_train, bigram_feat_mat))

        unibigram_matrix = jaccard_gram_matrix(features_train, None)

        clf = svm.SVC(kernel='precomputed')
        clf.fit(unibigram_matrix, Y_train)

        # Compute train accuracy
        tr_acc = compute_accuracy(clf.predict(unibigram_matrix), Y_train)

        bigram_feat_mat = bigram(allStates_valid, adjacency_lists_valid, sizes_valid, A, C, concatenate_fingerprints)
        features_valid = np.hstack((unigram_valid, bigram_feat_mat))

        unibigram_matrix = jaccard_gram_matrix(features_valid, features_train)

        # Compute train accuracy
        vl_acc = compute_accuracy(clf.predict(unibigram_matrix), Y_valid)
    else:
        features_train = unigram_train

        unigram_matrix = jaccard_gram_matrix(features_train, None)

        clf = svm.SVC(kernel='precomputed')
        clf.fit(unigram_matrix, Y_train)
        # Compute train accuracy
        tr_acc = compute_accuracy(clf.predict(unigram_matrix), Y_train)

        features_valid = unigram_valid

        unigram_matrix = jaccard_gram_matrix(features_valid, features_train)

        # Compute train accuracy
        vl_acc = compute_accuracy(clf.predict(unigram_matrix), Y_valid)

    return tr_acc, vl_acc


def compute_fingerprints(C, K, A, Lprec, adjacency_lists_train, X_train, Y_train, sizes_train,
                         adjacency_lists_valid, X_valid, Y_valid, sizes_valid,
                         layers, threshold, max_epochs, run, concatenate_fingerprints, fingerprint_name):
    '''
    Train an architecture, compute the fingerprints and store them in a folfed named 'fingerprints' for future reuse.
    Precondition: the folder must exist
    :param C: the size of the hidden states' alphabet
    :param K: the size of the emission alphabet
    :param A: the size of the edges' alphabet
    :param Lprec: an array with a subset of preceding layers to consider. For example, np.array([1,2,3])
    considers the immediate 3 preceding layers, whenever possible.
    :param adjacency_lists_train: a list of lists (one for each vertex) representing the connections between vertexes
    :param X_train: the list of all vertexes labels of the training dataset. See unravel in DatasetUtilities.
    :param Y_train: the list of all vertexes targets of the training dataset. See unravel in DatasetUtilities.
    :param sizes_train: a list of integers, each of them representing the size of a training graph
    :param adjacency_lists_valid: a list of lists (one for each vertex) representing the connections between vertexes
    :param X_valid: the list of all vertexes labels of the training dataset. See unravel in DatasetUtilities.
    :param Y_valid: the list of all vertexes targets of the training dataset. See unravel in DatasetUtilities.
    :param sizes_valid: a list of integers, each of them representing the size of a validation graph
    :param layers: the depth of the architecture
    :param threshold: threshold for the EM algorithm
    :param max_epochs: maximum number of epochs per training
    :param run: specifies the current run (needed for storing fingerprints relative to different runs)
    :param concatenate_fingerprints: if true, concatenate fingerprints produced at each layer
    :param fingerprint_name: the initial name of the fingerprint
    :return:
    '''

    np.random.seed()
    layers = list(layers)  # create a copy
    layers.sort()  # Otherwise I cannot reuse the architecture
    architecture = None
    prev_statistics = None
    last_states = None

    for layer in layers:
        architecture, prev_statistics, last_states = \
            incremental_training(C, K, A, Lprec, adjacency_lists_train, X_train,
                                 layers=layer, threshold=threshold,
                                 max_epochs=max_epochs,
                                 architecture=architecture,
                                 prev_statistics=prev_statistics,
                                 last_states=last_states)

        unigram_train, all_states = \
            compute_input_matrix(architecture, C, X_train, adjacency_lists_train, sizes_train,
                                 concatenate=concatenate_fingerprints, return_all_states=True)
        unigram_valid, all_states_valid = \
            compute_input_matrix(architecture, C, X_valid, adjacency_lists_valid, sizes_valid,
                                 concatenate=concatenate_fingerprints, return_all_states=True)

        lock.acquire()

        try:
            # Store the fingerprints produced for training and validation (intermediate step which acts as
            # a checkpoint)
            with open("fingerprints/" + fingerprint_name + '_' + str(run) + '_' + str(C) + '_' + str(layer) + '_' +
                              str(Lprec) + '_' + str(concatenate_fingerprints), 'wb') as f:
                pickle.dump([unigram_train, unigram_valid, all_states, all_states_valid, adjacency_lists_train,
                             adjacency_lists_valid, sizes_train, sizes_valid, Y_train, Y_valid], f)
        except Exception as e:
            print(e)

        lock.release()


def fingerprints_to_svm_accuracy(C, Lprec, layer, runs, svmC, gamma, concatenate_fingerprints, fingerprint_name):
    '''
    Load the stored fingerprint (see compute_fingerprints), computes the accuracy on it using a SVM with RBF kernel
    and log information about average (over runs) training and validation accuracy and standard deviation.
    Precondition: a logging file must have been opened
    :param C: the size of the hidden states' alphabet
    :param Lprec: an array with a subset of preceding layers to consider. For example, np.array([1,2,3])
    :param layer: the layer associated to the fingerprint
    :param runs: how many runs have been done for this particular configuration
    :param svmC: the RBF C parameter
    :param gamma: the RBF gamma parameter
    :param concatenate_fingerprints: the boolean decision associated to the fingerprint
    :param fingerprint_name: the initial name of the fingerprint
    :return:
    '''
    np.random.seed()

    tr_runs = [0. for _ in range(0, runs)]
    vl_runs = [0. for _ in range(0, runs)]

    for run in range(0, runs):
        with open("./fingerprints/" + fingerprint_name + '_' + str(run) + '_' + str(C) + '_' +
                          str(layer) + '_' + str(Lprec) + '_' +
                          str(concatenate_fingerprints), 'rb') as f:
            [unigram_train, unigram_valid, allStates_train, allStates_valid, adjacency_lists_train,
             adjacency_lists_valid, sizes_train, sizes_valid, Y_train, Y_valid] = pickle.load(f)

            # SVC performs a AVA approach for multiclass classification
            clf = svm.SVC(C=svmC, kernel='rbf', gamma=gamma, shrinking=False)

            # Train on train set
            clf.fit(unigram_train, Y_train)

            # Compute train accuracy
            tr_runs[run] = compute_accuracy(clf.predict(unigram_train), Y_train)

            # Compute valid accuracy
            predictions_valid = clf.predict(unigram_valid)
            vl_runs[run] = compute_accuracy(predictions_valid, Y_valid)

    tr_acc = np.average(tr_runs)
    tr_std = np.std(tr_runs)
    vl_acc = np.average(vl_runs)
    vl_std = np.std(vl_runs)

    log = fingerprint_name + ' ' + str(C) + ' ' + str(layer) + ' ' + str(Lprec) + ' ' + \
          str(concatenate_fingerprints) + ' ' + str(svmC) + ' ' + str(gamma) + ' ' + str(tr_acc) + ' ' + \
          str(tr_std) + ' ' + str(vl_acc) + ' ' + str(vl_std)

    logging.info(log)


def fingerprints_to_jaccard_accuracy(C, A, Lprec, layer, runs, unibigram, concatenate_fingerprints, fingerprint_name):
    '''
    Load the stored fingerprint (see compute_fingerprints), computes the accuracy on it using a SVM with Jaccard kernel
    and log information about average (over runs) training and validation accuracy and standard deviation.
    Precondition: a logging file must have been opened
    :param C: the size of the hidden states' alphabet
    :param A: the size of the edges alphabet
    :param Lprec: an array with a subset of preceding layers to consider. For example, np.array([1,2,3])
    :param layer: the layer associated to the fingerprint
    :param runs: how many runs have been done for this particular configuration
    :param unibigram: whether the fingerprint has used unigrams+bigrams or not
    :param concatenate_fingerprints: the boolean decision associated to the fingerprint
    :param fingerprint_name: the initial name of the fingerprint
    :return:
    '''
    np.random.seed()

    tr_runs = [0. for _ in range(0, runs)]
    vl_runs = [0. for _ in range(0, runs)]

    for run in range(0, runs):

        with open("./fingerprints/" + fingerprint_name + '_' + str(run) + '_' + str(C) + '_' + str(layer) + '_' +
                          str(Lprec) + '_' + str(concatenate_fingerprints), 'rb') as f:

            [unigram_train, unigram_valid, allStates_train, allStates_valid,
             adjacency_lists_train, adjacency_lists_valid, sizes_train, sizes_valid, Y_train,
             Y_valid] = pickle.load(f)

            # Passing the unigram etc avoids rec-omputing then
            tr_runs[run], vl_runs[run] = _compute_jaccard_accuracy(None, A, C, unibigram, None, Y_train,
                                      adjacency_lists_train,
                                      sizes_train, None, Y_valid, adjacency_lists_valid,
                                      sizes_valid, concatenate_fingerprints, unigram_train,
                                      unigram_valid, allStates_train, allStates_valid)

    tr_acc = np.average(tr_runs)
    tr_std = np.std(tr_runs)
    vl_acc = np.average(vl_runs)
    vl_std = np.std(vl_runs)

    log = fingerprint_name + ' ' + str(C) + ' ' + str(layer) + ' ' + str(Lprec) + ' ' + \
          str(concatenate_fingerprints) + ' ' + str(unibigram) + \
          ' ' + str(tr_acc) + ' ' + str(tr_std) + ' ' + str(vl_acc) + ' ' + str(vl_std)

    logging.info(log)


def _helper_kfold_ms(k, folds, pool_dim, svmCs, gammas, C, K, A, Lprec, graphs,
                     layers, threshold, max_epochs, concatenate_fingerprints, useSVM, name):

    np.random.seed()

    set_dim = len(graphs)
    fold_dim = math.floor(set_dim / folds)

    # Build the folds
    val_k = graphs[k * fold_dim + 1:(k + 1) * fold_dim]

    if k == 0:
        train_k = graphs[(k + 1) * fold_dim + 1:]
    elif k == folds - 1:
        train_k = graphs[1:k * fold_dim]
    else:
        train_k = graphs[1:k * fold_dim] + graphs[(k + 1) * fold_dim + 1:]

    X_train, Y_train, adjacency_lists_train, train_sizes_k = unravel(train_k, one_target_per_graph=True)
    X_valid, Y_valid, adjacency_lists_valid, val_sizes_k = unravel(val_k, one_target_per_graph=True)

    architecture = None
    prev_statistics = None
    last_states = None

    for layer in layers:

        best_arch, best_prev_statistics, best_last_states = None, None, None
        best_vl_acc = 0.

        for p in range(0, pool_dim):

            # Estimate the expected risk over this val_k fold after training

            if architecture is None:
                arch = None
            else:
                arch = list(architecture)
                # list() needed because incremental training modifies the list passed as argument

            new_architecture, new_prevStats, new_lastStates = \
                incremental_training(C, K, A, Lprec, adjacency_lists_train, X_train,
                                     layers=layer, threshold=threshold, max_epochs=max_epochs,
                                     architecture=arch, prev_statistics=prev_statistics,
                                     last_states=last_states)

            if useSVM:
                for svmC in svmCs:
                    for gamma in gammas:

                        tr_a, vl_acc = _compute_svm_accuracy(new_architecture, C, svmC, gamma, X_train, Y_train,
                                                             adjacency_lists_train, train_sizes_k, X_valid, Y_valid,
                                                             adjacency_lists_valid, val_sizes_k,
                                                             concatenate_fingerprints)
                        if vl_acc > best_vl_acc:
                            best_vl_acc = vl_acc
                            best_arch = new_architecture
                            best_last_states = new_lastStates
                            best_prev_statistics = new_prevStats
                            tr_acc = tr_a
                            best_svmC = svmC
                            best_gamma = gamma
                            print("Best accuracy of the pool on vl set is ", vl_acc)

            else:
                for unibigram in [False, True]:

                    tr_a, vl_acc = _compute_jaccard_accuracy(new_architecture, A, C, unibigram, X_train, Y_train,
                                                             adjacency_lists_train,
                                                             train_sizes_k, X_valid, Y_valid, adjacency_lists_valid,
                                                             val_sizes_k, concatenate_fingerprints)
                    if vl_acc > best_vl_acc:
                        best_vl_acc = vl_acc
                        best_arch = new_architecture
                        best_last_states = new_lastStates
                        best_prev_statistics = new_prevStats
                        tr_acc = tr_a
                        best_unibigram = unibigram
                        print("Best accuracy of the pool on vl set is ", vl_acc)

        prev_statistics = best_prev_statistics
        last_states = best_last_states
        architecture = best_arch

        unigram_train, allStates = compute_input_matrix(best_arch, C, X_train, adjacency_lists_train,
                                                        train_sizes_k, concatenate=concatenate_fingerprints,
                                                        return_all_states=True)
        unigram_valid, allStates_valid = compute_input_matrix(best_arch, C, X_valid, adjacency_lists_valid,
                                                              val_sizes_k,
                                                              concatenate=concatenate_fingerprints,
                                                              return_all_states=True)

        lock.acquire()

        try:
            # Store the fingeprints computed by a particular configuration on a particular fold.
            # NOTE: We have to select the configuration that works best (on average) over all folds.
            if useSVM:
                with open("fingerprints/" + name + '_' + str(k) + '_' + str(C) + '_' + str(layer) + '_' +
                                  str(Lprec) + '_' + str(concatenate_fingerprints), 'wb') as f:
                    pickle.dump([unigram_train, unigram_valid, allStates, allStates_valid, adjacency_lists_train,
                                 adjacency_lists_valid, train_sizes_k, val_sizes_k, Y_train, Y_valid], f)
            else:
                with open("fingerprints/" + name + '_' + str(k) + '_' + str(C) + '_' + str(layer) + '_' +
                                  str(Lprec) + '_' + str(concatenate_fingerprints), 'wb') as f:
                    pickle.dump([unigram_train, unigram_valid, allStates, allStates_valid, adjacency_lists_train,
                                 adjacency_lists_valid, train_sizes_k, val_sizes_k, Y_train, Y_valid], f)
        except Exception as e:
            print(e)

        lock.release()


def k_fold_model_selection(folds, pool_dim, svmCs, gammas, C, K, A, Lprec, graphs, layers, threshold, max_epochs,
                           concatenate_fingerprints, useSVM, name):
    '''
    Trains a particular configuration of a CGMM over multiple folds, using a pooling technique.
    useSVM=True uses the svm parameters to select the best of the pool. useSVM=False uses the Jaccard.
    Stores the fingerprints produced by the final architecture, for each layer in layers.
    They can be used later to find the best performing model across all folds.
    :param folds: the number of folds
    :param pool_dim: pool dimension to implement a pooling technique (todo parallelise even that)
    :param svmCs: a list of SVM's C parameters to try
    :param gammas: a list of SVM's C parameters to try
    :param C: the size of the hidden states' alphabet
    :param K: the size of the emission alphabet
    :param A: the size of the edges' alphabet
    :param Lprec:
    :param graphs: the dataset
    :param layers: a list of depths for which to store the fingerprints
    :param threshold: the threshold for the EM algorithm
    :param max_epochs: maximum number of epochs per training
    :param concatenate_fingerprints: whether to concatenate the fingerprints
    :param useSVM: if False, try both unibigram=False and True
    :param name: the initial name for the fingerprints
    :return:
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=folds) as pool:
        # Start k_fold for this configuration
        for k in range(0, folds):
            pool.submit(_helper_kfold_ms, k, folds, pool_dim, svmCs, gammas, C, K, A, Lprec, graphs,
                        layers, threshold, max_epochs, concatenate_fingerprints, useSVM, name)


# ------------------- MODEL ASSESSMENT THROUGH DOUBLE CROSS VALIDATION ------------------------ #
def inner_kfold_cross_validation(folds, pool_dim, Cs, svmCs, gammas, M, A, Lprec, graphs,
           layers, appends, unibigrams, threshold, max_epochs):

    # Returns the best hyperparameters
    np.random.seed()
    set_dim = len(graphs)

    fold_dim = math.floor(set_dim / folds)

    averages_SVM = np.zeros((len(Cs), len(svmCs), len(gammas), len(appends), len(layers)))
    averages_Jaccard = np.zeros((len(Cs), len(unibigrams), len(appends), len(layers)))

    for inner_k in range(0, folds):

        # Build the folds
        val_k = graphs[inner_k * fold_dim + 1:(inner_k + 1) * fold_dim]

        if inner_k == 0:
            train_k = graphs[(inner_k + 1) * fold_dim + 1:]
        elif inner_k == folds - 1:
            train_k = graphs[1:inner_k * fold_dim]
        else:
            train_k = graphs[1:inner_k * fold_dim] + graphs[(inner_k + 1) * fold_dim + 1:]

        X_train, Y_train, adjacency_lists_train, train_sizes_k = unravel(train_k, one_target_per_graph=True)
        X_valid, Y_valid, adjacency_lists_valid, val_sizes_k = unravel(val_k, one_target_per_graph=True)

        for C in Cs:
            for append in appends:

                architecture = None
                prevStats = None
                lastStates = None

                for layer in layers:
                    best_arch, best_prevStats, best_lastStates = None, None, None
                    best_pool_vl_acc = 0.

                    for p in range(0, pool_dim):
                        # Estimate the expected risk over this val_k fold after training

                        # needed because incremental training modifies the list passed as argument
                        if architecture is None:
                            arch = None
                        else:
                            arch = list(architecture)

                        new_architecture, new_prevStats, new_lastStates = incremental_training(C, M, A, Lprec,
                                                                                               adjacency_lists_train, X_train,
                                                                                               layers=layer, threshold=threshold,
                                                                                               max_epochs=max_epochs,
                                                                                               architecture=arch,
                                                                                               prev_statistics=prevStats,
                                                                                               last_states=lastStates)

                        # Try rbf hyper-params
                        for svmC in svmCs:
                            for gamma in gammas:

                                tr_a, vl_acc = _compute_svm_accuracy(new_architecture, C, svmC, gamma, X_train, Y_train,
                                                                     adjacency_lists_train, train_sizes_k, X_valid,
                                                                     Y_valid, adjacency_lists_valid, val_sizes_k,
                                                                     append)
                                if vl_acc > best_pool_vl_acc:
                                    best_pool_vl_acc = vl_acc
                                    best_arch = new_architecture
                                    best_lastStates = new_lastStates
                                    best_prevStats = new_prevStats
                                    print("SVM New best in the pool with acc ", vl_acc)

                        for unibigram in unibigrams:
                            tr_a, vl_acc = _compute_jaccard_accuracy(new_architecture, A, C, unibigram, X_train,
                                                                     Y_train,
                                                                     adjacency_lists_train,
                                                                     train_sizes_k, X_valid, Y_valid,
                                                                     adjacency_lists_valid,
                                                                     val_sizes_k, append)
                            if vl_acc > best_pool_vl_acc:
                                best_pool_vl_acc = vl_acc
                                best_arch = new_architecture
                                best_lastStates = new_lastStates
                                best_prevStats = new_prevStats
                                print("JACCARD New best in the pool with acc ", vl_acc)

                    prevStats = best_prevStats
                    lastStates = best_lastStates
                    architecture = best_arch

                    # Try rbf hyperparams
                    for svmC in svmCs:
                        for gamma in gammas:
                            tr_a, vl_acc = _compute_svm_accuracy(architecture, C, svmC, gamma, X_train, Y_train,
                                                                 adjacency_lists_train, train_sizes_k, X_valid,
                                                                 Y_valid, adjacency_lists_valid, val_sizes_k,
                                                                 append)

                            averages_SVM[Cs.index(C), svmCs.index(svmC), gammas.index(gamma), appends.index(append),
                            layers.index(layer)] += vl_acc

                    for unibigram in unibigrams:

                        tr_a, vl_acc = _compute_jaccard_accuracy(architecture, A, C, unibigram, X_train,
                                                                 Y_train,
                                                                 adjacency_lists_train,
                                                                 train_sizes_k, X_valid, Y_valid,
                                                                 adjacency_lists_valid,
                                                                 val_sizes_k, append)

                        averages_Jaccard[Cs.index(C), unibigrams.index(unibigram), appends.index(append),
                                layers.index(layer)] += vl_acc

    # Now choose the best model and return the best parameters of the inner k fold
    averages_SVM = averages_SVM/folds
    averages_Jaccard = averages_Jaccard / folds

    best_vl_acc = 0.
    for C in Cs:
        for append in appends:
            for layer in layers:
                for svmC in svmCs:
                    for gamma in gammas:
                        acc = averages_SVM[Cs.index(C), svmCs.index(svmC), gammas.index(gamma), appends.index(append),
                                      layers.index(layer)]
                        if acc > best_vl_acc:
                            best_vl_acc = acc
                            useSVM = True
                            bestC = C
                            bestSvmC = svmC
                            bestGamma = gamma
                            bestLayers = layer
                            bestAppend = append
                            bestUnibigram = None

                for unibigram in unibigrams:
                    acc = averages_Jaccard[Cs.index(C), unibigrams.index(unibigram), appends.index(append),
                                      layers.index(layer)]
                    if acc > best_vl_acc:
                        best_vl_acc = acc
                        useSVM = False
                        bestC = C
                        bestSvmC = None
                        bestGamma = None
                        bestLayers = layer
                        bestAppend = append
                        bestUnibigram = unibigram

    return bestC, useSVM, bestSvmC, bestGamma, bestUnibigram, bestLayers, bestAppend


def _helper_double_cv(folds, outer_fold, pool_dim, Cs, svmCs, gammas, K, A, Lprec, train_k, val_k,
           layers, appends, unibigrams, threshold, max_epochs, name):

    bestC, useSVM, bestSvmC, bestGamma, unibigram, bestlayers, append = \
        inner_kfold_cross_validation(folds, pool_dim, Cs, svmCs, gammas, K, A, Lprec, train_k, layers, appends, unibigrams,
                                     threshold, max_epochs)

    with open(name + str(outer_fold), 'wb') as f:
        pickle.dump([train_k, val_k, bestC, useSVM, bestSvmC, bestGamma, unibigram, bestlayers, append], f)

    # NOW TRAIN AGAIN ON TRAIN FOLD AND COMPUTE RISK ON OUTER FOLD
    # TRAIN USING POOLING
    with open(str(name) + str(outer_fold), 'rb') as f:
        train_k, test_k, \
        bestC, useSVM, bestSvmC, bestGamma, unibigram, bestlayers, append = pickle.load(f)

    print("BEST PARAMETERS FOR FOLD ", outer_fold, ": ", bestC, useSVM, bestSvmC, bestGamma, unibigram, bestlayers, append)
    X_test, Y_test, adjacency_lists_test, test_sizes = unravel(test_k, one_target_per_graph=True)

    train_perc = 0.9
    valid_perc = 0.1
    test_perc = 0.0

    # TRAIN_K is the outer training fold
    graphs_train, graphs_valid, _ = \
        split_dataset(train_k, train_perc, valid_perc, test_perc, shuffle=False)

    X_train, Y_train, adjacency_lists_train, train_sizes = unravel(graphs_train, one_target_per_graph=True)
    X_valid, Y_valid, adjacency_lists_valid, valid_sizes = unravel(graphs_valid, one_target_per_graph=True)

    architecture = None
    prevStats = None
    lastStates = None

    for layer in range(1, bestlayers + 1):

        # POOL TECHNIQUE
        archsPool = [None for _ in range(0, pool_dim)]
        prev_statsPool = [None for _ in range(0, pool_dim)]
        last_statesPool = [None for _ in range(0, pool_dim)]
        vl_accPool = np.zeros(pool_dim)

        for p in range(0, pool_dim):
            # Estimate the expected risk over this val_k fold after training
            # needed because incremental training modifies the list passed as argument
            if architecture is None:
                arch = None
            else:
                arch = list(architecture)

            new_architecture, new_prevStats, new_lastStates = incremental_training(bestC, K, A, Lprec,
                                                                                   adjacency_lists_train, X_train,
                                                                                   layers=layer, threshold=0.,
                                                                                   max_epochs=max_epochs,
                                                                                   architecture=arch,
                                                                                   prev_statistics=prevStats,
                                                                                   last_states=lastStates)

            if useSVM:
                tr_acc, vl_acc = _compute_svm_accuracy(new_architecture, bestC, bestSvmC, bestGamma, X_train, Y_train,
                                                       adjacency_lists_train,
                                                       train_sizes, X_valid, Y_valid, adjacency_lists_valid,
                                                       valid_sizes,
                                                       compute_accuracy,
                                                       append)
            else:
                tr_acc, vl_acc = _compute_jaccard_accuracy(new_architecture, A, bestC, unibigram, X_train, Y_train,
                                                           adjacency_lists_train,
                                                           train_sizes, X_valid, Y_valid, adjacency_lists_valid,
                                                           valid_sizes,
                                                           append)

            vl_accPool[p] = vl_acc
            archsPool[p] = new_architecture
            last_statesPool[p] = new_lastStates
            prev_statsPool[p] = new_prevStats

        # Take the idx relative to the best performing model
        idx = np.argmax(vl_accPool)

        architecture = archsPool[idx]
        prevStats = prev_statsPool[idx]
        lastStates = last_statesPool[idx]

    # Compute accuracy on outer fold
    if useSVM:
        _, test_acc = _compute_svm_accuracy(architecture, bestC, bestSvmC, bestGamma, X_train, Y_train,
                                            adjacency_lists_train,
                                            train_sizes, X_test, Y_test, adjacency_lists_test, test_sizes,
                                            compute_accuracy,
                                            append)
    else:
        _, test_acc = _compute_jaccard_accuracy(architecture, A, bestC, unibigram, X_train, Y_train,
                                                adjacency_lists_train,
                                                train_sizes, X_test, Y_test, adjacency_lists_test,
                                                test_sizes, append)

    logging.info("Test acc for outer fold " + str(outer_fold) + " is " + str(test_acc))



def double_cross_validation(folds, inner_folds, pool_dim, Cs, svmCs, gammas, K, A, Lprec, graphs,
           layers, appends, unibigrams, threshold, max_epochs, name):
    '''
    Performs robust risk assessment launching one process per outer fold (can be further parallelised).
    The pooling strategy used both the RBF and Jaccard Kernels to select the best one.
    This procedure does not return a model, but an estimate.
    Notice that this procedure does stores results of the inner cross validations.
    :param folds: the number of outer folds
    :param inner_folds: the number of inner folds
    :param pool_dim: the dimension of the pool.
    :param Cs: a list of C values to try in the inner k-fold for model selection
    :param svmCs: a list of SVM's C values to try in the inner k-fold for model selection
    :param gammas: a list of SVM's gamma values to try in the inner k-fold for model selection
    :param K: the size of the emission alphabet
    :param A: the size of the edges' alphabet
    :param Lprec:
    :param graphs: the dataset
    :param layers: a list of layers to try in the inner k-fold for model selection
    :param appends: a list (at most [False,True]) which states whether to try single fingerprints or concatenations
    :param unibigrams: same for appends, but it states whether to try only unigram, bigram, both or None.
    :param threshold: threshold for EM algorithm
    :param max_epochs: maximum number of epochs for a training step
    :param name: the name of the log file which will be created.
    :return:
    '''

    np.random.seed()

    outer_set_dim = len(graphs)
    outer_fold_dim = math.floor(outer_set_dim / folds)

    logging.basicConfig(
        filename=str(name) + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + '.log',
        level=logging.DEBUG, filemode='a')

    with concurrent.futures.ProcessPoolExecutor(max_workers=folds) as pool:

        # Start OUTER k_fold for this configuration
        for outer_k in range(0, folds):

            # Build the folds
            val_k = graphs[outer_k * outer_fold_dim + 1:(outer_k + 1) * outer_fold_dim]

            if outer_k == 0:
                train_k = graphs[(outer_k+1) * outer_fold_dim + 1:]
            elif outer_k == folds-1:
                train_k = graphs[1:outer_k * outer_fold_dim]
            else:
                train_k = graphs[1:outer_k * outer_fold_dim] + graphs[(outer_k+1) * outer_fold_dim + 1:]

            pool.submit(_helper_double_cv, inner_folds, outer_k, pool_dim, Cs, svmCs, gammas, K, A, Lprec, train_k, val_k,
                        layers, appends, unibigrams, threshold, max_epochs, name)

    pool.shutdown(wait=True)

