import time
import shutil
from sklearn import svm
from torch.utils.data import DataLoader

from utils.DatasetUtilities import *
from CGMM_Layer import CGMM_Layer

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


def incremental_training(C, K, A, use_statistics, adjacency_lists, label_dataset, layers, exp_name,
                         threshold=0, max_epochs=100, save_name=None):
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
    stats_folder = 'train_statistics'
    stats_filename = 'stats'


    architecture = []

    for layer in range(layers):
        print("Training Layer", layer+1)
        # e.g 1 - [1, 3] = [0, -2] --> [0]
        # e.g 5 - [1, 3] = [4, 2]  --> [4, 2]
        layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

        L = len(layer_wise_statistics)

        cgmm_layer = CGMM_Layer(K, C, current_layer=(layer + 1)) if layer == 0 \
            else CGMM_Layer(K, C, C, L, A, current_layer=(layer + 1))

        if cgmm_layer.is_layer_0:
            stats = None
        else:
            stats = load_unigrams_or_statistics(exp_name, stats_folder, stats_filename, layer_wise_statistics)

        train_set = LabelAndStatsDataset(label_dataset, stats)
        train_set = DataLoader(train_set, batch_size=2000, shuffle=False, num_workers=2)

        old_likelihood = -np.inf
        for epoch in range(max_epochs):

            cgmm_layer.init_accumulators()

            likelihood = 0.
            for input_batch in train_set:
                likelihood += cgmm_layer.EM_step(*input_batch).detach().numpy()

            # If you want to perform a batch update of parameters use this method
            cgmm_layer.update_parameters()

            print('Likelihood at epoch', epoch+1, ':', likelihood)

            if likelihood - old_likelihood <= threshold:
                break
            else:
                old_likelihood = likelihood

        # Perform inference at the end of training and accumulate results
        predictions = None

        for input_batch in train_set:

            inferred_batch = cgmm_layer(*input_batch)
            inferred_batch = inferred_batch.detach().numpy()

            if predictions is None:
                predictions = inferred_batch
            else:
                predictions = np.append(predictions, inferred_batch)

        if save_name is not None:
            print("NOT SAVING THE MODEL YET!")
            # TODO UPDATE DICT TO SAVE (LIST OF LAYERS and PARAMETERS (IN A DICTIONARY)
            # saver = tf.train.Saver(variables_to_save)
            # print("Model saved in", saver.save(sess, os.path.join(checkpoint_folder, save_name, 'model.ckpt')))

        save_statistics(adjacency_lists, predictions, A, C, exp_name, stats_folder, stats_filename, layer)

        architecture.append(cgmm_layer)

    # Clear the statistics files (no longer necessary)
    for layer_no in range(0, layers):
        shutil.rmtree(os.path.join(exp_name, stats_folder), ignore_errors=True)


    if save_name is not None:

        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        if not os.path.exists(os.path.join(checkpoint_folder, save_name)):
            os.makedirs(os.path.join(checkpoint_folder, save_name))

        # TODO SAVE THE ARCHITECTURE

    return architecture

def incremental_inference(K, A, C, layers, use_statistics, label_dataset, adjacency_lists, sizes,
                        exp_name, architecture=None):
    '''
    Performs inference throughout the architecture. Assumes C = C2
    '''
    unigram_folder = 'unigrams_per_layer'
    unigram_filename = 'unigrams'

    stats_folder = 'stats_tmp'
    stats_filename = 'tmp_stats'


    if architecture is None:
        raise NotImplementedError('Still to be implemented!')
        # TODO RESTORE THE ARCHITECTURE (LIST OF LAYERS and PARAMETERS (IN A DICTIONARY)

    for layer in range(0, layers):
        cgmm_layer = architecture[layer]

        print("Inference: layer", layer)

        predictions = None

        layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

        if cgmm_layer.is_layer_0:
            stats = None
        else:
            stats = load_unigrams_or_statistics(exp_name, stats_folder, stats_filename, layer_wise_statistics)

        train_set = LabelAndStatsDataset(label_dataset, stats)
        train_set = DataLoader(train_set, batch_size=2000, shuffle=False, num_workers=2)

        for input_batch in train_set:

            inferred_batch = cgmm_layer(*input_batch)
            inferred_batch = inferred_batch.detach().numpy()

            if predictions is None:
                predictions = inferred_batch
            else:
                predictions = np.append(predictions, inferred_batch)

        save_statistics(adjacency_lists, predictions, A, C, exp_name, stats_folder, stats_filename, layer)
        save_tensor(_aggregate_states(C, sizes, predictions), exp_name, unigram_folder, unigram_filename, layer)

    # Clear the statistics files (no longer necessary)
    for layer_no in range(0, layers):
        shutil.rmtree(os.path.join(exp_name, stats_folder), ignore_errors=True)


def compute_accuracy(predictions, ground_truth):
    assert len(predictions) == len(ground_truth)
    return 100*(np.sum(ground_truth == predictions) / len(predictions))


def compute_svm_accuracy(unigrams, Y, svmC, gamma, unigrams_valid=None, Y_valid=None):

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
