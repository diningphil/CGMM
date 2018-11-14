import time
import shutil
from sklearn import svm
from torch.utils.data import DataLoader

from utils.DatasetUtilities import *
from CGMM_Layer import CGMM_Layer

checkpoint_folder = './checkpoints'


def current_milli_time():
    return int(round(time.time() * 1000))


def _aggregate_states(C, adj_lists, sizes, states, statistics=None, aggregate_posteriors=False, bigram=False):
    '''
    Aggregates the states into a frequency vector
    :param C: the size of the hidden states' alphabet
    :param sizes: a list of integers, each of them representing the size of a graph
    :param states: the array of states produced for all the graphs or a matrix UxC with the posterior for each node
    :return: The frequency state vector of dimension no_graphsxC if unibigram is False, else no_graphsx(3C)
    where you have concatenated both the unigram and the bigram representation
    '''
    freq_aggregated = []

    curr_size = 0
    for size in sizes:
        freq_unigram = np.zeros(C)
        if unibigram:
            freq_bigram = np.zeros(C*C)
        if not aggregate_posteriors:
            # ---------- infer with argmax ---------- #

            np.add.at(freq_unigram, states[curr_size:(curr_size + size)], 1)
            freq_unigram = np.divide(freq_unigram, size)  # Normalize

            if unibigram:
                assert statistics is not None
                # Update the feature matrix for each structure
                for u in range(curr_size, curr_size + size):
                    # info about A is not exploited
                    freq_bigram[states[u] * C: states[u] * C + C] += np.sum(statistics[u, :, :], axis=0)

                freq_bigram = np.divide(freq_bigram, size)  # Normalize
        else:
            # ---------- infer with posterior ---------- #

            freq_unigram = np.sum(states[curr_size:(curr_size + size)], axis=0)
            freq_unigram = np.divide(freq_unigram, size)  # Normalize

            if unibigram:
                assert statistics is not None
                # Update the feature matrix for each structure
                for u in range(curr_size, curr_size + size):
                    state_u = np.argmax(states[u])

                    # info about A is not exploited
                    for neighbour, a in adj_lists[u]:
                        freq_bigram[state_u * C: state_u * C + C] += states[neighbour, :]

                freq_bigram = np.divide(freq_bigram, size)  # Normalize

        freq_aggregated.append(np.concatenate((freq_unigram, freq_bigram)))
        curr_size += size

    return np.array(freq_aggregated)


def bigram(all_states, adjacency_lists, sizes, A, C):
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


def incremental_training(C, K, A, use_statistics, adjacency_lists, label_dataset, layers,
                         radius, add_self_arc, exp_name, threshold=0, max_epochs=100, save_name=None):
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
    :return: an architecture
    '''
    stats_folder = 'train_statistics'
    stats_filename = 'stats'

    architecture = []

    architecture_dict = {}

    for layer in range(layers):

        print("Training Layer", layer+1)
        # e.g 1 - [1, 3] = [0, -2] --> [0]
        # e.g 5 - [1, 3] = [4, 2]  --> [4, 2]
        layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

        L = len(layer_wise_statistics)

        # TODO GENERALIZE
        C2 = C

        cgmm_layer = CGMM_Layer(K, C, A, radius=radius) if layer == 0 \
            else CGMM_Layer(K, C, A, C2, L, radius=radius)

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
        inferred_states = None

        for input_batch in train_set:

            posterior_batch = cgmm_layer(*input_batch)

            # Always used for statistics
            prediction_batch = torch.argmax(posterior_batch, dim=1).detach().numpy()

            if inferred_states is None:
                inferred_states = prediction_batch
            else:
                inferred_states = np.append(inferred_states, prediction_batch)

        save_statistics(cgmm_layer, adjacency_lists, inferred_states, exp_name, stats_folder, stats_filename,
                        layer, add_self_arc)

        if save_name is not None:
            architecture_dict[layer] = cgmm_layer.build_dict()
            architecture_dict[layer]['use_statistics'] = use_statistics  # it may be useful to remember

        architecture.append(cgmm_layer)

    # Clear the statistics files (no longer necessary)
    for layer_no in range(0, layers):
        shutil.rmtree(os.path.join(exp_name, stats_folder), ignore_errors=True)

    if save_name is not None:

        if not os.path.exists(exp_name):
            os.makedirs(exp_name)
        if not os.path.exists(os.path.join(exp_name, checkpoint_folder)):
            os.makedirs(os.path.join(exp_name, checkpoint_folder))

        torch.save(architecture_dict, os.path.join(exp_name, checkpoint_folder, save_name))

    return architecture


def incremental_inference(A, layers, use_statistics, label_dataset, adjacency_lists, sizes,
                        infer_with_posterior, bigram, add_self_arc,
                          exp_name, tr_val_test, architecture=None, save_name=None):
    '''
    Performs inference throughout the architecture. Assumes C = C2 for the moment
    '''
    unigram_folder = 'unigrams_layer_' + tr_val_test
    unigram_filename = 'unigrams'

    stats_folder = 'stats_tmp'
    stats_filename = 'tmp_stats'

    # Clear the (possibly previous) unigrams
    for layer_no in range(0, layers):
        shutil.rmtree(os.path.join(exp_name, unigram_folder), ignore_errors=True)

    if architecture is None:
        ckpt = torch.load(os.path.join(exp_name, checkpoint_folder, save_name))
        architecture = CGMM_Layer.build_architecture(ckpt)

    for layer in range(0, layers):
        cgmm_layer = architecture[layer]

        print("Inference: layer", layer+1)

        inferred_states = None
        posteriors = None

        layer_wise_statistics = [(layer - x) for x in use_statistics if (layer - x) >= 0]

        if cgmm_layer.is_layer_0:
            stats = None
        else:
            stats = load_unigrams_or_statistics(exp_name, stats_folder, stats_filename, layer_wise_statistics)

        train_set = LabelAndStatsDataset(label_dataset, stats)
        train_set = DataLoader(train_set, batch_size=2000, shuffle=False, num_workers=2)

        for input_batch in train_set:

            posterior_batch = cgmm_layer(*input_batch)

            # Always used for statistics
            prediction_batch = torch.argmax(posterior_batch, dim=1).detach().numpy()

            if inferred_states is None and posteriors is None:
                inferred_states = prediction_batch
            else:
                inferred_states = np.append(inferred_states, prediction_batch)

            if posteriors is None and infer_with_posterior:
                posteriors = posterior_batch.detach().numpy()
            elif infer_with_posterior:
                posteriors = np.concatenate((posteriors, posterior_batch.detach().numpy()), axis=0)

        statistics = save_statistics(cgmm_layer, adjacency_lists, inferred_states, exp_name, stats_folder,
                                     stats_filename, layer, add_self_arc)

        save_tensor(_aggregate_states(cgmm_layer.C, adjacency_lists, sizes,
                                      posteriors if infer_with_posterior else inferred_states,
                                      statistics=statistics if bigram else None,
                                      aggregate_posteriors=infer_with_posterior, bigram=bigram),
                    exp_name, unigram_folder, unigram_filename, layer)

    # Clear the statistics files (no longer necessary)
    for layer_no in range(0, layers):
        shutil.rmtree(os.path.join(exp_name, stats_folder), ignore_errors=True)


def compute_accuracy(predictions, ground_truth):
    assert len(predictions) == len(ground_truth)
    return 100*(np.sum(ground_truth == predictions) / len(predictions))


def compute_svm_accuracy(input, Y, svmC, gamma, input_valid=None, Y_valid=None):

    # SVC performs a AVA approach for multiclass classification
    if gamma is None:
        clf = svm.SVC(C=svmC, kernel='linear', shrinking=False)
    else:
        clf = svm.SVC(C=svmC, kernel='rbf', gamma=gamma, shrinking=False)

    # Train on train set
    clf.fit(input, Y)

    # Compute train accuracy
    tr_acc = compute_accuracy(clf.predict(input), Y)

    vl_acc = -1
    if input_valid is not None and Y_valid is not None:
        vl_acc = compute_accuracy(clf.predict(input_valid), Y_valid)

    return tr_acc, vl_acc
