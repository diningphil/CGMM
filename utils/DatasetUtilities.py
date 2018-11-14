import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, TensorDataset


class ZipDataset(TensorDataset):

    def __init__(self, tensors, dim_to_stack=0):
        '''
        for tensor in tensors:
            print(tensor.size())
        '''
        self.tensors = [tensor for tensor in tensors]
        self.dim_to_stack = dim_to_stack

    def __getitem__(self, index):
        return torch.stack([tensor[index] for tensor in self.tensors], dim=self.dim_to_stack)

    def __len__(self):
        return sum([tensor.size()[0] for tensor in self.tensors])

    def _get_data(self):
        return self.tensors


class LabelAndStatsDataset(Dataset):

    def __init__(self, label_dataset, stats_dataset=None):
        self.label_dataset = label_dataset
        self.stats_dataset = stats_dataset

    def __getitem__(self, index):
        return (self.label_dataset[index])[0], self.stats_dataset[index] if self.stats_dataset is not None\
            else (self.label_dataset[index])[0]

    def __len__(self):
        return len(self.label_dataset) # NOTICE THAT WE ASSUME IT IS EQUAL TO STATS_DATASET WHEN PRESENT


def concat_graph_fingerprints(zipdataset):
    zipdataset = zipdataset._get_data()
    # CONCAT UNIGRAM FINGERPRINTS
    return np.concatenate(tuple([zipdataset[i] for i in range(len(zipdataset))]), axis=1)


def save_tensor(tensor, exp_name, folder_name, filename, layer_no):
    '''
    :param tensor:
    :param filename:
    :param folder_name:
    :param layer_np:
    :return:
    '''
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    if not os.path.exists(os.path.join(exp_name, folder_name)):
        os.makedirs(os.path.join(exp_name, folder_name))

    filepath = os.path.join(exp_name, folder_name, filename) + '_' + str(layer_no)

    # print("Saving to", filepath)
    np.save(filepath, tensor, allow_pickle=False)


def save_statistics(cgmm_layer, adjacency_lists, inferred_states, exp_name, folder_name, filename, layer_no,
                    add_self_arc=False):
    '''
    :param adjacency_lists:
    :param inferred_states:
    :param A:
    :param C:
    :param filename:
    :param layer_no:
    :param radius:
    :return:
    '''

    # TODO YOU CAN ALSO ARRANGE EACH GRAPH INTO A NPZ AND LOAD THEM SEPARATELY. THEN YOU JUST DO SOME COMPUTATION
    # WHEN BATCHING STATISTICS

    statistics = cgmm_layer.compute_statistics(adjacency_lists, inferred_states, add_self_arc)

    save_tensor(statistics, exp_name, folder_name, filename, layer_no)

    return statistics

def load_unigrams_or_statistics(exp_name, folder_name, filename, layers):
    '''
    This function creates the dataset by reading from different files according to "layers".
    :param filename:
    :param layers:
    :return: A Dataset, where each element has shape [L, A, C2]. The first dimension is obtained by merging
    the L required files specified by the "layers" argument
    '''

    L = len(layers)

    filepaths = [os.path.join(exp_name, folder_name, filename) + '_' + str(layer) + '.npy' for layer in layers]

    files = [torch.from_numpy(np.array(np.load(filepath, mmap_mode='r'))) for filepath in filepaths]

    dataset = ZipDataset(files, dim_to_stack=0)

    return dataset


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

