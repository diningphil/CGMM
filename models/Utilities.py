import numpy as np
import os
import torch
from torch.utils.data import Dataset, TensorDataset


class ZipDataset(TensorDataset):

    def __init__(self, tensors, dim_to_stack=0):
        """
        for tensor in tensors:
            print(tensor.size())
        """
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
    return np.concatenate(tuple([zipdataset[i] for i in range(len(zipdataset))]), axis=1)


def save_tensor(tensor, folderpath, filename, layer_no):
    """
    :param tensor:
    :param filename:
    :param folder_name:
    :param layer_np:
    :return:
    """
    filepath = os.path.join(folderpath, filename) + '_' + str(layer_no)

    # print("Saving to", filepath)
    np.save(filepath, tensor, allow_pickle=False)


def save_statistics(model, adjacency_lists, node_states, edge_states, sizes, max_arieties,
                    folderpath, filename, layer_no, radius=1, add_self_arc=False):
    """
    :param adjacency_lists:
    :param node_states:
    :param edge_states:
    :param folderpath:
    :param filename:
    :param layer_no:
    :param radius:
    :return:
    """

    stats_node, stats_edge = model.compute_statistics(adjacency_lists, node_states, edge_states, sizes, max_arieties,
                                                      add_self_arc)

    save_tensor(stats_node, folderpath, filename + '_node', layer_no)
    if stats_edge is not None:
        save_tensor(stats_edge, folderpath, filename + '_edge', layer_no)

    return stats_node, stats_edge


def load_to_ZipDataset(filepaths):
    """
    This function creates the dataset by reading from different files according to "layers".
    :param filename:
    :return: A Dataset, where each element has shape [L, A, C2]. The first dimension is obtained
    """
    files = [torch.from_numpy(np.array(np.load(filepath, mmap_mode='r'))) for filepath in filepaths]
    dataset = ZipDataset(files, dim_to_stack=0)
    return dataset


def unravel(graphs, one_target_per_graph=True):
    """
    Takes "graphs" and unravels them into a single big one, using the size of each graph to shift indices
    in the adjacency lists.
    This is required to give the dataset to the model, and it should be done after having split between
    training, validation and test.
    The method returns a new copy of adjacency_lists, so that the original graph is not modified.
    :param graphs: a list of tuples (X_values, Y_values, adjacency_lists, size), each representing a graph.
    Each X_values is a ndarray of size NxF, where N is the no of vertices and F the features
    :param one_target_per_graph: if true, Y will have one target per graph; if false,Y will have a target for each node
    :return: a single graph tuple: (X, Y, adjacency_lists, sizes). "sizes" is a list of all graphs' sizes
    """

    X_final = None
    edges_final = []
    Y_final = []
    adjacency_lists_final = []
    sizes = []
    max_arieties = []

    start_id = 0
    for (X, edges, Y, adjacency_lists, size, max_ariety) in graphs:

        if X_final is None:
            X_final = X
        else:
            X_final = np.concatenate((X_final, X), axis=0)

        edges_final.extend(edges)
        sizes.append(size)
        max_arieties.append(max_ariety)

        if one_target_per_graph:
            Y_final.append(Y[0])  # The class is equal for all nodes of the graph
        else:
            Y_final.extend(Y)

        adjacency_lists_shifted = [[u + start_id for u in l] for l in adjacency_lists]
        adjacency_lists_final.extend(adjacency_lists_shifted)

        start_id += size

    if len(X_final.shape) == 1:  # no of features is 1, add an axis to generalize implementation to higher dim features
        X_final = np.reshape(X_final, (-1, 1))
    return np.array(X_final), np.array(edges_final), np.array(Y_final), adjacency_lists_final, sizes, max_arieties

