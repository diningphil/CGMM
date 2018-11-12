import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader

from utils.TrainingUtilities import *
from utils.DatasetUtilities import unravel

task_name = 'CPDB'

with open('Graph_Tasks/' + task_name + '_data/' + task_name + '_dataset', 'rb') as f:
    [graphs, A, K] = pickle.load(f)

X, Y, adjacency_lists, sizes = unravel(graphs, one_target_per_graph=True)

label_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))  # It will return a tuple

# Hyper-Parameters
C = 5
C2 = 5
# use_statistics = [1, 3]  # e.g use the layer-1 and layer-3 statistics
use_statistics = [1]
layers = 8  # How many layers you will train
max_epochs = 10

batch_size = 2000


exp_name = 'first_experiment'


# Training and inference phase
architecture = incremental_training(C, K, A, use_statistics, adjacency_lists, label_dataset, layers, exp_name,
                         threshold=0., max_epochs=max_epochs, save_name=None)

incremental_inference(K, A, C, layers, use_statistics, label_dataset, adjacency_lists, sizes,
                          exp_name, architecture=architecture)


# You should take care of information about A and C for each layer
unigrams_train_dataset = load_unigrams_or_statistics(exp_name, 'unigrams_per_layer', 'unigrams',
                                             layers=[i for i in range(layers)])

# Concatenate the fingerprints (e.g. C=10, then [?, 10], [?,10] becomes [?, 20]
fingerprints = concat_graph_fingerprints(unigrams_train_dataset)

tr_acc, vl_acc = compute_svm_accuracy(fingerprints, Y, 20, 5, unigrams_valid=None, Y_valid=None)
print(tr_acc, vl_acc)

'''
# FOR VALIDATION
#incremental_inference(save_name, K, A, C, layers, use_statistics, target_dataset_valid, adjacency_lists_valid, sizes_valid,
#                          unigram_inference_name_valid, statistics_inference_name)

# -------------------------------------------------------------------------------------------------------------


# FOR VALIDATION
#unigrams_valid = recover_unigrams(unigram_inference_name_train, layers=[0, 1, 2, 3, 4, 5, 6, 7],
#                            C=C, concatenate=True, return_numpy=True)


# vl_acc is -1 if you do not pass validation parameters
tr_acc, vl_acc = compute_svm_accuracy(unigrams_train, Y, 10, 5, unigrams_valid=None, Y_valid=None)
'''