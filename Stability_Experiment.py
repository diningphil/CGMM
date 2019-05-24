import os
import pickle
import torch

from model_selection import SimpleModelSelector
from risk_assessment.HoldOut import HoldOut
from utils.utils import inductive_split_dataset, shuffle_dataset
from experiments.Graph_Stability_Experiment import StabilityExperiment

# ---------- Hyper-Parameters ---------- #

# C and layers will be (possibly) be automatically determined
possible_CN = [20]
possible_CA = [None]

# This should be investigated but could be left to [1] in our experiments (keep complexity low?)
possible_use_statistics = [[1]]  # np.arange(layers)+1

# This can be searched as well
possible_max_epochs = [10]

# ---------- Construct possible configurations ---------- #

# These are the keys that my experiment will use
model_configurations = {
    'max_layers': [3],
    'threshold': [0.],
    'runs': [1],
    'unibigram': [False],
    'CN': possible_CN,
    'CA': possible_CA,
    'max_epochs': possible_max_epochs,
    'use_statistics': possible_use_statistics,
    'infer_with_posterior': [True],
    'add_self_arc': [False],
    'aggregation': ['sum'],
    'classifier': ['mlp'],
    'hidden_units': [8],
    'l2': [0],
    'learning_rate': [1e-3],
    'l_batch_size': [8],
    'training_epochs': [500],
    'early_stopping': [100],  # min no of epochs to begin using early stopping
    'model_class': ['CGMM'],

}

# ---------------------------------------#

# Needed to avoid thread spawning
torch.set_num_threads(1)

experiment_class = StabilityExperiment

max_processes = 2
debug = True

for task_name in ['AIDS']:
    print('Starting experiments on', task_name, '...')
    exp_path = os.path.join('Results', model_configurations['model_class'][0], task_name + '_stability')
    if os.path.exists(exp_path):
        raise Exception("Path", exp_path, "Already present, be careful!")

    with open(os.path.join('Datasets', task_name, task_name + '_dataset'), 'rb') as f:
        [graphs, A, K] = pickle.load(f)
        # Do not shuffle for now. Just do tests to compare different models
        shuffle_dataset(graphs)

    K = int(K)
    A = int(A)

    model_configurations['edge_type'] = ['discrete']
    model_configurations['K'] = [K]
    model_configurations['A'] = [A]

    if task_name in ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        model_configurations['node_type'] = ['continuous']
    else:
        model_configurations['node_type'] = ['discrete']

    model_selector = SimpleModelSelector.SimpleModelSelector(max_processes, train_perc=0.9)
    model_selector.model_selection(graphs, StabilityExperiment, exp_path, model_configurations,
                                   shuffle=False, debug=debug)
