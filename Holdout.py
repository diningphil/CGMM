import os
import pickle
import torch

from experiments.Graph_Trivial_Experiment import GraphTrivialExperiment
from model_selection.K_Fold import KFold
from model_selection.SimpleModelSelector import SimpleModelSelector
from risk_assessment.HoldOut import HoldOut
from risk_assessment.Nested_CV import NestedCV
from experiments.Graph_Assessment_Experiment import GraphAssessmentExperiment

# ---------- Hyper-Parameters ---------- #

# C and layers will be (possibly) be automatically determined
from utils.utils import shuffle_dataset

possible_CN = [10]
possible_CA = [None]

# This should be investigated but could be left to [1] in our experiments (keep complexity low?)
possible_use_statistics = [[1]]  # np.arange(layers)+1

# This can be searched as well
possible_max_epochs = [10]

# ---------- Construct possible configurations ---------- #

# These are the keys that my experiment will use
model_configurations = {
    'model_class': ['CGMM'],
    'max_layers': [10],
    'threshold': [0.],
    'CN': possible_CN,
    'CA': possible_CA,
    'max_epochs': possible_max_epochs,
    'use_statistics': possible_use_statistics,
    'add_self_arc': [False],
    'unibigram': [False],
    'infer_with_posterior': [True],
    'aggregation': ['sum'],
    'classifier': ['mlp'],
    'l2': [0],
    'learning_rate': [1e-3],
    'l_batch_size': [100],
    'training_epochs': [5000],
    'early_stopping': [4900],
    # This is redundant for logistic!
    'hidden_units': [128],
    'plot': [False]
}

# ---------------------------------------#

# Needed to avoid thread spawning
torch.set_num_threads(1)

experiment_class = GraphAssessmentExperiment

shuffle = True
debug = True
outer_folds = 10
outer_processes = 1
inner_processes = 2

train_perc = 0.9

for task_name in ['NCI1']:
    print('Starting experiments on', task_name, '...')
    exp_path = os.path.join('Results', model_configurations['model_class'][0], task_name + '_assessment')

    with open(os.path.join('Datasets', task_name, task_name + '_dataset'), 'rb') as f:
        [graphs, A, K] = pickle.load(f)
        shuffle_dataset(graphs)

    K = int(K)
    A = int(A)

    model_configurations['edge_type'] = ['discrete']
    model_configurations['K'] = [K]
    model_configurations['A'] = [A]

    if task_name in ['IMDB-BINARY', 'IMDB-MULTI', 'COLLAB']:
        model_configurations['node_type'] = ['continuous']
    else:
        model_configurations['node_type'] = ['discrete']

    model_selector = SimpleModelSelector(inner_processes, train_perc=train_perc)
    risk_assesser = HoldOut(outer_processes, model_selector, train_perc, exp_path, model_configurations)
    risk_assesser.risk_assessment(graphs, experiment_class, shuffle=shuffle, debug=debug)
