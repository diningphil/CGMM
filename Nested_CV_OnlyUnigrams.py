import os
import pickle
import torch

from model_selection.K_Fold import KFold
from risk_assessment.Nested_CV import NestedCV
from experiments.Graph_Unigrams_Experiment import GraphUnigramsExperiment

'''
This file precomputes all possible graph fingerprints (unigram, unibigram ecc) for all possible configurations.
Then we use Nested_CV_OnlyClassifier to finalize model selection and risk assessment.
'''


# ---------- Hyper-Parameters ---------- #
from Nested_Configs import model_configurations, outer_folds, inner_folds, task_names
# ---------------------------------------#

# Needed to avoid thread spawning
torch.set_num_threads(1)

shuffle = True
debug = False 
outer_processes = 10
inner_processes = 14

for task_name in task_names:
    print('Starting experiments on', task_name, '...')
    exp_path = os.path.join('Results', model_configurations['model_class'][0], task_name + '_assessmentUnigrams')
    # if os.path.exists(exp_path):
    #   raise Exception("Path", exp_path, "Already present, be careful!")

    with open(os.path.join('Datasets', task_name, task_name + '_dataset'), 'rb') as f:
        [graphs, A, K] = pickle.load(f)

    K = int(K)
    A = int(A)

    model_configurations['edge_type'] = ['discrete']
    model_configurations['K'] = [K]
    model_configurations['A'] = [A]

    if task_name in ['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'COLLAB']:
        model_configurations['node_type'] = ['continuous']
    else:
        model_configurations['node_type'] = ['discrete']

    # This experiment generates the unigrams only. It tries each possible combination of infer_with_post and unibigram
    experiment_class = GraphUnigramsExperiment

    model_selector = KFold(inner_folds, inner_processes)
    risk_assesser = NestedCV(outer_folds, model_selector, outer_processes, exp_path, model_configurations)
    risk_assesser.risk_assessment(graphs, experiment_class, shuffle=shuffle, debug=debug)
