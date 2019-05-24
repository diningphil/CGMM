import torch
import os
import pickle

from experiments.Graph_Trivial_Experiment import GraphTrivialExperiment
from utils.utils import shuffle_dataset


def main():

    # These are the keys that my experiment will use
    model_config = {
        'classifier': 'mlp',
        'l2': 1e-3,
        'learning_rate': 1e-4,
        'l_batch_size': 100,
        'training_epochs': 2000,  # max no of training epochs
        'early_stopping': 50,  # min no of epochs to begin using early stopping
        'hidden_units': 64,
        'plot': True,

    }
    # ---------------------------------------#
    task_name = 'DD'  # Choose it

    f = open(os.path.join('Datasets', task_name, task_name + '_dataset'), 'rb')
    graphs, A, K = pickle.load(f)
    shuffle_dataset(graphs)

    # Split in train and test, pass train to model selector
    train_perc = 0.9

    last_tr_idx = int(len(graphs) * train_perc)
    train, test = graphs[:last_tr_idx], graphs[last_tr_idx:]

    # Additional configurations
    model_config['edge_type'] = 'discrete'
    model_config['K'] = int(K)
    model_config['A'] = int(A)
    if task_name in ['IMDB-BINARY', 'IMDB-MULTI', 'COLLAB']:
        model_config['node_type'] = 'continuous'
    else:
        model_config['node_type'] = 'discrete'

    # Prepare and run the experiment
    torch.set_num_threads(1)  # Needed to avoid thread spawning

    exp_path = os.path.join('Results', 'CGMM', task_name + '_trivialfingerprint')

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    experiment = GraphTrivialExperiment(model_config, exp_path)
    _, _ = experiment.run_test(train, test)


if __name__ == '__main__':
    main()
