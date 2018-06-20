from __future__ import absolute_import, division, print_function
from CGMMTF.MultinomialMixtureTF import MultinomialMixture
from CGMMTF.VStructureTF import VStructure
from CGMMTF.DatasetUtilities import *
from CGMMTF.TrainingUtilities import *

import pickle

task_name = 'CPDB'

with open('Graph_Tasks/' + task_name + '_data/' + task_name + '_dataset', 'rb') as f:
    [graphs, A, K] = pickle.load(f)

X, Y, adjacency_lists, sizes = unravel(graphs, one_target_per_graph=True)

# Hyper-Parameters
C = 5
C2 = 5
# use_statistics = [1, 3]  # e.g use the layer-1 and layer-3 statistics
use_statistics = [1, 3]
layers = 5  # How many layers you will train

batch_size = 2000


incremental_training(C, K, A, use_statistics, adjacency_lists, X, layers, 'statistiche',
                         threshold=0, max_epochs=100, batch_size=2000)
