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

target_dataset = tf.data.Dataset.from_tensor_slices(np.reshape(X, (X.shape[0], 1)))

# Hyper-Parameters
C = 10
C2 = 10
# use_statistics = [1, 3]  # e.g use the layer-1 and layer-3 statistics
use_statistics = [1, 2]
layers = 3  # How many layers you will train

batch_size = 2000

'''
# WARNING: if you reuse the statistics, make sure the order of the vertexes is the same
# do NOT re-shuffle the dataset if you want to keep using them
'''

save_name = 'first_experiment'
statistics_name = 'statistiche'
statistics_inference_name = 'statistiche_inferenza'

incremental_training(C, K, A, use_statistics, adjacency_lists, target_dataset, layers, statistics_name,
                         threshold=0, max_epochs=3, batch_size=2000, save_name=save_name)


# Now recreate the dataset and the computation graph, because incremental_training resets the graph at the end
# (after saving the model)
target_dataset = tf.data.Dataset.from_tensor_slices(np.reshape(X, (X.shape[0], 1)))
architecture = build_architecture(K, A, C, use_statistics, layers)

incremental_inference(architecture, save_name, A, C, use_statistics, target_dataset, adjacency_lists, sizes,
                          statistics_inference_name, batch_size=batch_size)
