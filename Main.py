from __future__ import absolute_import, division, print_function
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
use_statistics = [1]
layers = 8  # How many layers you will train
max_epochs = 30

batch_size = 2000

'''
# WARNING: if you reuse the statistics, make sure the order of the vertexes is the same
# do NOT re-shuffle the dstatisticheataset if you want to keep using them
'''

save_name = 'first_experiment'
statistics_name = save_name + '_statistiche'
unigram_inference_name_train = save_name + '_unigrams_train'
unigram_inference_name_valid = save_name + '_unigrams_valid'

statistics_inference_name = save_name + '_statistiche_inferenza'


# Training and inference phase
batch_dataset = target_dataset.batch(batch_size)

incremental_training(C, K, A, use_statistics, adjacency_lists, batch_dataset, layers, statistics_name,
                         threshold=0, max_epochs=max_epochs, save_name=save_name)

# Now recreate the dataset and the computation graph, because incremental_training resets the graph at the end
# (after saving the model)
target_dataset = tf.data.Dataset.from_tensor_slices(np.reshape(X, (X.shape[0], 1)))

incremental_inference(save_name, K, A, C, layers, use_statistics, batch_dataset, adjacency_lists, sizes,
                          unigram_inference_name_train, statistics_inference_name)

# FOR VALIDATION
#incremental_inference(save_name, K, A, C, layers, use_statistics, target_dataset_valid, adjacency_lists_valid, sizes_valid,
#                          unigram_inference_name_valid, statistics_inference_name, batch_size=batch_size)

# -------------------------------------------------------------------------------------------------------------


unigrams_train = recover_unigrams(unigram_inference_name_train, layers=[0, 1, 2, 3, 4, 5, 6, 7],
                            C=C, concatenate=True, return_numpy=True)

# FOR VALIDATION
#unigrams_valid = recover_unigrams(unigram_inference_name_train, layers=[0, 1, 2, 3, 4, 5, 6, 7],
#                            C=C, concatenate=True, return_numpy=True)


# vl_acc is -1 if you do not pass validation parameters
tr_acc, vl_acc = compute_svm_accuracy(unigrams_train, Y, 10, 5, unigrams_valid=None, Y_valid=None)