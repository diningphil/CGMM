from __future__ import absolute_import, division, print_function
from CGMMTF.MultinomialMixtureTF import MultinomialMixture
from CGMMTF.VStructureTF import VStructure
from CGMMTF.DatasetUtilities import *

import pickle
from CGMMTF.DatasetUtilities import unravel

task_name = 'CPDB'

with open('Graph_Tasks/' + task_name + '_data/' + task_name + '_dataset', 'rb') as f:
    [graphs, A, K] = pickle.load(f)

X, Y, adjacency_lists, sizes = unravel(graphs, one_target_per_graph=True)


L = 1  # TODO it has to be 1 now, when we will load statistics from multiple files L may be set arbitrarily
C = 2
C2 = 2

batch_size = 2000
layers = 2

# TODO
# TODO   In the old implementation the likelihood asssociated to the transition part decreases much more quickly,
# TODO   whilst in the new implementation it stays almost fixed after the 2nd epoch

np.random.seed(0)

# Comparing with the old implementation
from CGMM.TrainingUtilities import incremental_training
incremental_training(C,K,A,np.array([1]), adjacency_lists, X, layers, max_epochs=4)

with tf.Session() as sess:

    # build minibatches from dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.reshape(X, (X.shape[0], 1)))
    batch_dataset = dataset.batch(batch_size=batch_size)

    np.random.seed(0)

    print("LAYER 0")
    with tf.variable_scope("base_layer"):
        mm = MultinomialMixture(C, K)
        mm.train(batch_dataset, sess)

    inferred_states = mm.perform_inference(batch_dataset, sess)

    for layer in range(1, layers):
        print("LAYER", layer)

        save_statistics(adjacency_lists, inferred_states, X, A, C2, 'statistiche', layer)

        stats_dataset = recover_statistics('statistiche', layer, A, C2)
        batch_statistics = stats_dataset.batch(batch_size=batch_size)

        stats_iterator = batch_statistics.make_initializable_iterator()
        stats_next_element = stats_iterator.get_next()

        sess.run(stats_iterator.initializer)

        with tf.variable_scope("general_layer"):
            vs = VStructure(C, C2, K, L, A)

            vs.train(batch_dataset, batch_statistics, sess, max_epochs=4)
            inferred_states = vs.perform_inference(batch_dataset, batch_statistics, sess)

            