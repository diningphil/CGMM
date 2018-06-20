from __future__ import absolute_import, division, print_function
from CGMMTF.MultinomialMixtureTF import MultinomialMixture
from CGMMTF.VStructureTF import VStructure
from CGMMTF.DatasetUtilities import *

import pickle

task_name = 'CPDB'

with open('Graph_Tasks/' + task_name + '_data/' + task_name + '_dataset', 'rb') as f:
    [graphs, A, K] = pickle.load(f)

X, Y, adjacency_lists, sizes = unravel(graphs, one_target_per_graph=True)

C = 5
C2 = 5

batch_size = 2000
# use_statistics = [1, 3]  # e.g use the layer-1 and layer-3 statistics
use_statistics = [1, 3]

layers = 5  # How many layers you will train


with tf.Session() as sess:

    # build minibatches from dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.reshape(X, (X.shape[0], 1)))
    batch_dataset = dataset.batch(batch_size=batch_size)

    print("LAYER 0")
    with tf.variable_scope("base_layer"):
        mm = MultinomialMixture(C, K)
        mm.train(batch_dataset, sess)

    inferred_states = mm.perform_inference(batch_dataset, sess)
    save_statistics(adjacency_lists, inferred_states, X, A, C2, 'statistiche', 0)

    for layer in range(1, layers):
        print("LAYER", layer)

        # e.g 1 - [1, 3] = [0, -2] --> [0]
        # e.g 5 - [1, 3] = [4, 2]  --> [4, 2]
        layer_wise_statistics = [(layer-x) for x in use_statistics if (layer-x) >= 0]

        L = len(layer_wise_statistics)

        # print(layer_wise_statistics)

        stats_dataset = recover_statistics('statistiche', layer_wise_statistics, A, C2)
        batch_statistics = stats_dataset.batch(batch_size=batch_size)

        stats_iterator = batch_statistics.make_initializable_iterator()
        stats_next_element = stats_iterator.get_next()

        sess.run(stats_iterator.initializer)

        with tf.variable_scope("general_layer"):
            vs = VStructure(C, C2, K, L, A)

            vs.train(batch_dataset, batch_statistics, sess, max_epochs=4)
            inferred_states = vs.perform_inference(batch_dataset, batch_statistics, sess)

            save_statistics(adjacency_lists, inferred_states, X, A, C2, 'statistiche', layer)

