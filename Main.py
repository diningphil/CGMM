from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from CGMMTF.MultinomialMixtureTF import MultinomialMixture
from CGMMTF.VStructureTF import VStructure

batch_size = 200

A = 3
L = 1
C = 20
C2 = 20
K = 2

# TODO what changes now is the way we build the statistics! we can put them in a batch dataset, computing them when they are needed
# and reusing what has been already computed

with tf.Session() as sess:
    # Create a toy dataset
    target_example = np.random.randint(size=(3000, 1), low=0, high=K)

    # build minibatches from dataset
    dataset = tf.data.Dataset.from_tensor_slices(target_example)
    batch_dataset = dataset.batch(batch_size=batch_size)

    with tf.variable_scope("base_layer"):
        mm = MultinomialMixture(C, K)

    print("TRAINING LAYER 0")
    mm.train(batch_dataset, sess)

    print("INFERENCE LAYER 0")
    inferred_states = mm.perform_inference(target_example, sess)
    statistics_example = np.random.randint(size=(3000, L, A, C2+1), low=0, high=5)

    # build minibatches from statistics
    statistics = tf.data.Dataset.from_tensor_slices(statistics_example)
    batch_statistics = statistics.batch(batch_size=batch_size)

    with tf.variable_scope("general_layer"):
        vs = VStructure(C, C2, K, L, A)

    print("TRAINING LAYER 1")
    vs.train(batch_dataset, batch_statistics, sess)

    print("INFERENCE LAYER 1")
    vs.perform_inference(batch_dataset, batch_statistics, sess)