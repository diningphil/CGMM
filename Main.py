from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from MultinomialMixtureTF import MultinomialMixture

mm = MultinomialMixture(10, 5)

with tf.Session() as sess:
    # Create a toy dataset
    target_example = np.random.randint(size=(30000, 1), low=0, high=5)

    mm.train(target_example, sess, batch_size=2000)

    print(mm.compute_inference(target_example, sess))