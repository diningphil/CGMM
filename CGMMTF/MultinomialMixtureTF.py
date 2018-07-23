from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

class MultinomialMixture:
    def __init__(self, c, k):
        """
        Multinomial mixture model
        :param c: the number of hidden states
        :param k: dimension of emission alphabet, which goes from 0 to k-1
        """
        self.C = tf.constant(c)
        self.K = tf.constant(k)
        self.smoothing = tf.constant(0.0000001, dtype=tf.float64)  # Laplace smoothing

        # Initialisation of the model's parameters.
        # Notice: the sum-to-1 requirement has been naively satisfied.
        pr = tf.random_uniform(shape=[self.C], dtype=tf.float64)
        pr = pr / tf.reduce_sum(pr)
        self.prior = tf.Variable(initial_value=pr, name='layer0_prior', dtype=tf.float64)

        # print(self.prior)

        emission = np.zeros((k, c))
        for i in range(0, c):
            em = np.random.uniform(size=k)
            em /= np.sum(em)
            emission[:, i] = em

        self.emission = tf.Variable(initial_value=emission, name='layer0_emission', dtype=tf.float64)

        c_float = tf.cast(self.C, tf.float64)
        k_float = tf.cast(self.K, tf.float64)

        # Build the computation graph
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='labels')

        # These are Variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.prior_numerator = tf.Variable(initial_value=tf.fill([self.C], self.smoothing),
                                           name='prior_num_acc')
        self.prior_denominator = tf.Variable(initial_value=tf.fill([1], self.smoothing*c_float),
                                             name='prior_den_acc')
        self.emission_numerator = tf.Variable(initial_value=tf.fill([self.K, self.C], self.smoothing),
                                              name='emission_num_acc')
        self.emission_denominator = tf.Variable(initial_value=tf.fill([1, self.C], self.smoothing*k_float),
                                                name='emission_den_acc')
        self.likelihood = tf.Variable(initial_value=tf.zeros([1], dtype=tf.float64))

        self.compute_likelihood, self.update_prior_num, self.update_prior_den, \
            self.update_emission_num, self.update_emission_den, self.update_prior, self.update_emission, \
            self.inference = self.build_computation_graph()

        self.initializing_likelihood_accumulators = tf.group(
            *[tf.assign(self.likelihood, [0.]),
                tf.assign(self.prior_numerator, tf.fill([self.C], self.smoothing)),
                tf.assign(self.prior_denominator, tf.fill([1], self.smoothing*c_float)),
                tf.assign(self.emission_numerator, tf.fill([self.K, self.C], self.smoothing)),
                tf.assign(self.emission_denominator, tf.fill([1, self.C], self.smoothing*k_float))
              ]
        )


    def build_computation_graph(self):
        # -------------------------------- E-step -------------------------------- #


        emission_for_labels = tf.gather_nd(self.emission, self.labels)  # UxC
        ''' See tf.gather_nd
        indices = [[1], [0]]
        params = [['a', 'b'], ['c', 'd']]
        output = [['c', 'd'], ['a', 'b']]
        '''

        # Broadcasting the prior
        numerator = tf.multiply(emission_for_labels, tf.reshape(self.prior, shape=[1, self.C]))  # --> UxC
        denominator = tf.matmul(emission_for_labels, tf.reshape(self.prior, shape=[self.C, 1]))  # --> Ux1

        posterior_estimate = tf.divide(numerator, denominator)  # --> UxC

        # Compute the expected complete log likelihood
        compute_likelihood = tf.assign_add(self.likelihood,
            [tf.reduce_sum(tf.multiply(posterior_estimate, tf.log(numerator)))])

        # -------------------------------- M-step -------------------------------- #

        # These are used at each minibatch
        update_prior_num = tf.assign_add(self.prior_numerator, tf.reduce_sum(posterior_estimate, axis=0))
        update_prior_den = tf.assign_add(self.prior_denominator, [tf.reduce_sum(posterior_estimate)])

        labels = tf.squeeze(self.labels)  # removes dimensions of size 1 (current is ?x1)

        update_emission_num = tf.scatter_add(self.emission_numerator, labels, posterior_estimate)

        update_emission_den = tf.assign_add(self.emission_denominator,
            [tf.reduce_sum(posterior_estimate, axis=0)])

        # These are used at the end of an epoch to update the distributions
        update_prior = tf.assign(self.prior, tf.divide(self.prior_numerator, self.prior_denominator))
        update_emission = tf.assign(self.emission, tf.divide(self.emission_numerator, self.emission_denominator))

        # ------------------------------- Inference ------------------------------ #
        inference = tf.argmax(numerator, axis=1)

        return compute_likelihood, update_prior_num, update_prior_den, update_emission_num, update_emission_den, \
               update_prior, update_emission, inference

    def train(self, batch_dataset, sess, threshold=0, max_epochs=10, debug=False):
        """
        Training with Expectation Maximisation (EM) Algorithm
        :param batch_dataset: the target labels in a single batch dataset
        :param sess: TensorFlow session
        :param threshold: stopping criterion based on the variation of the likelihood
        :param max_epochs: maximum number of epochs
        """
        # EM Algorithm
        current_epoch = 0
        old_likelihood = - np.inf
        delta = np.inf

        if debug:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        iterator = batch_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        sess.run(tf.global_variables_initializer())

        while current_epoch <= max_epochs and delta > threshold:
            sess.run(iterator.initializer)

            # Run the nodes that initialize likelihood and accumulators
            sess.run([self.initializing_likelihood_accumulators])

            while True:
                try:
                    batch = sess.run(next_element)

                    if debug:
                        # For batch in batches
                        likelihood, _, _, _, _, = sess.run([self.compute_likelihood,
                                                            self.update_prior_num, self.update_prior_den,
                                                            self.update_emission_num, self.update_emission_den],
                                                           feed_dict={self.labels: batch},
                                                           options=run_options,
                                                           run_metadata=run_metadata)
                        # Create the Timeline object, and write it to a json
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('MM_timeline' + str(current_epoch) + '.json', 'w') as f:
                            f.write(ctf)
                    else:
                        # For batch in batches
                        likelihood, _, _, _, _, = sess.run([self.compute_likelihood,
                                                            self.update_prior_num, self.update_prior_den,
                                                            self.update_emission_num, self.update_emission_den],
                                                           feed_dict={self.labels: batch})

                except tf.errors.OutOfRangeError:
                    break

            delta = likelihood - old_likelihood
            old_likelihood = likelihood

            # Run update variables passing the required variables
            sess.run([self.update_prior, self.update_emission])

            current_epoch += 1

            print("End of epoch", current_epoch, "likelihood:", likelihood)

    def perform_inference(self, batch_dataset, sess):
        """
        Takes a set and returns the most likely hidden state assignment for each node
        :param target: the target labels in a single array
        :returns: most likely hidden state labels for each vertex
        """

        iterator = batch_dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        predictions = None

        sess.run(iterator.initializer)

        while True:
            try:
                batch = sess.run(next_element)

                # For batch in batches
                states = sess.run([self.inference], feed_dict={self.labels: batch})

                if predictions is None:
                    predictions = states
                else:
                    predictions = np.append(predictions, states)

            except tf.errors.OutOfRangeError:
                break

        return predictions
