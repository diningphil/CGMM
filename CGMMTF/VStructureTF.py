from __future__ import absolute_import, division, print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

def current_milli_time():
    return int(round(time.time() * 1000))


class VStructure:
    def __init__(self, c, c2, k, l, a, current_layer):
        """
        Multinomial mixture model
        :param c: the number of hidden states
        :param c2: the number of states of the "children"
        :param k: dimension of output's alphabet, which goes from 0 to K-1
        :param l: number of layers to consider. You must pass the appropriate number of statistics at training
        :param a: dimension of edges' alphabet, which goes from 0 to A-
        :param current_layer: used to save and restore the model
        """
        self.current_layer = str(current_layer)

        c2 = c2 + 1  # always consider a bottom state also

        self.C  = tf.constant(c)
        self.C2 = tf.constant(c2)
        self.K  = tf.constant(k)
        self.L  = tf.constant(l)
        self.A  = tf.constant(a)
        self.smoothing = tf.constant(0.001)  # Laplace smoothing
        self.emission, self.arcS, self.layerS, self.transition = self.initialize_parameters(c, c2, k, l, a)

        # ------------------------- Build the computation graph ------------------------- #

        k_float = tf.cast(self.K, tf.float32)
        l_float = tf.cast(self.L, tf.float32)
        a_float = tf.cast(self.A, tf.float32)
        c_float = tf.cast(self.C, tf.float32)

        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='labels')
        self.stats = tf.placeholder(shape=[None, l, a, c2], dtype=tf.float32, name='statistics')

        batch_size = tf.shape(self.labels)[0]

        # These are Variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.layerS_numerator = tf.Variable(initial_value=tf.fill([self.L], self.smoothing),
                                           name='levelS_num_acc')
        self.layerS_denominator = tf.Variable(initial_value=tf.fill([1], self.smoothing*l_float),
                                             name='levelS_den_acc')
        self.emission_numerator = tf.Variable(initial_value=tf.fill([self.K, self.C], self.smoothing),
                                              name='emission_num_acc')
        self.emission_denominator = tf.Variable(initial_value=tf.fill([1, self.C], self.smoothing*k_float),
                                                name='emission_den_acc')
        self.arcS_numerator = tf.Variable(initial_value=tf.fill([self.L, self.A], self.smoothing),
                                              name='arcS_num_acc')
        self.arcS_denominator = tf.Variable(initial_value=tf.fill([self.L, 1], self.smoothing*a_float),
                                                name='arcS_den_acc')
        self.transition_numerator = tf.Variable(
            initial_value=tf.fill([self.L, self.A, self.C, self.C2], self.smoothing), name='transition_num_acc')
        self.transition_denominator = tf.Variable(
            initial_value=tf.fill([self.L, self.A, 1, self.C2], self.smoothing*c_float), name='transition_den_acc')

        self.likelihood = tf.Variable(initial_value=[0.])

        self.initializing_likelihood_accumulators = tf.group(
            *[tf.assign(self.likelihood, [0.]),
                tf.assign(self.layerS_numerator, tf.fill([self.L], self.smoothing)),
                tf.assign(self.layerS_denominator, tf.fill([1], self.smoothing*l_float)),
                tf.assign(self.arcS_numerator, tf.fill([self.L, self.A], self.smoothing)),
                tf.assign(self.arcS_denominator, tf.fill([self.L, 1], self.smoothing*a_float)),
                tf.assign(self.transition_numerator, tf.fill([self.L, self.A, self.C, self.C2], self.smoothing)),
                tf.assign(self.transition_denominator, tf.fill([self.L, self.A, 1, self.C2], self.smoothing*c_float)),
                tf.assign(self.emission_numerator, tf.fill([self.K, self.C], self.smoothing)),
                tf.assign(self.emission_denominator, tf.fill([1, self.C], self.smoothing*k_float))
              ]
        )

        # Compute the neighbourhood dimension for each vertex
        neighbDim = tf.reduce_sum(self.stats[:, 0, :, :], axis=2)  # --> ? x A

        # Replace zeros with ones to avoid divisions by zero
        # This does not alter learning: the numerator can still be zero
        neighbDim = tf.where(tf.equal(neighbDim, 0.), tf.ones(tf.shape(neighbDim)), neighbDim)

        # -------------------------------- E-step -------------------------------- #

        broadcastable_transition = tf.expand_dims(self.transition, axis=0)  # --> 1 x L x A x C x C2
        broadcastable_stats = tf.expand_dims(self.stats, axis=3)            # --> ? x L x A x 1 x C2

        tmp = tf.reduce_sum(tf.multiply(broadcastable_transition, broadcastable_stats), axis=4)    # --> ? x L x A x C

        broadcastable_layerS = tf.expand_dims(self.layerS, 1)  # --> L x 1

        tmp2 = tf.reshape(tf.multiply(broadcastable_layerS, self.arcS), [1, self.L, self.A, 1])    # --> 1 x L x A x 1
        
        div_neighb = tf.reshape(neighbDim, [batch_size, 1, self.A, 1])                             # --> ? x 1 x A x 1

        posterior_estimate = tf.divide(tf.multiply(tmp, tmp2), div_neighb)                         # --> ? x L x A x C

        emission_for_labels = tf.gather_nd(self.emission, self.labels)  # --> ? x C
        ''' See tf.gather_nd
        indices = [[1], [0]]
        params = [['a', 'b'], ['c', 'd']]
        output = [['c', 'd'], ['a', 'b']]
        '''
        tmp_emission = tf.reshape(emission_for_labels,
                                  [batch_size, 1, 1, self.C])  # --> ? x 1 x 1 x C

        posterior_estimate = tf.multiply(posterior_estimate, tmp_emission)    # --> ? x L x A x C

        self.unnorm_posterior_ui = tf.reduce_sum(posterior_estimate, axis=[1, 2])

        # Normalize
        norm_constant = tf.reshape(tf.reduce_sum(posterior_estimate, axis=[1, 2, 3]), [batch_size, 1, 1, 1])
        norm_constant = tf.where(tf.equal(norm_constant, 0.), tf.ones(tf.shape(norm_constant)), norm_constant)

        posterior_estimate = tf.divide(posterior_estimate, norm_constant)  # --> ? x L x A x C

        self.posterior_estimate = posterior_estimate

        posterior_uli = tf.reduce_sum(posterior_estimate, axis=2)          # --> ? x L x C
        posterior_ui = tf.reduce_sum(posterior_uli, axis=1)                # --> ? x C

        self.posterior_ui = posterior_ui

        # -------------------------------- M-step -------------------------------- #

        labels = tf.squeeze(self.labels)  # removes dimensions of size 1 (current is ?x1)

        # These are equivalent to the multinomial mixture model, it just changes how the posterior is computed
        self.update_emission_num = tf.scatter_add(self.emission_numerator, labels, posterior_ui)  # --> K x C
        self.update_emission_den = tf.assign_add(self.emission_denominator,
                                                 [tf.reduce_sum(posterior_ui, axis=0)])           # --> 1 x C

        # Arc and layer selector must be updated AFTER they have been used to compute the new transition distr.
        new_arc_num = tf.reduce_sum(posterior_estimate, axis=[0, 3])  # --> L x A

        self.update_arcS_num = tf.assign_add(self.arcS_numerator, new_arc_num)
        self.update_arcS_den = tf.assign_add(self.arcS_denominator,
                                             tf.expand_dims(tf.reduce_sum(new_arc_num, axis=1), axis=1))  # --> L x 1

        new_layer_num = tf.reduce_sum(posterior_uli, axis=[0, 2])  # --> [L]

        self.update_layerS_num = tf.assign_add(self.layerS_numerator, new_layer_num)
        self.update_layerS_den = tf.assign_add(self.layerS_denominator, [tf.reduce_sum(new_layer_num)])  # --> [1]

        log_trans = tf.log(self.transition)

        # NOTE: these terms become expensive in terms of memory computation! A smaller batch may be required!
        # This is not a problem because we are not doing stochastic optimisations

        num = tf.divide(
                    tf.multiply(self.transition,
                            tf.multiply(tf.reshape(self.layerS, [self.L, 1, 1, 1]),
                                        tf.reshape(self.arcS,   [self.L, self.A, 1, 1]))),
                            tf.expand_dims(div_neighb, axis=4))

        num = tf.multiply(num, tf.reshape(emission_for_labels, [batch_size, 1, 1, self.C, 1]))
        num = tf.multiply(num, broadcastable_stats)

        den = tf.reduce_sum(num, keepdims=True, axis=[1, 2, 3, 4])  # --> ? x 1 x 1 x 1 x 1
        den = tf.where(tf.equal(den, 0.), tf.ones(tf.shape(den)), den)

        eulaij = tf.divide(num, den)  # --> ? x L x A x C x C2

        new_trans_num = tf.reduce_sum(tf.multiply(eulaij, broadcastable_stats), axis=0)

        self.update_transition_num = tf.assign_add(self.transition_numerator, new_trans_num)
        self.update_transition_den = tf.assign_add(self.transition_denominator,
                                                   tf.expand_dims(tf.reduce_sum(new_trans_num, axis=2),
                                                                  axis=2))  # --> L x A x 1 x C2

        # Compute the expected complete log likelihood
        self.likelihood1 = tf.reduce_sum(tf.multiply(posterior_ui, tf.log(emission_for_labels)))
        self.likelihood2 = tf.reduce_sum(np.multiply(posterior_uli, tf.log(broadcastable_layerS)))
        self.likelihood3 = tf.reduce_sum(np.multiply(posterior_estimate,
                                                 tf.reshape(tf.log(self.arcS), [1, self.L, self.A, 1])))

        self.likelihood4 = tf.reduce_sum(tf.multiply(tf.multiply(eulaij, broadcastable_stats), log_trans))

        likelihood_sum = self.likelihood1 + self.likelihood2 + self.likelihood3 + self.likelihood4

        # self.compute_likelihood becomes an assign op
        self.compute_likelihood = tf.assign_add(self.likelihood, [likelihood_sum])

        self.update_emission   = tf.assign(self.emission, tf.divide(self.emission_numerator, self.emission_denominator))
        self.update_transition = tf.assign(self.transition, tf.divide(self.transition_numerator,
                                                                      self.transition_denominator))
        self.update_arcS       = tf.assign(self.arcS, tf.divide(self.arcS_numerator, self.arcS_denominator))
        self.update_layerS     = tf.assign(self.layerS, tf.divide(self.layerS_numerator, self.layerS_denominator))

        # -------------------------------- Inference -------------------------------- #

        # NOTE: this is exactly the same formula as in MultinomialMixture, it changes how you compure the posterior
        self.inference = tf.cast(tf.argmax(self.unnorm_posterior_ui, axis=1), dtype=tf.int32)

    def initialize_parameters(self, c, c2, k, l, a):
        emission_dist = np.zeros((k, c))
        for i in range(0, c):
            em = np.random.uniform(size=k)
            em /= np.sum(em)
            emission_dist[:, i] = em

        arc_dist = np.zeros((l, a))
        for layer in range(0, l):
            dist = np.random.uniform(size=a)
            dist /= np.sum(dist)
            arc_dist[layer, :] = dist

        layer_dist = tf.random_uniform(shape=[l])
        layer_dist = layer_dist / tf.reduce_sum(layer_dist)

        transition_dist = np.zeros((l, a, c, c2))

        for layer in range(0, l):
            for arc in range(0, a):
                for j in range(0, c2):
                    tr = np.random.uniform(size=c)
                    transition_dist[layer, arc, :, j] = tr/np.sum(tr)

        emission = tf.Variable(initial_value=emission_dist, name='layer-' + self.current_layer + '-emission',
                               dtype=tf.float32)
        arcS = tf.Variable(initial_value=arc_dist, name='layer-' + self.current_layer + '-arcSelector',
                           dtype=tf.float32)
        layerS = tf.Variable(initial_value=layer_dist, name='layer-' + self.current_layer + '-layerSelector',
                             dtype=tf.float32)
        transition = tf.Variable(initial_value=transition_dist, name='layer-' + self.current_layer + '-transition',
                                 dtype=tf.float32)

        return emission, arcS, layerS, transition

    def train(self, batch_dataset, batch_statistics, sess, threshold=0, max_epochs=10):
        """
        Training with Expectation Maximisation (EM) Algorithm
        :param batch_dataset: the target labels all in a batch dataset
        :param batch_statistics: dataset of shape ? x L x A x C2
        :param sess: TensorFlow session
        :param threshold: stopping criterion based on the variation off the likelihood
        :param max_epochs: maximum number of epochs
        """
        # EM Algorithm
        current_epoch = 0
        old_likelihood = - np.inf
        delta = np.inf

        dataset = tf.data.Dataset.zip((batch_dataset, batch_statistics))
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        sess.run(tf.global_variables_initializer())
      
        #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        
        while current_epoch < max_epochs and delta > threshold:

            sess.run(iterator.initializer)

            # Reinitialize the likelihood
            sess.run([self.initializing_likelihood_accumulators])

            while True:
                try:
                    batch, stats = sess.run(next_element)
                    print(batch.shape, stats.shape)
                    '''
                    print(sess.run(
                        [self.likelihood1, self.likelihood2, self.likelihood3, self.likelihood4],
                        feed_dict={self.labels: batch, self.stats: stats}))
                    '''

                    # For batch in batches
                    likelihood, _, _, _, _, _, _, _, _, = sess.run(
                        [self.compute_likelihood,
                         self.update_layerS_num, self.update_layerS_den,
                         self.update_arcS_num, self.update_arcS_den,
                         self.update_transition_num, self.update_transition_den,
                         self.update_emission_num, self.update_emission_den],
                        feed_dict={self.labels: batch, self.stats: stats}#,
                        #options=options,
                        #run_metadata=run_metadata
                    )

                    #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    #with open('timeline_02_step_%d.json' % current_epoch, 'w') as f:
                    #    f.write(chrome_trace)
                except tf.errors.OutOfRangeError:
                    break

            delta = likelihood - old_likelihood
            old_likelihood = likelihood
            current_epoch += 1
            print("End of epoch", current_epoch, "likelihood:", likelihood[0])

            # Run update variables passing the required variables
            sess.run([self.update_layerS, self.update_arcS, self.update_emission, self.update_transition])

    def perform_inference(self, batch_dataset, batch_statistics, sess):
        """
        Takes a set and returns the most likely hidden state assignment for each node's label
        :param batch_dataset: the target labels all in a batch dataset
        :param batch_statistics: dataset of shape ? x L x A x C2
        :param sess: TensorFlow session
        :returns: most likely hidden state labels
        """
        dataset = tf.data.Dataset.zip((batch_dataset, batch_statistics))
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer)

        predictions = None

        while True:
            try:
                batch, stats = sess.run(next_element)

                # For batch in batches
                inferred_states = sess.run([self.inference], feed_dict={self.labels: batch, self.stats: stats})

                if predictions is None:
                    predictions = inferred_states
                else:
                    predictions = np.append(predictions, inferred_states)

            except tf.errors.OutOfRangeError:
                break

        return predictions
