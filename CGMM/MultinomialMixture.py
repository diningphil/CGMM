import numpy as np


class MultinomialMixture:
    def __init__(self, c, k):
        """
        Multinomial mixture model
        :param c: the number of hidden states
        :param k: dimension of emission alphabet, which goes from 0 to k-1
        """
        self.C = c
        self.K = k
        self.smoothing = 0.00000001  # Laplace smoothing

        # Initialisation of the model's parameters.
        # Notice: the sum-to-1 requirement has been naively satisfied.
        pr = np.random.uniform(size=self.C)
        pr = pr / np.sum(pr)
        self.prior = pr

        self.emission = np.empty((self.K, self.C))
        for i in range(0, self.C):
            em = np.random.uniform(size=self.K)
            em = em / np.sum(em)
            self.emission[:, i] = em

    def train(self, target, threshold=0, max_epochs=10):
        """
        Training with Expectation Maximisation (EM) Algorithm
        :param target: the target labels in a single array
        :param threshold: stopping criterion based on the variation of the likelihood
        :param max_epochs: maximum number of epochs
        """

        likelihood_list = []

        # EM Algorithm
        current_epoch = 1
        old_likelihood = - np.inf
        delta = np.inf

        while current_epoch <= max_epochs and delta > threshold:

            datasetSize = len(target)
            batch_size = 2000
            no_batches = np.floor(datasetSize/batch_size).astype('int')
            no_batches = no_batches+1 if datasetSize%batch_size != 0 else no_batches

            likelihood = 0.

            num_prior = np.full(self.C, self.smoothing)
            den_prior = np.full(self.C, self.smoothing*self.C)

            num_emission = np.full((self.K, self.C), self.smoothing)
            den_emission = np.full((self.K, self.C), self.smoothing*self.K)

            for batch in range(0, no_batches):

                start = batch_size * batch
                end = start + batch_size if batch < no_batches - 1 else datasetSize
                curr_batch_sz = end - start

                # E-step
                numerator = np.multiply(self.emission[target[start:end], :], np.reshape(self.prior, (1, self.C)))  # len(dataset)xC
                denominator = np.dot(self.emission[target[start:end], :], np.reshape(self.prior, (self.C, 1)))  # Ux1
                posterior_estimate = np.divide(numerator, denominator)
                # todo does numpy ensure correct broadcasting when len(dataset) == C?

                # Compute the expected complete log likelihood
                likelihood += np.sum(np.multiply(posterior_estimate, np.log(np.multiply(self.emission[target[start:end], :], np.reshape(self.prior, (1, self.C))))))

                # M-step
                num_prior += np.sum(posterior_estimate, axis=0)
                den_prior += np.sum(posterior_estimate)

                np.add.at(num_emission, target[start:end], posterior_estimate)  # KxC
                den_emission += np.sum(posterior_estimate, axis=0)  # 1xC broadcasted to KxC

            self.prior = np.divide(num_prior, den_prior)
            self.emission = np.divide(num_emission, den_emission)

            delta = likelihood - old_likelihood
            old_likelihood = likelihood

            likelihood_list.append(likelihood)
            print("Mixture model training: epoch ", current_epoch, ",  E[likelihood_c] = ", likelihood)
            current_epoch += 1

        return likelihood_list

    def inference(self, prediction_set):
        """
        Takes a set and returns the most likely hidden state assignment for each node
        :param prediction_set: the target labels in a single array
        :returns: most likely hidden state labels for each vertex
        """
        # TODO mini-batch
        prods = self.emission[prediction_set, :]*np.reshape(self.prior, (1, self.C))  # len(prediction_set)xC
        return np.argmax(prods, axis=1)

    def generate(self, size):
        """
        Generates labels
        :param size: the number of labels to be generated
        :param plot: True if you want to plot the generated histogram
        :returns: a 1-D numpy array of generated labels and a 1D array of generated states
        """
        Y_gen = []
        states_gen = []
        for _ in range(0, size):
            state = np.random.choice(np.arange(0, self.C), p=self.prior)
            states_gen.append(state)
            emitted_label = np.random.choice(np.arange(0, self.K), p=self.emission[:, state])
            Y_gen.append(emitted_label)

        Y_gen = np.array(Y_gen)
        states_gen = np.array(states_gen)

        return Y_gen, states_gen

    def collective_inference(self, prediction_set, chosen):
        """
        Takes a set and returns the most likely hidden state assignment for each node
        :param prediction_set: the target labels in a single array
        :returns: most likely hidden state labels
        """
        states = np.empty(shape=len(prediction_set), dtype='int')

        # Compute hidden states for known nodes

        states[chosen] = self.inference(prediction_set[chosen])

        # Now generate state and emission label for unknown nodes using learnt prior probabilities
        size = len(chosen) - np.sum(chosen)
        prediction_set[np.logical_not(chosen)], states[np.logical_not(chosen)] = self.generate(size)

        return prediction_set, states
