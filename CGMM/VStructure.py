import time
import numpy as np


def current_milli_time():
    return int(round(time.time() * 1000))


class VStructure:

    def initParameters(self):
        # Initialisation of the model's parameters.
        self.emission = np.empty((self.K, self.C))
        for i in range(0, self.C):
            em = np.random.uniform(size=self.K)
            em = em / np.sum(em)
            self.emission[:, i] = em

        levS = np.random.uniform(size=self.L)  # layer Selector
        self.layerS = levS / np.sum(levS)
        self.arcS = np.empty((self.L, self.A))
        self.transition = np.empty((self.L, self.A, self.C, self.C2))

        for l in range(0, self.L):
            arcDist = np.random.uniform(size=self.A)  # arc Selector
            self.arcS[l, :] = arcDist / np.sum(arcDist)
            for a in range(0, self.A):
                for j in range(0, self.C2):
                    tr = np.random.uniform(size=self.C)
                    self.transition[l, a, :, j] = tr/np.sum(tr)

    def __init__(self, c, c2, k, Lprec, a):
        """
        Multinomial mixture model
        :param c: the number of hidden states
        :param c2: the number of states of the "children"
        :param k: dimension of output's alphabet, which goes from 0 to K-1
        :param Lprec: ORDERED (shallow to deep) numpy array of integers: value i tells the model to take the ith precedent layer
        :param a: dimension of edges' alphabet, which goes from 0 to A-1
        """
        self.C = c
        self.C2 = c2 + 1  # always consider the bottom state
        self.K = k
        self.L = len(Lprec)
        self.A = a
        self.Lprec = Lprec  # indices must start from 1
        self.smoothing = 0.001  # Laplace smoothing
        self.emission = None
        self.transition = None
        self.layerS = None
        self.arcS = None

    def compute_statistics(self, adjacency_lists, last_states, prev_statistics=None):
        """
        :param last_states: the last array of states
        :param prev_statistics: the statistics computed at ALL the previous layers: list of numpy matrices UxAxC2
        :return: the statistics needed for this model, according to the Lprec parameter
        """

        # Compute statistics
        new_statistics = np.zeros((len(adjacency_lists), self.A, self.C2))

        for u in range(0, len(adjacency_lists)):
            incident_nodes = adjacency_lists[u]
            for u2, a in incident_nodes:
                node_state = last_states[u2]

                # THIS CHECK IS USEFUL FOR COLLECTIVE INFERENCE
                if node_state != -1:
                    new_statistics[u, a, node_state] += 1

        # Save it into a field for future use (incremental inference/training) --> strongly coupled
        last_statistics = np.reshape(new_statistics, (len(adjacency_lists), 1, self.A, self.C2))

        # take just the stats I am interested in
        if prev_statistics is not None:

            # Todo this is responsible for the increase in computation time while
            all_statistics = np.concatenate([np.reshape(last_statistics, (len(adjacency_lists), 1, self.A, self.C2)),
                                             prev_statistics], axis=1)  # UxLxAxC2

            bound = -1

            if all_statistics.shape[1] >= self.Lprec[bound]:  # I can take all the desired precedent layers
                all_statistics = all_statistics[:, self.Lprec - 1, :, :]

            else:  # take the max previous number of states

                while all_statistics.shape[1] < self.Lprec[bound]:  # Lprec[bound] is still too much
                    bound -= 1

                all_statistics = all_statistics[:, self.Lprec[:bound+1] - 1, :, :]

                # L has to adapt to the number of preceeding layers
                self.L = len(self.Lprec[:bound+1])

        else:
            # Update parameters dimension (during training parameters will be initialised)
            self.L = 1
            self.Lprec = np.array([0])
            all_statistics = np.reshape(new_statistics, (len(adjacency_lists), 1, self.A, self.C2))

        # print("L is ", self.L)

        return all_statistics

    def train(self, target, threshold=-1, max_epochs=10, adjacency_lists=None, last_states=None, prevNeighbourStats=None, layer=None):
        """last_states, prevNeighbourStats
        Training with Expectation Maximisation (EM) Algorithm
        :param target: the target labels all in a single arraylast_states, prevNeighbourStats
        :param threshold: stopping criterion based on the variation off the likelihood
        :param max_epochs: maximum number of epochs
        :param plot: True if you want to see real-time visualisation of likelihood and parameters' histograms
        """
        time = current_milli_time()

        # This updates lastLayerStats, L and Lprec
        neighbourhoodStats = self.compute_statistics(adjacency_lists, last_states, prevNeighbourStats)

        # Now I can initialise parameters
        self.initParameters()

        # print(self.transition[0, 0, :, :])
        likelihood_list = []

        # EM Algorithm
        current_epoch = 1
        old_likelihood = - np.inf
        delta = np.inf

        # do not remove this line
        neighbourhoodStats = np.reshape(neighbourhoodStats, (len(target), self.L, self.A, self.C2))

        # print(last_states)
        # print(neighbourhoodStats[:, 0, 0, :])
        while current_epoch <= max_epochs and delta > threshold:
            time = current_milli_time()

            datasetSize = len(target)
            batch_size = 2000
            no_batches = np.floor(datasetSize/batch_size).astype('int')
            no_batches = no_batches+1 if datasetSize%batch_size != 0 else no_batches

            likelihood = 0.
            num_emission = np.full((self.K, self.C), self.smoothing)
            den_emission = np.full((self.K, self.C), self.smoothing*self.K)
            num_trans = np.full((self.L, self.A, self.C, self.C2), self.smoothing)
            den_trans = np.full((self.L, self.A, self.C, self.C2), 0.)
            num_arc = np.full((self.L, self.A), self.smoothing)
            den_arc = np.full((self.L, self.A), self.smoothing*self.A)
            num_layer = np.full(self.L, self.smoothing)
            den_layer = np.full(self.L, self.smoothing*self.L)

            for batch in range(0, no_batches):

                start = batch_size*batch
                end = start+batch_size if batch < no_batches-1 else datasetSize
                curr_batch_sz = end - start

                # E-step
                t = current_milli_time()

                neighbDim = np.sum(neighbourhoodStats[start:end, 0, :, :], axis=2)
                neighbDim[neighbDim == 0] = 1

                posterior_ulai = \
                    np.sum(np.multiply(np.reshape(self.transition, (1, self.L, self.A, self.C, self.C2)),
                                       np.reshape(neighbourhoodStats[start:end], (curr_batch_sz, self.L, self.A, 1, self.C2))),
                           axis=4) \
                    * np.reshape(np.multiply(self.arcS, np.reshape(self.layerS, (self.L, 1))),
                                 (1, self.L, self.A, 1)) \
                    / np.reshape(neighbDim, (curr_batch_sz, 1, self.A, 1))

                tmp_em = np.reshape(self.emission[target[start:end], :], (curr_batch_sz, 1, 1, self.C))
                posterior_ulai = np.multiply(posterior_ulai, tmp_em)

                # Normalize
                norm_constant = np.reshape(np.sum(posterior_ulai, axis=(1, 2, 3)), (curr_batch_sz, 1, 1, 1))
                norm_constant[norm_constant == 0] = 1
                posterior_ulai = posterior_ulai / norm_constant

                posterior_uli = np.sum(posterior_ulai, axis=2)
                posterior_ui = np.sum(posterior_uli, axis=1)

                timeE = current_milli_time()
                # print("E-step in ", timeE-time)

                # M-step
                np.add.at(num_emission, target[start:end], posterior_ui)  # KxC
                den_emission += np.sum(posterior_ui, axis=0)  # 1xC

                # Arc and layer selector must be updated AFTER they have been used to compute the new transition distr.
                num_arc += np.sum(posterior_ulai, axis=(0, 3))

                num_layer += np.sum(posterior_uli, axis=(0, 2))

                timeM = current_milli_time()
                # print("M-step in ", timeM-timeE)

                # Compute the expected complete log likelihood
                likelihood += np.sum(np.multiply(posterior_ui, np.log(self.emission[target[start:end], :])))
                likelihood += np.sum(np.multiply(posterior_uli, np.reshape(np.log(self.layerS), (self.L, 1))))
                likelihood += np.sum(np.multiply(posterior_ulai, np.reshape(np.log(self.arcS), (self.L, self.A, 1))))

                log_trans = np.log(self.transition[:, :, :, :])

                for u in range(0, curr_batch_sz):

                    num_u = np.multiply(self.transition,
                                        np.multiply(np.reshape(self.layerS, (self.L, 1, 1, 1)),
                                                    np.reshape(self.arcS, (self.L, self.A, 1, 1)))) \
                                                              / np.reshape(neighbDim[u, :], (1, self.A, 1, 1))

                    start_u = u + start

                    num_u = np.multiply(num_u, np.reshape(self.emission[target[start_u], :], (1, 1, self.C, 1)))
                    num_u = np.multiply(num_u, np.reshape(neighbourhoodStats[start_u, :, :, :], (self.L, self.A, 1, self.C2)))

                    den_u = np.sum(num_u)
                    den_u = 1. if den_u == 0. else den_u

                    eulaij = num_u/den_u

                    likelihood += np.sum(
                        np.multiply(
                            np.multiply(eulaij,
                                        np.reshape(neighbourhoodStats[start_u, :, :, :], (self.L, self.A, 1, self.C2))),
                            log_trans
                        ))

                    # TRANSITION M-step (included here to save another for loop)
                    num_trans += np.multiply(eulaij, np.reshape(neighbourhoodStats[start_u, :, :, :], (self.L, self.A, 1, self.C2)))


            # They needs to be put here, since num_* is increased at each minibatch
            den_arc += np.reshape(np.sum(num_arc, axis=1), (self.L, 1))
            den_layer += np.sum(num_layer)
            den_trans += np.reshape(np.sum(num_trans, axis=2), (self.L, self.A, 1, self.C2))

            timeL = current_milli_time()
            print("Layer ", layer, " VStructure model training: epoch ", current_epoch, ",  likelihood = ", likelihood)

            delta = likelihood - old_likelihood
            old_likelihood = likelihood

            likelihood_list.append(likelihood)

            # Update parameters
            if delta > 0:

                np.divide(num_trans, den_trans, out=self.transition)
                np.divide(num_emission, den_emission, out=self.emission)
                np.divide(num_arc, den_arc, out=self.arcS)
                np.divide(num_layer, den_layer, out=self.layerS)

            # print("Epoch in ", current_milli_time()-time, " ms.")
            # print("Ms for each node u ", (current_milli_time()-time)/datasetSize)

            current_epoch += 1

        return likelihood_list

    def inference(self, prediction_set, adjacency_lists, last_states, prev_statistics=None):
        """
        Takes a set and returns the most likely hidden state assignment for each node's label
        :param prediction_set: the target labels in a single array
        :param last_states: states computed at the previous layer
        :param prev_statistics: the statistics computed at ALL the previous layers: list of numpy matrices UxAxC2
        :returns: most likely hidden state labels
        """
        predictionStats = self.compute_statistics(adjacency_lists, last_states, prev_statistics)
        arcSbcast = np.reshape(self.arcS, (1, self.L, self.A, 1))
        layerSbcast = np.reshape(self.layerS, (1, self.L, 1))

        datasetSize = len(prediction_set)
        batch_size = 2000
        no_batches = np.floor(datasetSize / 2000).astype('int')
        no_batches = no_batches + 1 if datasetSize % 2000 != 0 else no_batches

        prods = np.empty((datasetSize, self.C))

        for batch in range(0, no_batches):
            start = batch_size * batch
            end = start + batch_size if batch < no_batches - 1 else datasetSize  # -1 WAS THE BUG
            curr_batch_sz = end - start

            stats_broadcast_i = np.reshape(predictionStats[start:end], (curr_batch_sz, self.L, self.A, 1, self.C2))
            tmp = np.sum(np.multiply(stats_broadcast_i, self.transition), axis=4)

            neighbDim = np.reshape(np.sum(predictionStats[start:end, 0,:, :], axis=2), (curr_batch_sz, 1, self.A, 1))
            neighbDim[neighbDim == 0] = 1

            tmp = tmp / neighbDim
            # sum(TotNxLxAx1xC2 x 1xLxAxCxC2) over last axis -> TotNxLxAxC
            tmp = np.sum(np.multiply(tmp, arcSbcast), axis=2)  # multiply for each i (and for each node) the vector S
            # tmp now is TotNxLxC
            s = np.sum(np.multiply(tmp, layerSbcast), axis=1)  # sum for all L values --> TotNxC

            prods[start:end] = np.multiply(s, self.emission[prediction_set[start:end], :])

        # print("INFERENCE")
        # print(prods)
        # print(np.argmax(prods, axis=1).astype('int'))
        return np.argmax(prods, axis=1).astype('int'), \
               np.reshape(predictionStats[:, 0, :, :], (len(adjacency_lists), 1, self.A, self.C2))

    def collective_inference(self, prediction_set, adjacency_lists, last_states, chosen, prevNeighbourStats=None):

        predictionStats = self.compute_statistics(adjacency_lists, last_states, prevNeighbourStats)

        arcSbcast = np.reshape(self.arcS, (1, self.L, self.A, 1))
        layerSbcast = np.reshape(self.layerS, (1, self.L, 1))

        datasetSize = len(prediction_set)

        stats_broadcast_i = np.reshape(predictionStats, (datasetSize, self.L, self.A, 1, self.C2))

        tmp = np.sum(np.multiply(stats_broadcast_i, self.transition), axis=4)

        neighbDim = np.reshape(np.sum(predictionStats, axis=3), (datasetSize, self.L, self.A, 1))
        neighbDim[neighbDim == 0] = 1

        tmp = tmp / neighbDim

        # sum(TotNxLxAx1xC2 x 1xLxAxCxC2) over last axis -> TotNxLxAxC
        tmp = np.sum(np.multiply(tmp, arcSbcast), axis=2)  # multiply for each i (and for each node) the vector S
        # tmp now is TotNxLxC
        s = np.sum(np.multiply(tmp, layerSbcast), axis=1)  # sum for all L values --> TotNxC

        prods_chosen = np.multiply(s[chosen], self.emission[prediction_set[chosen], :])

        dim_not_chosen = datasetSize - np.sum(chosen)
        assert dim_not_chosen == np.sum(np.logical_not(chosen))

        # NOW COMPUTE FOR NON chosen: i must find both y and Q that maximise the product
        prods_not_chosen = np.multiply(np.reshape(s[np.logical_not(chosen)], (dim_not_chosen, 1, self.C)),
                                       np.reshape(self.emission, (1, self.K, self.C)))

        unrolled_prods_not_chosen = np.reshape(prods_not_chosen, (dim_not_chosen, self.K * self.C))
        argmax = unrolled_prods_not_chosen.argmax(axis=1)  # returns dim_not_chosen indices, one for each row of dim K*C

        # TODO OPTIMIZE IT!
        states_not_chosen = np.empty(dim_not_chosen)
        emission_not_chosen = np.empty(dim_not_chosen)
        for i in range(0, dim_not_chosen):
            t = np.unravel_index(argmax[i], (self.K, self.C))
            emission_not_chosen[i] = t[0]
            states_not_chosen[i] = t[1]

        # best_indices = np.asarray(np.unravel_index(argmax, (self.K, self.C)), dtype='int')  # tuple: dim_not_chosenx2
        # print(best_indices.shape)

        best_states = np.empty(datasetSize, dtype='int')
        best_states[chosen] = np.argmax(prods_chosen, axis=1).astype('int')

        best_states[np.logical_not(chosen)] = states_not_chosen.astype('int')  # best_indices[:, 1]

        predicted = prediction_set
        predicted[np.logical_not(chosen)] = emission_not_chosen.astype('int')  # best_indices[:, 0]

        return predicted, best_states, \
               np.reshape(predictionStats[:, 0, :, :], (len(adjacency_lists), 1, self.A, self.C2))
