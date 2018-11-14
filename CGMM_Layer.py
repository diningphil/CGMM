import torch
import numpy as np


class CGMM_Layer(torch.nn.Module):
    def __init__(self, k, c, a, c2=None, l=None, radius=1):
        """
        utils Layer
        :param k: dimension of output's alphabet, which goes from 0 to K-1
        :param c: the number of hidden states
        :param c2: the number of states of the neighbours
        :param l: number of previous layers to consider. You must pass the appropriate number of statistics at training
        :param a: dimension of edges' alphabet, which goes from 0 to A-1
        """
        super().__init__()

        # For comparison w.r.t Numpy implementation
        # np.random.seed(seed=10)

        self.is_layer_0 = True
        if c2 is not None or l is not None:
            assert c2 is not None and l is not None, 'You should specify both C2, L and A'
            self.is_layer_0 = False

        self.eps = 1e-8  # Laplace smoothing
        self.C = c
        self.K = k
        if radius == 1:
            self.A = a + 1  # may consider a special case of the recurrent arc
        elif radius == 2:
            self.A = a * 2 + 1
        else:
            raise NotImplementedError('Radius can be 1 or 2')
        self.radius = radius

        if not self.is_layer_0:
            self.C2 = c2
            self.L = l

        # Initialisation of the model's parameters.

        if self.is_layer_0:
            # For debugging w.r.t Numpy version
            # pr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))

            pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float32))
            self.prior = pr / pr.sum()

            # print(self.prior)

        self.emission = torch.empty(self.K, self.C)

        for i in range(0, self.C):
            # For debugging w.r.t Numpy version
            # em = torch.from_numpy(np.random.uniform(size=self.K).astype(np.float32))

            em = torch.nn.init.uniform_(torch.empty(self.K, dtype=torch.float32))
            self.emission[:, i] = em / em.sum()

        # print(self.emission)

        if not self.is_layer_0:
            # For debugging w.r.t Numpy version
            # self.layerS = torch.from_numpy(np.random.uniform(size=self.L).astype(np.float32))  #

            self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float32))
            self.layerS /= self.layerS.sum()

            self.arcS = torch.zeros((self.L, self.A), dtype=torch.float32)
            self.transition = torch.empty([self.L, self.A, self.C, self.C2])

            for layer in range(0, self.L):
                # For debugging w.r.t Numpy version
                # elf.arcS[layer, :] = torch.from_numpy(np.random.uniform(size=self.A).astype(np.float32))

                self.arcS[layer, :] = torch.nn.init.uniform_(self.arcS[layer, :])
                self.arcS[layer, :] /= self.arcS[layer, :].sum()
                for arc in range(0, self.A):
                    for j in range(0, self.C2):
                        # For debugging w.r.t Numpy version
                        # tr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))

                        tr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float32))
                        self.transition[layer, arc, :, j] = tr / tr.sum()

            # print(self.arcS)
            # print(self.transition)

        self.init_accumulators()

    def build_dict(self):
        if self.is_layer_0:
            return {'is_layer_0': self.is_layer_0,
                    'C': self.C,
                    'K': self.K,
                    # remove the special arc for self-recurrency with prev layers
                    # it will be added again when loading the model
                    'A': (self.A - 1) // self.radius,  # can be only one or 2
                    'radius': self.radius,
                    'prior': self.prior.numpy(),
                    'emission': self.emission.numpy(),
                    }
        else:
            return {'is_layer_0': self.is_layer_0,
                    'C': self.C,
                    'K': self.K,
                    'C2': self.C2,
                    # remove the special arc for self-recurrency with prev layers
                    # it will be added again when loading the model
                    'A': (self.A - 1) // self.radius,  # can be only one or 2
                    'radius': self.radius,
                    'L': self.L,
                    'emission': self.emission.numpy(),
                    'transition': self.transition.numpy(),
                    'layerS': self.layerS.numpy(),
                    'arcS': self.arcS.numpy()

                    }

    @staticmethod
    def build_architecture(ckpt):
        layers = len(ckpt.keys())
        architecture = []

        for layer in range(layers):
            if layer == 0:
                cgmm_layer = CGMM_Layer(ckpt[layer]['K'], ckpt[layer]['C'],
                                        ckpt[layer]['A'], radius=ckpt[layer]['radius'])
                cgmm_layer.prior = torch.from_numpy(ckpt[layer]['prior'])
                cgmm_layer.emission = torch.from_numpy(ckpt[layer]['emission'])
            else:
                cgmm_layer = CGMM_Layer(ckpt[layer]['K'], ckpt[layer]['C'], ckpt[layer]['A'], ckpt[layer]['C2'],
                                        ckpt[layer]['L'], radius=ckpt[layer]['radius'])
                cgmm_layer.arcS = torch.from_numpy(ckpt[layer]['arcS'])
                cgmm_layer.layerS = torch.from_numpy(ckpt[layer]['layerS'])
                cgmm_layer.emission = torch.from_numpy(ckpt[layer]['emission'])
                cgmm_layer.transition = torch.from_numpy(ckpt[layer]['transition'])

            architecture.append(cgmm_layer)

        return architecture

    def compute_statistics(self, adjacency_lists, inferred_states, add_self_arc=False):
        # Compute statistics
        statistics = np.zeros((len(adjacency_lists), self.A, self.C), dtype=np.int64)

        for u in range(0, len(adjacency_lists)):
            incident_nodes = adjacency_lists[u]

            if add_self_arc:
                statistics[u, self.A-1, inferred_states[u]] += 1

            for u2, a in incident_nodes:
                node_state = inferred_states[u2]
                statistics[u, a, node_state] += 1

                if self.radius == 2:
                    # I have special arcs for this!
                    # looking for the neighbours of my neighbours
                    for u3, a2 in adjacency_lists[u2]:
                        node_state2 = inferred_states[u3]
                        statistics[u3, a2*2, node_state2] += 1

        return statistics

    def init_accumulators(self):

        # These are variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.prior_numerator = torch.full([self.C], self.eps)
        self.prior_denominator = self.eps*self.C

        self.emission_numerator = torch.full([self.K, self.C], self.eps)
        self.emission_denominator = torch.full([1, self.C], self.eps*self.K)

        if not self.is_layer_0:

            self.layerS_numerator = torch.full([self.L], self.eps)
            self.layerS_denominator = self.eps*self.L

            self.arcS_numerator = torch.full([self.L, self.A], self.eps)
            self.arcS_denominator = torch.full([self.L, 1], self.eps*self.A)

            self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps)
            self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps*self.C)

    def _compute_posterior_estimate(self, emission_for_labels, stats):

        batch_size = emission_for_labels.size()[0]

        # Compute the neighbourhood dimension for each vertex
        neighbDim = torch.sum(stats[:, 0, :, :], dim=2).float()  # --> ? x A

        # Replace zeros with ones to avoid divisions by zero
        # This does not alter learning: the numerator can still be zero

        neighbDim = torch.where(neighbDim == 0., torch.tensor([1.]),  neighbDim)

        broadcastable_transition = torch.unsqueeze(self.transition, 0)  # --> 1 x L x A x C x C2
        broadcastable_stats = torch.unsqueeze(stats, 3).float()  # --> ? x L x A x 1 x C2

        tmp = torch.sum(torch.mul(broadcastable_transition, broadcastable_stats), dim=4)  # --> ? x L x A x C

        broadcastable_layerS = torch.unsqueeze(self.layerS, 1)  # --> L x 1

        tmp2 = torch.reshape(torch.mul(broadcastable_layerS, self.arcS), [1, self.L, self.A, 1])  # --> 1 x L x A x 1

        div_neighb = torch.reshape(neighbDim, [batch_size, 1, self.A, 1]) # --> ? x 1 x A x 1

        tmp_unnorm_posterior_estimate = torch.div(torch.mul(tmp, tmp2), div_neighb)  # --> ? x L x A x C

        tmp_emission = torch.reshape(emission_for_labels,
                                     [batch_size, 1, 1, self.C])  # --> ? x 1 x 1 x C

        unnorm_posterior_estimate = torch.mul(tmp_unnorm_posterior_estimate, tmp_emission)  # --> ? x L x A x C

        # Normalize
        norm_constant = torch.reshape(torch.sum(unnorm_posterior_estimate, dim=[1, 2, 3]), [batch_size, 1, 1, 1])
        norm_constant = torch.where(norm_constant == 0., torch.Tensor([1.]), norm_constant)

        posterior_estimate = torch.div(unnorm_posterior_estimate, norm_constant)  # --> ? x L x A x C

        return posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb

    def _E_step(self, labels, stats=None):

        batch_size = labels.size()[0]

        emission_for_labels = torch.index_select(self.emission, dim=0, index=labels)  # ?xC

        if self.is_layer_0:
            # Broadcasting the prior
            numerator = torch.mul(emission_for_labels, torch.reshape(self.prior, shape=[1, self.C]))  # --> ?xC
            denominator = torch.matmul(emission_for_labels, torch.reshape(self.prior, shape=[self.C, 1]))  # --> ?x1

            posterior_estimate = torch.div(numerator, denominator)  # --> ?xC



            # Compute the expected complete log likelihood

            # -------------------------------- Likelihood ------------------------------- #

            likelihood = torch.sum(torch.mul(posterior_estimate, torch.log(numerator)))

            return likelihood, posterior_estimate

        else:

            posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb \
                = self._compute_posterior_estimate(emission_for_labels, stats)

            posterior_uli = torch.sum(posterior_estimate, dim=2)  # --> ? x L x C
            posterior_ui = torch.sum(posterior_uli, dim=1)  # --> ? x C

            # -------------------------------- Likelihood -------------------------------- #

            # NOTE: these terms can become expensive in terms of memory consumption, mini-batch computation is required.

            log_trans = torch.log(self.transition)

            num = torch.div(
                torch.mul(self.transition,
                            torch.mul(torch.reshape(self.layerS, [self.L, 1, 1, 1]),
                                        torch.reshape(self.arcS, [self.L, self.A, 1, 1]))),
                torch.unsqueeze(div_neighb, 4))

            num = torch.mul(num, torch.reshape(emission_for_labels, [batch_size, 1, 1, self.C, 1]))
            num = torch.mul(num, broadcastable_stats)

            den = torch.sum(num, dim=[1, 2, 3, 4], keepdim=True)  # --> ? x 1 x 1 x 1 x 1
            den = torch.where(torch.eq(den, 0.), torch.tensor([1.]), den)

            eulaij = torch.div(num, den)  # --> ? x L x A x C x C2

            # Compute the expected complete log likelihood
            likelihood1 = torch.sum(torch.mul(posterior_ui, torch.log(emission_for_labels)))
            likelihood2 = torch.sum(torch.mul(posterior_uli, torch.log(broadcastable_layerS)))
            likelihood3 = torch.sum(torch.mul(posterior_estimate,
                                                         torch.reshape(torch.log(self.arcS), [1, self.L, self.A, 1])))

            likelihood4 = torch.sum(torch.mul(torch.mul(eulaij, broadcastable_stats), log_trans))

            likelihood = likelihood1 + likelihood2 + likelihood3 + likelihood4

            return likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats


    def _M_step(self, labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats):

        if self.is_layer_0:

            tmp = torch.sum(posterior_estimate, dim=0)

            # These are used at each minibatch
            self.prior_numerator += tmp
            self.prior_denominator += torch.sum(tmp)

            labels = torch.squeeze(labels)  # removes dimensions of size 1 (current is ?x1)

            self.emission_numerator.index_add_(dim=0, source=posterior_estimate, index=labels)
            self.emission_denominator += tmp

        else:

            # These are equivalent to the multinomial mixture model, it just changes how the posterior is computed
            self.emission_numerator.index_add_(dim=0,  source=posterior_ui, index=labels)  # --> K x C
            self.emission_denominator += torch.sum(posterior_ui, dim=0)  # --> 1 x C

            tmp_arc_num = torch.sum(posterior_estimate, dim=[0, 3])  # --> L x A
            self.arcS_numerator += tmp_arc_num
            self.arcS_denominator += torch.unsqueeze(torch.sum(tmp_arc_num, dim=1), 1)  # --> L x 1

            new_layer_num = torch.sum(posterior_uli, dim=[0, 2])  # --> [L]
            self.layerS_numerator += new_layer_num
            self.layerS_denominator += torch.sum(new_layer_num)  # --> [1]

            new_trans_num = torch.sum(torch.mul(eulaij, broadcastable_stats), dim=0)
            self.transition_numerator += new_trans_num
            self.transition_denominator += torch.unsqueeze(torch.sum(new_trans_num, dim=2), 2)  # --> L x A x 1 x C2


    def update_parameters(self):
        self.emission = self.emission_numerator / self.emission_denominator

        if self.is_layer_0:
            self.prior = self.prior_numerator / self.prior_denominator

        else:
            self.layerS = self.layerS_numerator / self.layerS_denominator
            self.arcS = self.arcS_numerator / self.arcS_denominator

            self.transition = self.transition_numerator / self.transition_denominator

    def forward(self, labels, stats=None):
        '''
        Implementation of nn.Module interface. Performs inference
        '''

        emission_for_labels = torch.index_select(self.emission, dim=0, index=labels)  # ?xC

        if self.is_layer_0:
            posterior = torch.mul(emission_for_labels, torch.reshape(self.prior, shape=[1, self.C]))  # --> ?xC

            # Use posterior is used to smooth a bit the fingerprints. For example, argmax may choose 1 state even if
            # it has probability 0.51 (in the case of C=2), while I would like to consider the entire distribution
            return posterior
        else:
            posterior_estimate, _, _, _ = self._compute_posterior_estimate(emission_for_labels, stats)
            posterior_ui = torch.sum(posterior_estimate, dim=[1, 2])  # --> ? x C

            return posterior_ui

    def EM_step(self, labels, stats=None):
        if self.is_layer_0:
            likelihood, posterior_estimate = self._E_step(labels, stats)
            self._M_step(labels, posterior_estimate, None, None, None, None)

        else:
            likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats \
                = self._E_step(labels, stats)
            self._M_step(labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats)

        return likelihood
