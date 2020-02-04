import torch
import numpy as np
from models.utils.CategoricalEmission import CategoricalEmission
from models.utils.GaussianEmission import GaussianEmission
from models.gnn_wrapper.NetWrapper import NetWrapper
from models.graph_classifiers.MLP import MLP
from models.graph_classifiers.LinearModel import LinearModel
from models.incremental_architectures.IncrementalModel import IncrementalLayer
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import global_mean_pool, global_add_pool

from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier


class CGMM(IncrementalLayer):

    def __init__(self, dim_features, dim_target, depth, layer_config, checkpoint=None,
                 loss_class=None, optim_class=None, sched_class=None, stopper_class=None, clipping=None, **kwargs):
        """
        CGMM
        :param k: dimension of a vertex output's alphabet, which goes from 0 to K-1 (when discrete)
        :param a: dimension of an edge output's alphabet, which goes from 0 to A-1
        :param cn: vertexes latent space dimension
        :param ca: edges latent space dimension
        :param l: number of previous layers to consider. You must pass the appropriate number of statistics at training
        """
        super().__init__(dim_features, dim_target, depth, layer_config, checkpoint,
                         loss_class, optim_class, sched_class, stopper_class, clipping, **kwargs)

        if checkpoint:
            print(f'Restoring checkpoint for layer {depth} of NN4G...')
            self.restore(checkpoint)
        else:
            print(f'Initializing layer {depth} of CGMM...')
            self.K = dim_features

            # TODO HANDLE DIFFERENT DATASETS FEATURES AND FIX OTHER THINGS (RUN TO SEE)

            self.node_type = layer_config['node_type']
            self.L = None  # tbd at training time, it may vary
            self.layer = None

            self.A = layer_config['A']
            self.C = layer_config['C']
            self.C2 = layer_config['C'] + 1
            self.add_self_arc = layer_config['self_arc'] if 'self_arc' in layer_config else False

            self.max_epochs = layer_config['max_epochs'] if 'max_epochs' in layer_config else 10
            self.threshold = layer_config['threshold'] if 'threshold' in layer_config else 0.
            self.use_continuous_states = layer_config['infer_with_posterior']
            self.unibigram = layer_config['unibigram']
            self.aggregation = layer_config['aggregation']

    def train_layer(self, data_loader, L, concat_axis, validation_loader=None, device='cpu'):

        if self.is_first_layer:
            self.layer = CGMMLayer(self.K, self.C, self.A, node_type=self.node_type, device=device)
        else:
            self.L = L
            self.layer = CGMMLayer(self.K, self.C, self.A, self.C2, self.L, node_type=self.node_type, device=device)
        # This is the layer responsible for training on VERTEXES

        '''
        v_outs contains the previous statistics. The tensor has shape VxLxAxC2.
        You may put the vertex states in
        '''

        # ---------- Expectation-Maximization Algorithm ---------- #

        self.layer.emission.to(device)
        if not self.layer.is_layer_0:
            self.layer.layerS.to(device)
            self.layer.arcS.to(device)
            self.layer.transition.to(device)


        old_likelihood = -float('inf')
        for epoch in range(1, self.max_epochs+1):
            likelihood = 0.

            # ---------- E-Step ---------- #

            for data in data_loader:

                data = data.to(device)

                x = data.x
                
                if self.is_first_layer:
                    statistics = None
                else:
                    statistics = data.o_outs
                    statistics.to(device)


                # todo I would like to avoid doing argmax every time
                # todo this does not work with PPI
                #data.x = torch.argmax(data.x, dim=1)

                l, eulaij = self.layer.E_step(x, statistics)
                likelihood += l.detach().item()

            # ---------------------------- #

            print('likelihood at epoch', epoch, ':', likelihood)
            if likelihood - old_likelihood <= self.threshold:
                break
            else:
                old_likelihood = likelihood

            # ---------- M-Step ---------- #

            self.layer.M_step()  # batch update

            # ---------------------------- #

        # ------------------------------------------------------- #

        # ------------------ Inference -------------------------- #

        v_out, e_out, g_out, statistics = self.infer(data_loader, device)

        # ------------------------------------------------------- #

        return v_out, e_out, g_out, statistics

    def infer(self, data_loader, device='cpu', output_v_out=True, output_g_out=True):

        self.layer.emission.to(device)
        if not self.layer.is_layer_0:
            self.layer.layerS.to(device)
            self.layer.arcS.to(device)
            self.layer.transition.to(device)

        v_out = []
        statistics = []
        graph_embeddings = []

        for data in data_loader:

            data = data.to(device)

            if self.is_first_layer:
                prev_stats = None
            else:
                prev_stats = data.o_outs
                prev_stats.to(device)

            # todo I would like to avoid doing argmax every time
            # todo this does not work with PPI
            #data.x = torch.argmax(data.x, dim=1)

            posterior_batch, l_batch = self.layer.forward(data.x, prev_stats)
            statistics_batch = self._compute_statistics(posterior_batch, data, device)

            node_unigram, graph_unigram = self._compute_unigram(posterior_batch, data.batch, device)

            if self.unibigram:
                node_bigram, graph_bigram = self._compute_bigram(posterior_batch.double(), data.edge_index, data.batch, graph_unigram.shape[0], device)

                node_embeddings_batch = torch.cat((node_unigram, node_bigram), dim=1)
                graph_embeddings_batch = torch.cat((graph_unigram, graph_bigram), dim=1)
            else:
                node_embeddings_batch = node_unigram
                graph_embeddings_batch = graph_unigram

            # ------------- Revert back to list of graphs ----------- #

            sizes = self._compute_sizes(data.batch, device)
            cum_nodes = 0
            for i, size in enumerate(sizes):
                v_out.append(node_embeddings_batch[cum_nodes:cum_nodes + size, :])
                graph_embeddings.append(torch.unsqueeze(graph_embeddings_batch[i], dim=0))
                statistics.append(statistics_batch[cum_nodes:cum_nodes + size, :])
                cum_nodes += size

            # ------------------------------------------------------- #

        return v_out, None, graph_embeddings, statistics

    def arbitrary_logic(self, train_loader, layer_config, is_last_layer, validation_loader=None, test_loader=None,
                        logger=None, device='cpu'):
        if is_last_layer:
            dim_features = self.C*self.depth if not self.unibigram else self.C*self.C2*self.depth

            # torch.manual_seed(0)

            model = MLP(dim_features, self.dim_target, layer_config)

            net = CGMMGraphClassifier(model, loss_function=self.loss_class(), device=device)

            optimizer = self.optim_class(model.parameters(),
                                         lr=layer_config['learning_rate'], weight_decay=layer_config['l2'])
            if self.sched_class is not None:
                scheduler = self.sched_class(optimizer)
            else:
                scheduler = None

            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, _ = \
                net.train(train_loader=train_loader, max_epochs=layer_config['classifier_epochs'],
                          optimizer=optimizer, scheduler=scheduler,
                          validation_loader=validation_loader, test_loader=test_loader,
                          early_stopping=self.stopper_class, clipping=self.clipping)

            return {'train_score': train_acc, 'validation_score': val_acc, 'test_score': test_acc}
        else:
            return {}

    def stopping_criterion(self, depth, dict_per_layer, layer_config, logger=None):
        return False

    def checkpoint(self):
        """
        :return: layer state dict to store
        """
        assert self.layer is not None, 'You should train the model before doing a checkpoint!'
        return self.layer.checkpoint()

    def restore(self, checkpoint):
        assert self.layer is not None, 'You should train the model before doing a checkpoint!'
        self.layer = CGMMLayer.restore(checkpoint)

    def _compute_statistics(self, posteriors, data, device):

        # Compute statistics
        if 'cuda' in device:
            statistics = torch.full((posteriors.shape[0], self.A + 2, self.C2), 1e-8, dtype=torch.float64).cuda()
        else:
            statistics = torch.full((posteriors.shape[0], self.A + 2, self.C2), 1e-8, dtype=torch.float64)

        srcs, dsts = data.edge_index

        if self.A == 1:
            for source, dest, in zip(srcs, dsts):
                statistics[dest, 0, :-1] += posteriors[source]
        else:
            arc_labels = data.edge_attr

            for source, dest, arc_label in zip(srcs, dsts, arc_labels):
                statistics[dest, arc_label, :-1] += posteriors[source]

        if self.add_self_arc:
            statistics[:, self.A, :-1] += posteriors

        # use self.A+1 as special edge for bottom states (all in self.C2-1)
        degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()

        max_arieties, _ = self._compute_max_ariety(degrees.int(), data.batch)
        statistics[:, self.A + 1, self.C2 - 1] += 1 - (degrees / max_arieties[data.batch].double())
        return statistics

    def _compute_unigram(self, posteriors, batch, device):

        aggregate = self._get_aggregation_fun()

        if self.use_continuous_states:
            node_embeddings_batch = posteriors
            graph_embeddings_batch = aggregate(posteriors, batch)
        else:
            # todo may avoid one hot with this
            # np.add.at(freq_node_unigram, node_states[curr_node_size:(curr_node_size + size)], 1)

            if 'cuda' in device:
                node_embeddings_batch = self._make_one_hot(posteriors.argmax(dim=1)).cuda()
            else:
                node_embeddings_batch = self._make_one_hot(posteriors.argmax(dim=1))
            graph_embeddings_batch = aggregate(node_embeddings_batch, batch)

        return node_embeddings_batch.double(), graph_embeddings_batch.double()

    def _compute_bigram(self, posteriors, edge_index, batch, no_graphs, device):

        node_bigram_batch = torch.zeros((posteriors.shape[0], self.C * self.C), dtype=torch.float64).to(device)
        graph_bigram_batch = torch.zeros((no_graphs, self.C * self.C), dtype=torch.float64).to(device)

        if self.use_continuous_states:
            srcs, dsts = edge_index
            for dest, source in zip(dsts, srcs):
                for i in range(self.C):
                    start, end = i * self.C, i * self.C + self.C

                    node_bigram_batch[dest, start:end] += posteriors[source]*posteriors[dest, i]
                    graph_bigram_batch[batch[dest], start:end] += posteriors[source]*posteriors[dest, i]
        else:
            posterior_batch_argmax = posteriors.argmax(dim=1)

            srcs, dsts = edge_index

            for dest, source in zip(dsts, srcs):
                state_u = posterior_batch_argmax[dest]
                state_neighb = posterior_batch_argmax[source]

                node_bigram_batch[dest, state_u * self.C + state_neighb] += 1
                graph_bigram_batch[batch[dest], state_u * self.C + state_neighb] += 1
        
        return node_bigram_batch.double(), graph_bigram_batch.double()

    def _compute_sizes(self, batch, device):
        return scatter_add(torch.ones(len(batch), dtype=torch.int).to(device), batch)

    def _compute_max_ariety(self, degrees, batch):
        return scatter_max(degrees, batch)

    def _get_aggregation_fun(self):
        if self.aggregation == 'mean':
            aggregate = global_mean_pool
        elif self.aggregation == 'sum':
            aggregate = global_add_pool
        return aggregate

    def _make_one_hot(self, labels):
        one_hot = torch.zeros(labels.size(0), self.C)

        # todo inefficient
        for u in range(labels.size(0)):
            one_hot[u, labels[u]] += 1

        return one_hot


class CGMMPPI(CGMM):

    def arbitrary_logic(self, train_loader, layer_config, is_last_layer, validation_loader=None, test_loader=None,
                        logger=None, device='cpu'):
        if is_last_layer:
            dim_features = self.C*self.depth if not self.unibigram else self.C*self.C2*self.depth

            dim_features = dim_features + self.dim_features

            # Assume all dataset in one batch
            for data in train_loader:
                H_train =  torch.cat((data.x.float(), torch.reshape(data.v_outs, (data.v_outs.shape[0], -1)).float()), dim=1).float().cpu().numpy()
                # TODO append input as well
                y_train = data.y.cpu().numpy()

            print(H_train.shape, y_train.shape)

            H_valid = None            
            if validation_loader is not None:
                for data in validation_loader:
                    H_valid =  torch.cat((data.x.float(), torch.reshape(data.v_outs, (data.v_outs.shape[0], -1)).float()), dim=1).float().cpu().numpy()
                    # TODO append input as well
                    y_valid = data.y.cpu().numpy()

            H_test = None            
            if test_loader is not None:
                for data in test_loader:
                    H_test = torch.cat((data.x.float(), torch.reshape(data.v_outs, (data.v_outs.shape[0], -1)).float()), dim=1).float().cpu().numpy()
                    # TODO append input as well
                    y_test = data.y.cpu().numpy()

            model = LinearModel(dim_features, self.dim_target, layer_config)

            #model = MLP(dim_features, self.dim_target, layer_config)

            net = CGMMPPIGraphClassifier(model, loss_function=self.loss_class(), device=device)

            optimizer = self.optim_class(model.parameters(),
                                         lr=layer_config['learning_rate'], weight_decay=layer_config['l2'])
            if self.sched_class is not None:
                scheduler = self.sched_class(optimizer)
            else:
                scheduler = None

            train_loss, micro_f1_train, val_loss, micro_f1_valid, test_loss, micro_f1_test, _ = \
                net.train(train_loader=train_loader, max_epochs=layer_config['classifier_epochs'],
                          optimizer=optimizer, scheduler=scheduler,
                          validation_loader=validation_loader, test_loader=test_loader,
                          early_stopping=self.stopper_class, clipping=self.clipping)

            print({'train_score': micro_f1_train, 'validation_score': micro_f1_valid, 'test_score': micro_f1_test})

            return {'train_score': micro_f1_train, 'validation_score': micro_f1_valid, 'test_score': micro_f1_test}
        else:
            return {}


class CGMMLayer:
    def __init__(self, k, c, a, c2=None, l=None, node_type='discrete', device='cpu'):
        """
        utils Layer
        :param k: dimension of output's alphabet, which goes from 0 to K-1 (when discrete)
        :param c: the number of hidden states
        :param c2: the number of states of the neighbours
        :param l: number of previous layers to consider. You must pass the appropriate number of statistics at training
        :param a: dimension of edges' alphabet, which goes from 0 to A-1
        """
        super().__init__()
        self.device = device
        # For comparison w.r.t Numpy implementation
        # np.random.seed(seed=10)
        self.node_type = node_type
        self.is_layer_0 = True
        if c2 is not None or l is not None:
            assert c2 is not None and l is not None, 'You should specify both C2, L and A'
            self.is_layer_0 = False

        self.eps = 1e-8  # Laplace smoothing
        self.C = c
        self.K = k
        self.orig_A = a
        self.A = a + 2  # may consider a special case of the recurrent arc and the special case of bottom state

        if not self.is_layer_0:
            self.C2 = c2
            self.L = l

        # Initialisation of the model's parameters.
        # torch.manual_seed(0)

        if self.is_layer_0:
            # For debugging w.r.t Numpy version
            # pr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))
            if 'cuda' in device:
                pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float64)).cuda()
            else:
                pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float64))
            self.prior = pr / pr.sum()

            # print(self.prior)

        if self.node_type == 'discrete':
            self.emission = CategoricalEmission(self.K, self.C, device)
        elif self.node_type == 'continuous':
            self.emission = GaussianEmission(self.K, self.C, device)

        # print(self.emission)

        if not self.is_layer_0:
            # For debugging w.r.t Numpy version
            # self.layerS = torch.from_numpy(np.random.uniform(size=self.L).astype(np.float32))  #
            if 'cuda' in device:
                self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float64)).cuda()
                self.arcS = torch.zeros((self.L, self.A), dtype=torch.float64).cuda()
                self.transition = torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float64).cuda()
            else:
                self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float64))
                self.arcS = torch.zeros((self.L, self.A), dtype=torch.float64)
                self.transition = torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float64)

            self.layerS /= self.layerS.sum()

            for layer in range(0, self.L):
                # For debugging w.r.t Numpy version
                # elf.arcS[layer, :] = torch.from_numpy(np.random.uniform(size=self.A).astype(np.float32))

                self.arcS[layer, :] = torch.nn.init.uniform_(self.arcS[layer, :])
                self.arcS[layer, :] /= self.arcS[layer, :].sum()
                for arc in range(0, self.A):
                    for j in range(0, self.C2):
                        # For debugging w.r.t Numpy version
                        # tr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))

                        tr = torch.nn.init.uniform_(torch.empty(self.C))
                        self.transition[layer, arc, :, j] = tr / tr.sum()

            # print(self.arcS)
            # print(self.transition)

        self.init_accumulators()

    def checkpoint(self):
        if self.is_layer_0:
            return {'is_layer_0': self.is_layer_0,
                    'node_type': self.node_type,
                    'C': self.C,
                    'K': self.K,
                    # remove the special arc for self-recurrency with prev layers
                    # it will be added again when loading the model
                    'A': self.orig_A,
                    'prior': self.prior,
                    'emission': self.emission.export_parameters(),
                    }
        else:
            return {'is_layer_0': self.is_layer_0,
                    'node_type': self.node_type,
                    'C': self.C,
                    'K': self.K,
                    'C2': self.C2,
                    # remove the special arc for self-recurrency with prev layers
                    # it will be added again when loading the model
                    'A': self.orig_A,
                    'L': self.L,
                    'emission': self.emission.export_parameters(),
                    'transition': self.transition,
                    'layerS': self.layerS,
                    'arcS': self.arcS

                    }

    @staticmethod
    def restore(ckpt):
        if ckpt['is_layer_0']:
            layer = CGMMLayer(ckpt['K'], ckpt['C'],
                              ckpt['A'],
                              node_type=ckpt['node_type'])
            layer.prior = torch.from_numpy(ckpt['prior'])
            layer.emission.import_parameters(ckpt['emission'])
        else:
            layer = CGMMLayer(ckpt['K'], ckpt['C'], ckpt['A'], ckpt['C2'],
                              ckpt['L'],
                              node_type=ckpt['node_type'])
            layer.arcS = torch.from_numpy(ckpt['arcS'])
            layer.layerS = torch.from_numpy(ckpt['layerS'])
            layer.emission.import_parameters(ckpt['emission'])
            layer.transition = torch.from_numpy(ckpt['transition'])

        return layer

    def init_accumulators(self):

        # These are variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.emission.init_accumulators()

        if self.is_layer_0:
            if 'cuda' in self.device:
                self.prior_numerator = torch.full([self.C], self.eps, dtype=torch.float64).cuda()
            else:
                self.prior_numerator = torch.full([self.C], self.eps, dtype=torch.float64)
            self.prior_denominator = self.eps * self.C

        else:
            if 'cuda' in self.device:

                self.layerS_numerator = torch.full([self.L], self.eps, dtype=torch.float64).cuda()
                self.arcS_numerator = torch.full([self.L, self.A], self.eps, dtype=torch.float64).cuda()
                self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps, dtype=torch.float64).cuda()
                self.arcS_denominator = torch.full([self.L, 1], self.eps * self.A, dtype=torch.float64).cuda()
                self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps * self.C,
                                                     dtype=torch.float64).cuda()
            else:
                self.layerS_numerator = torch.full([self.L], self.eps, dtype=torch.float64)
                self.arcS_numerator = torch.full([self.L, self.A], self.eps, dtype=torch.float64)
                self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps, dtype=torch.float64)
                self.arcS_denominator = torch.full([self.L, 1], self.eps * self.A, dtype=torch.float64)
                self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps * self.C,
                                                         dtype=torch.float64)

            self.layerS_denominator = self.eps * self.L

    def _compute_posterior_estimate(self, emission_for_labels, stats):

        # print(stats.shape)

        batch_size = emission_for_labels.size()[0]

        # Compute the neighbourhood dimension for each vertex
        neighbDim = torch.sum(stats[:, :, :, :], dim=3).float()  # --> ? x L x A

        # Replace zeros with ones to avoid divisions by zero
        # This does not alter learning: the numerator can still be zero

        neighbDim = torch.where(neighbDim == 0., torch.tensor([1.]).to(self.device), neighbDim)
        neighbDim[:, :, -1] = 1 

        broadcastable_transition = torch.unsqueeze(self.transition, 0)  # --> 1 x L x A x C x C2
        broadcastable_stats = torch.unsqueeze(stats, 3).double()  # --> ? x L x A x 1 x C2

        tmp = torch.sum(torch.mul(broadcastable_transition, broadcastable_stats), dim=4)  # --> ? x L x A x C2

        broadcastable_layerS = torch.unsqueeze(self.layerS, 1)  # --> L x 1

        tmp2 = torch.reshape(torch.mul(broadcastable_layerS, self.arcS), [1, self.L, self.A, 1])  # --> 1 x L x A x 1

        div_neighb = torch.reshape(neighbDim, [batch_size, self.L, self.A, 1]).double()  # --> ? x L x A x 1

        tmp_unnorm_posterior_estimate = torch.div(torch.mul(tmp, tmp2), div_neighb)  # --> ? x L x A x C2

        tmp_emission = torch.reshape(emission_for_labels,
                                     [batch_size, 1, 1, self.C])  # --> ? x 1 x 1 x C2

        unnorm_posterior_estimate = torch.mul(tmp_unnorm_posterior_estimate, tmp_emission)  # --> ? x L x A x C2

        # Normalize
        norm_constant = torch.reshape(torch.sum(unnorm_posterior_estimate, dim=[1, 2, 3]), [batch_size, 1, 1, 1])
        norm_constant = torch.where(norm_constant == 0., torch.Tensor([1.]).double().to(self.device), norm_constant)

        posterior_estimate = torch.div(unnorm_posterior_estimate, norm_constant)  # --> ? x L x A x C2

        return posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb

    def _E_step(self, labels, stats=None):
        batch_size = labels.size()[0]

        emission_of_labels = self.emission.get_distribution_of_labels(labels)

        if self.is_layer_0:
            # Broadcasting the prior
            numerator = torch.mul(emission_of_labels, torch.reshape(self.prior, shape=[1, self.C]))  # --> ?xC

            denominator = torch.sum(numerator, dim=1, keepdim=True)

            posterior_estimate = torch.div(numerator, denominator)  # --> ?xC

            # -------------------------------- Likelihood ------------------------------- #

            likelihood = torch.sum(torch.mul(posterior_estimate, torch.log(numerator)))

            return likelihood, posterior_estimate

        else:

            posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb \
                = self._compute_posterior_estimate(emission_of_labels, stats)

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

            num = torch.mul(num, torch.reshape(emission_of_labels, [batch_size, 1, 1, self.C, 1]))
            num = torch.mul(num, broadcastable_stats)

            den = torch.sum(num, dim=[1, 2, 3, 4], keepdim=True)  # --> ? x 1 x 1 x 1 x 1
            den = torch.where(torch.eq(den, 0.), torch.tensor([1.]).double().to(self.device), den)

            eulaij = torch.div(num, den)  # --> ? x L x A x C x C2

            # Compute the expected complete log likelihood
            likelihood1 = torch.sum(torch.mul(posterior_ui, torch.log(emission_of_labels)))
            likelihood2 = torch.sum(torch.mul(posterior_uli, torch.log(broadcastable_layerS)))
            likelihood3 = torch.sum(torch.mul(posterior_estimate,
                                              torch.reshape(torch.log(self.arcS), [1, self.L, self.A, 1])))

            likelihood4 = torch.sum(torch.mul(torch.mul(eulaij, broadcastable_stats), log_trans))

            likelihood = likelihood1 + likelihood2 + likelihood3 + likelihood4

            return likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats

    def E_step(self, labels, stats=None):

        with torch.no_grad():
            if self.is_layer_0:
                likelihood, posterior_estimate = self._E_step(labels, stats)
                self._M_step(labels, posterior_estimate, None, None, None, None)

                return likelihood, posterior_estimate

            else:
                likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats \
                    = self._E_step(labels, stats)
                self._M_step(labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats)

                return likelihood, eulaij

    def _M_step(self, labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats):

        if self.is_layer_0:

            tmp = torch.sum(posterior_estimate, dim=0)
            # These are used at each minibatch
            self.prior_numerator += tmp
            self.prior_denominator += torch.sum(tmp)

            self.emission.update_accumulators(posterior_estimate, labels)

        else:

            # These are equivalent to the categorical mixture model, it just changes how the posterior is computed
            self.emission.update_accumulators(posterior_ui, labels)

            tmp_arc_num = torch.sum(posterior_estimate, dim=[0, 3])  # --> L x A
            self.arcS_numerator += tmp_arc_num
            self.arcS_denominator += torch.unsqueeze(torch.sum(tmp_arc_num, dim=1), 1)  # --> L x 1

            new_layer_num = torch.sum(posterior_uli, dim=[0, 2])  # --> [L]
            self.layerS_numerator += new_layer_num
            self.layerS_denominator += torch.sum(new_layer_num)  # --> [1]

            new_trans_num = torch.sum(torch.mul(eulaij, broadcastable_stats), dim=0)
            self.transition_numerator += new_trans_num
            self.transition_denominator += torch.unsqueeze(torch.sum(new_trans_num, dim=2), 2)  # --> L x A x 1 x C2

    def M_step(self):

        self.emission.update_parameters()
        if self.is_layer_0:
            self.prior = self.prior_numerator / self.prior_denominator

        else:

            self.layerS = self.layerS_numerator / self.layerS_denominator
            self.arcS = self.arcS_numerator / self.arcS_denominator

            self.transition = self.transition_numerator / self.transition_denominator

        # I need to re-init accumulators, otherwise they will contain statistics of the previous epochs
        self.init_accumulators()

    def forward(self, labels, stats=None):
        """
        Implementation of nn.Module interface. Performs inference
        """

        with torch.no_grad():

            if self.is_layer_0:
                likelihood, posterior_estimate = self._E_step(labels, stats)
                likelihood = likelihood.detach().item()

                # print(posterior_estimate[:10, :])

                # Use posterior is used to smooth a bit the fingerprints. For example, argmax may choose 1 state even if
                # it has probability 0.51 (in the case of C=2), while I would like to consider the entire distribution
                return posterior_estimate, likelihood
            else:
                likelihood, _, _, posterior_ui, _, _ = self._E_step(labels, stats)
                likelihood = likelihood.detach().item()

                return posterior_ui, likelihood


class CGMMGraphClassifier(NetWrapper):

    # TODO I could make NetWrapper more flexible allowing to preprocess the batch before giving it to the model!

    def _train(self, train_loader, optimizer, clipping=None):

        model = self.model.to(self.device)

        model.train()

        loss_all = 0
        acc_all = 0
        for data in train_loader:

            data = data.to(self.device)
            optimizer.zero_grad()

            # concat vertex embeddings of different layers
            g_outs = torch.reshape(data.g_outs, (data.g_outs.shape[0], -1)).float()
            output = model(g_outs, data.batch)
            if not isinstance(output, tuple):
                output = (output,)

            #print(data.y.shape, output[0].shape)
            loss, acc = self.loss_fun(data.y, *output)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            acc_all += acc.item() * data.num_graphs
            optimizer.step()

        return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)

    def predict(self, loader):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        acc_all = 0
        for data in loader:
            data = data.to(self.device)

            # concat vertex embeddings of different layers
            g_outs = torch.reshape(data.g_outs, (data.g_outs.shape[0], -1)).float()
            output = model(g_outs, data.batch)
            if not isinstance(output, tuple):
                output = (output,)

            loss, acc = self.loss_fun(data.y, *output)
            loss_all += loss.item() * data.num_graphs
            acc_all += acc.item() * data.num_graphs

        return acc_all / len(loader.dataset), loss_all / len(loader.dataset)


class CGMMPPIGraphClassifier(NetWrapper):

    def _train(self, train_loader, optimizer, clipping=None):

        self.mu = None
        self.std = None

        model = self.model.to(self.device)

        model.train()

        loss_all = 0
        acc_all = 0
        for data in train_loader:

            data = data.to(self.device)
            optimizer.zero_grad()

            x = data.x

            # concat vertex embeddings of different layers
            # print(data)
            v_outs = torch.reshape(data.v_outs, (data.v_outs.shape[0], -1)).float()

            # apply sigmoid after the classifier (and introduce the vertex features as well)
            output = torch.sigmoid(model(torch.cat((v_outs, x), dim=1), data.batch))

            # print(data.y.shape, output[0].shape)
            loss, acc = self.loss_fun(data.y.long(), output)
            loss.backward()
            loss_all += loss.item()
            acc_all += acc.item()
            optimizer.step()

        return acc_all, loss_all

    def predict(self, loader):
        model = self.model.to(self.device)
        model.eval()

        loss_all = 0
        acc_all = 0
        for data in loader:
            data = data.to(self.device)

            x = data.x

            # concat vertex embeddings of different layers
            # print(data)
            v_outs = torch.reshape(data.v_outs, (data.v_outs.shape[0], -1)).float()

            # apply sigmoid after the classifier
            output = torch.sigmoid(model(torch.cat((v_outs, x), dim=1), data.batch))

            loss, acc = self.loss_fun(data.y.long(), output)
            loss_all += loss.item()
            acc_all += acc.item()

        return acc_all, loss_all
