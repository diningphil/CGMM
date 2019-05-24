import time
#import seaborn as sns
from torch.utils.data import DataLoader
from models.CGMM.CGMM_Layer import CGMM_Layer
from models.Utilities import *

checkpoint_folder = 'checkpoints'
#sns.set()


def current_milli_time():
    return int(round(time.time() * 1000))


class CGMM(torch.nn.Module):
    def __init__(self, k, a, cn, use_statistics, node_type='discrete'):
        """
        GCGN
        :param k: dimension of a vertex output's alphabet, which goes from 0 to K-1 (when discrete)
        :param a: dimension of an edge output's alphabet, which goes from 0 to A-1
        :param cn: vertexes latent space dimension
        :param ca: edges latent space dimension
        :param l: number of previous layers to consider. You must pass the appropriate number of statistics at training
        """
        super().__init__()
        self.K = k
        self.A = a
        self.CN = cn
        self.CN2 = cn + 1
        self.use_statistics = use_statistics
        self.node_type = node_type

    def incremental_training(self, adjacency_lists, node_dataset, edges, sizes, max_arieties, layers,
                             add_self_arc, exp_name, threshold=0, max_epochs=100,
                             prev_architecture=None, prev_stats_path=None, ckpt_name=None, node_type='discrete',
                             edge_type=None):
        """
        Build an architecture. Assumes CN is equal to self.CN2 and CA equal to CA2
        :param use_statistics: an array with a subset of preceding layers to consider. For example, np.array([1,2,3])
        considers the immediate 3 preceding layers, whenever possible.
        :param adjacency_lists: a list of lists (one for each vertex) representing the connections between vertexes
        :param label_dataset: a Dataset.infer_'
        :param layers: the depth of the architecture
        :param threshold: the threshold parameter for the EM algorithm
        :param max_epochs: the maximum number of epochs per training
        :param prev_architecture:
        :param ckpt_name:
        :return: an architecture
        """
        stats_folderpath = 'stats_train'
        stats_filename = 'stats'
        store_stats_folderpath = os.path.join(exp_name, stats_folderpath)
        load_stats_folderpath = store_stats_folderpath

        if not os.path.exists(os.path.join(exp_name, checkpoint_folder)):
            os.makedirs(os.path.join(exp_name, checkpoint_folder))

        if not os.path.exists(store_stats_folderpath):
            os.makedirs(store_stats_folderpath)

        if prev_architecture is None:
            architecture = []
            architecture_dict = {}
            len_prev_architecture = 0
        else:
            # I will load stats from another (winner) folder. Used for pooling
            assert prev_stats_path is not None
            load_stats_folderpath = os.path.join(prev_stats_path, stats_folderpath)
            architecture, architecture_dict = prev_architecture

            # Create shallow copies of the arcitectures
            architecture = list(architecture)
            architecture_dict = dict(architecture_dict)
            len_prev_architecture = len(architecture)
            assert layers > len_prev_architecture, str(layers) + ' e ' + str(len_prev_architecture)

        for layer in range(len_prev_architecture, layers):

            print("Training Layer", layer+1)
            # e.g 1 - [1, 3] = [0, -2] --> [0]
            # e.g 5 - [1, 3] = [4, 2]  --> [4, 2]
            layer_wise_statistics = [(layer - x + 1) for x in self.use_statistics if (layer - x) >= 0]

            self.L = len(layer_wise_statistics)

            

            # This is the layer responsible for training on VERTEXES
            cgmm_layer = CGMM_Layer(self.K, self.CN, self.A, node_type=node_type) if layer == 0 \
                else CGMM_Layer(self.K, self.CN, self.A, self.CN2, self.L, node_type=node_type)

            if cgmm_layer.is_layer_0:
                stats_nodes = None
            else:
                # Load the statistics for the node model
                stats_to_load = [os.path.join(load_stats_folderpath, stats_filename + '_node_' + str(level) + '.npy')
                                 for level in layer_wise_statistics]
                stats_nodes = load_to_ZipDataset(stats_to_load)

            train_set = LabelAndStatsDataset(node_dataset, stats_nodes)
            train_set = DataLoader(train_set, batch_size=4000, shuffle=False, num_workers=2)

            old_likelihood = -np.inf
            for epoch in range(max_epochs):
                likelihood = 0.
                for input_batch in train_set:
                    l, eulaij = cgmm_layer.EM_step(*input_batch)
                    likelihood += l.detach().numpy()
                # If you want to perform a batch update of parameters use this method
                cgmm_layer.update_parameters()
                print('likelihood at epoch', epoch+1, ':', likelihood)
                if likelihood - old_likelihood <= threshold:
                    break
                else:
                    old_likelihood = likelihood

                '''
                if not cgmm_layer.is_layer_0:
                    print(cgmm_layer.arcS)
                '''

            # Perform inference at the end of training and accumulate results
            # node_inferred_states = None
            posteriors = None
            for input_batch in train_set:
                posterior_batch, batch_likelihood = cgmm_layer(*input_batch)

                '''
                # Always used for statistics
                prediction_batch = torch.argmax(posterior_batch, dim=1).detach().numpy()
                if node_inferred_states is None:
                    node_inferred_states = prediction_batch
                else:
                    node_inferred_states = np.append(node_inferred_states, prediction_batch)
                '''

                if posteriors is None:
                    posteriors = posterior_batch.detach().numpy()
                else:
                    posteriors = np.concatenate((posteriors, posterior_batch.detach().numpy()), axis=0)

            # Save statistics for this layer
            save_statistics(self,
                            adjacency_lists, posteriors, edges, sizes, max_arieties,
                            store_stats_folderpath, stats_filename, layer+1,
                            add_self_arc)

            save_tensor(eulaij, store_stats_folderpath, 'eulaij_train', layer + 1)

            if ckpt_name is not None:
                architecture_dict[layer] = cgmm_layer.build_dict()
                # todo it is not so elegant..
                architecture_dict[layer]['use_statistics'] = self.use_statistics  # it may be useful to remember

            architecture.append(cgmm_layer)

        if ckpt_name is not None:
            torch.save(architecture_dict, os.path.join(exp_name, checkpoint_folder, ckpt_name))

        return architecture, architecture_dict

    def incremental_inference(self, layers, node_dataset, edges, adjacency_lists, sizes, max_arieties,
                              infer_with_posterior, unibigram, add_self_arc,
                              exp_name, tr_val_test, architecture=None, prev_stats_path=None,
                              ckpt_name=None, aggregation_type='mean', all_combinations=False):
        """
        Performs inference throughout the architecture. Assumes C = C2 for the moment.
        IF prev_stats_folder IS NOT NONE, INFER JUST THE LAST LAYER (MAYBE RUN SOME CHECKS)
        """
        unigrams_folderpath = 'unigrams_' + tr_val_test
        unigram_filename = 'unigrams'
        inferred_states_folderpath = 'states_' + tr_val_test
        inferred_states_filename = 'states'
        stats_folderpath = 'stats_' + tr_val_test
        stats_filename = 'stats'
        store_stats_folderpath = os.path.join(exp_name, stats_folderpath)
        load_stats_folderpath = store_stats_folderpath
        inferred_states_folderpath = os.path.join(exp_name, inferred_states_folderpath)
        unigrams_folderpath = os.path.join(exp_name, unigrams_folderpath)

        if not os.path.exists(unigrams_folderpath):
            os.makedirs(unigrams_folderpath)
        if not os.path.exists(store_stats_folderpath):
            os.makedirs(store_stats_folderpath)
        if not os.path.exists(inferred_states_folderpath):
            os.makedirs(inferred_states_folderpath)

        if architecture is None:
            ckpt = torch.load(os.path.join(exp_name, checkpoint_folder, ckpt_name))
            architecture = CGMM_Layer.build_architecture(ckpt)
        else:
            if prev_stats_path is not None:
                load_stats_folderpath = os.path.join(prev_stats_path, stats_folderpath)
            else:
                load_stats_folderpath = os.path.join(exp_name, stats_folderpath)

        print('Inference for', layers,  'layers on', tr_val_test)

        likelihood_by_layer = []

        start_layer = 0 if prev_stats_path is None else len(architecture)-1  # else I need to infer only the last layer

        for layer in range(start_layer, layers):
            cgmm_layer = architecture[layer]

            layer_wise_statistics = [(layer - x + 1) for x in self.use_statistics if (layer - x) >= 0]

            if cgmm_layer.is_layer_0:
                stats_nodes = None
            else:
                stats_to_load = [os.path.join(load_stats_folderpath, stats_filename + '_node_' + str(level) + '.npy')
                                 for level in layer_wise_statistics]
                stats_nodes = load_to_ZipDataset(stats_to_load)

            train_set = LabelAndStatsDataset(node_dataset, stats_nodes)
            train_set = DataLoader(train_set, batch_size=4000, shuffle=False, num_workers=2)

            # Accumulate the likelihood of all batches
            l = 0.

            inferred_states = None
            posteriors = None

            for input_batch in train_set:

                posterior_batch, batch_likelihood = cgmm_layer(*input_batch)

                l += batch_likelihood

                # Always used for statistics
                prediction_batch = torch.argmax(posterior_batch, dim=1).detach().numpy()

                if inferred_states is None:
                    inferred_states = prediction_batch
                else:
                    inferred_states = np.append(inferred_states, prediction_batch)

                if posteriors is None:
                    posteriors = posterior_batch.detach().numpy()
                else:
                    posteriors = np.concatenate((posteriors, posterior_batch.detach().numpy()), axis=0)

            stats_nodes, stats_edges = save_statistics(self,
                                         adjacency_lists, posteriors, edges, sizes, max_arieties,
                                         store_stats_folderpath, stats_filename, layer+1, add_self_arc)

            if not all_combinations:
                fingerprints = self.aggregate_states(adjacency_lists, sizes,
                                                 posteriors if infer_with_posterior else inferred_states,
                                                 stats_nodes=stats_nodes if unibigram else None,
                                                 aggregate_posteriors=infer_with_posterior, unibigram=unibigram,
                                                 aggregation_type=aggregation_type)

                save_tensor(fingerprints, unigrams_folderpath, unigram_filename, layer + 1)

            else:
                for infer in [True, False]:
                    for unib in [True, False]:
                        fingerprints = self.aggregate_states(adjacency_lists, sizes,
                                                             posteriors if infer else inferred_states,
                                                             stats_nodes=stats_nodes if unib else None,
                                                             aggregate_posteriors=infer,
                                                             unibigram=unib,
                                                             aggregation_type=aggregation_type)

                        comb_unig_filename = 'unigrams_' + str(infer) + '_' + str(unib)
                        save_tensor(fingerprints, unigrams_folderpath, comb_unig_filename, layer + 1)

            inferred_states = posteriors

            save_tensor(inferred_states, inferred_states_folderpath, inferred_states_filename, layer+1)

            likelihood_by_layer.append(l)

        return likelihood_by_layer

    '''
    OLD, did not use complete neighbor posterior
    def compute_statistics(self, adjacency_lists, inferred_states, edges, add_self_arc=False):
        # Compute statistics
        statistics = np.zeros((len(adjacency_lists), self.A + 2, self.CN), dtype=np.int64)

        arc_id = 0
        for u in range(0, len(adjacency_lists)):
            incident_nodes = adjacency_lists[u]

            if add_self_arc:
                statistics[u, self.A-1, inferred_states[u]] += 1

            for u2 in incident_nodes:
                node_state = inferred_states[u2]
                statistics[u, edges[arc_id], node_state] += 1
                
                arc_id += 1

        return statistics, None
    '''

    def compute_statistics(self, adjacency_lists, inferred_posterior, edges, sizes, max_arieties, add_self_arc=False):
        # Compute statistics
        statistics = np.full((len(adjacency_lists), self.A + 2, self.CN2), 1e-8, dtype=np.float32)

        arc_id = 0
        current_graph = 0  # idx of the current graph
        graph_u = 0  # idx of u in a specific graph
        graph_ariety = max_arieties[current_graph]  # avoid being
        for u in range(0, len(adjacency_lists)):
            graph_u += 1

            incident_edges = adjacency_lists[u]

            # Degree of vertex defined as the number of incident edges
            degree_u = len(incident_edges)

            if add_self_arc:
                statistics[u, self.A, :] += inferred_posterior[u]

            # USE self.A+1 as special edge for bottom states (all in self.C2-1)
            statistics[u, self.A+1, self.CN2-1] += 1 - degree_u/graph_ariety


            for u2 in incident_edges:
                statistics[u, edges[arc_id], :-1] += inferred_posterior[u2]

            if sizes[current_graph] == graph_u:
                graph_u = 0
                current_graph += 1
                if current_graph < len(max_arieties):
                    graph_ariety = max_arieties[current_graph]

        # print(statistics[:, self.A+1, -1])

        return statistics, None

    def aggregate_states(self, adj_lists, sizes, node_states, stats_nodes=None, aggregate_posteriors=False,
                         unibigram=False, aggregation_type='mean'):
        """
        Aggregates the states into a frequency vector.
        """
        freq_aggregated = []

        curr_node_size = 0
        for size in sizes:

            assert size != 0

            freq_node_unigram = np.zeros(self.CN)
            if unibigram:
                freq_node_bigram = np.zeros(self.CN * self.CN)
            if not aggregate_posteriors:
                # ---------- infer with argmax ---------- #
                np.add.at(freq_node_unigram, node_states[curr_node_size:(curr_node_size + size)], 1)
                if aggregation_type == 'mean':
                    freq_node_unigram = np.divide(freq_node_unigram, size)  # Normalize

                if unibigram:
                    assert stats_nodes is not None
                    # Update the feature matrix for each structure
                    for u in range(curr_node_size, curr_node_size + size):
                        freq_node_bigram[node_states[u] * self.CN: node_states[u] * self.CN + self.CN] += \
                            np.sum(stats_nodes[u, :, :], axis=0)

                    if aggregation_type == 'mean':
                        freq_node_bigram = np.divide(freq_node_bigram, size)  # Normalize

            else:
                # ---------- infer with posterior ---------- #

                freq_node_unigram = np.sum(node_states[curr_node_size:(curr_node_size + size)], axis=0)
                if aggregation_type == 'mean':
                    freq_node_unigram = np.divide(freq_node_unigram, size)  # Normalize

                if unibigram:
                    assert stats_nodes is not None
                    # Update the feature matrix for each structure
                    for u in range(curr_node_size, curr_node_size + size):

                        for i in range(self.CN):
                            q_u_i = node_states[u, i]
                            for neighbour in adj_lists[u]:
                                    freq_node_bigram[i * self.CN: i * self.CN + self.CN] +=\
                                        node_states[neighbour, :]*q_u_i

                    if aggregation_type == 'mean':
                        freq_node_bigram = np.divide(freq_node_bigram, size)  # Normalize

            if unibigram:
                freq_aggregated.append(np.concatenate((freq_node_unigram, freq_node_bigram)))
            else:
                freq_aggregated.append(freq_node_unigram)

            curr_node_size += size

        freq_aggregated = np.array(freq_aggregated)
        return freq_aggregated
