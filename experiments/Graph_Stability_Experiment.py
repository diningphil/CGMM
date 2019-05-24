import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.CGMM.CGMM import CGMM
from models.LogisticRegression import LogisticRegressionModel
from models.SimpleMLPClassifier import SimpleMLPClassifier
from models.Utilities import *
from experiments.Experiment import Experiment
from utils.utils import inductive_split_dataset


class StabilityExperiment(Experiment):
    """
    This experiment wants to study how much fingerprints may vary and how this affects successive layers
    """

    def __init__(self, model_config, exp_path):
        super(StabilityExperiment, self).__init__(model_config, exp_path)

    def heatmap_plot(self, x, **kwargs):
        # x is a Series Object
        sns.heatmap(x.values[0])

    def run_valid(self, train_data, valid_data, other=None):
        """
        This function returns the training and validation accuracy. DO WHATEVER YOU WANT WITH VL SET,
        BECAUSE YOU WILL MAKE PERFORMANCE ASSESSMENT ON A TEST SET
        :return: (training accuracy, validation accuracy)
        """
        return self._helper(train_data, valid_data, other=None)  # _helper in this case does not perform early stopping on VL

    def run_test(self, train_data, test_data, other=None):
        """
        This function returns the training and test accuracy
        :return: (training accuracy, test accuracy)
        """
        return self._helper(train_data, test_data, other=None)

    def _helper(self, train_data, valid_or_test_data, other=None):

        # TODO collect information about likelihood per layer, compare fingerprints by plotting their distributions
        # one alongside the others (sort before), compute JSD divergence score on those distributions

        # ------------- EXTRACT CURRENT CONFIGURATION -------------- #

        threshold = self.model_config['threshold']
        runs  = self.model_config['runs']
        max_epochs = self.model_config['max_epochs']
        max_layers = self.model_config['max_layers']

        model_class = self.model_config['model_class']

        K  = self.model_config['K']
        A  = self.model_config['A']
        CN = self.model_config['CN']
        CA = self.model_config['CA']
        node_type = self.model_config['node_type']
        edge_type = self.model_config['edge_type']
        use_statistics = self.model_config['use_statistics']
        add_self_arc = self.model_config['add_self_arc']

        unibigram = self.model_config['unibigram']
        aggregation_type = self.model_config['aggregation']
        infer_with_posterior = self.model_config['infer_with_posterior']

        classifier = self.model_config['classifier']
        hidden_units = self.model_config['hidden_units']
        learning_rate = self.model_config['learning_rate']
        l2 = self.model_config['l2']
        l_batch_size = self.model_config['l_batch_size']
        training_epochs = self.model_config['training_epochs']
        early_stopping = self.model_config['early_stopping']
        # ------------------------------------------------ #

        # ------------- DATASET PREPARATION -------------- #

        train, valid, _ = inductive_split_dataset(train_data, 0.9, 0.1, 0., shuffle=True)

        X_train, edges_train, Y_train, adjacency_lists_train, sizes_train, max_arieties_train = unravel(train)
        X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, max_arieties_valid = unravel(valid)
        X_test, edges_test, Y_test, adjacency_lists_test, sizes_test, max_arieties_test = unravel(valid_or_test_data)

        no_classes = len(np.unique(np.concatenate((Y_train, Y_valid, Y_test))))

        if node_type == 'continuous':
            X_train = X_train.astype(np.float64)
            X_valid = X_valid.astype(np.float64)

        if edge_type == 'continuous':
            edges_valid = edges_valid.astype(np.float64)
            edges_train = edges_train.astype(np.float64)


        nodes_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train))  # It will return a tuple
        edges_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(edges_train))  # It will return a tuple
        target_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(Y_train))  # It will return a tuple

        nodes_dataset_valid = torch.utils.data.TensorDataset(torch.from_numpy(X_valid))  # It will return a tuple
        edges_dataset_valid = torch.utils.data.TensorDataset(torch.from_numpy(edges_valid))  # It will return a tuple
        target_dataset_valid = torch.utils.data.TensorDataset(torch.from_numpy(Y_valid))  # It will return a tuple

        nodes_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test))  # It will return a tuple
        edges_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(edges_test))  # It will return a tuple
        target_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(Y_test))  # It will return a tuple


        # ------------------------------------------------ #

        if model_class == 'CGMM':
            model = CGMM(K, A, CN, use_statistics, node_type)
            model_types = ['node']

        tr_matrix = np.zeros((runs, max_layers))
        vl_matrix = np.zeros((runs, max_layers))
        te_matrix = np.zeros((runs, max_layers))

        for run in range(1, runs+1):

            save_path = os.path.join(self.exp_path, 'run_' + str(run))

            run_arch = None

            layers = 1
            while layers <= max_layers:  # very naive early stopping technique

                # Resume from the best architecture (if any) and add a new level
                arch, arch_dict = model.incremental_training(adjacency_lists_train,
                                                             nodes_dataset_train, edges_dataset_train, sizes_train,
                                                             max_arieties_train, layers, add_self_arc,
                                                             save_path, threshold=threshold, max_epochs=max_epochs,
                                                             prev_architecture=run_arch,
                                                             prev_stats_path=save_path,
                                                             node_type=node_type, edge_type=edge_type)

                likelihood_tr = model.incremental_inference(layers, nodes_dataset_train, edges_dataset_train,
                                                            adjacency_lists_train, sizes_train,
                                                            max_arieties_train, infer_with_posterior, unibigram, add_self_arc,
                                                            save_path, tr_val_test='train',
                                                            architecture=arch, prev_stats_path=save_path,
                                                            aggregation_type=aggregation_type)

                likelihood_vl = model.incremental_inference(layers, nodes_dataset_valid, edges_dataset_valid,
                                                            adjacency_lists_valid, sizes_valid,
                                                            max_arieties_valid, infer_with_posterior, unibigram, add_self_arc,
                                                            save_path, tr_val_test='valid',
                                                            architecture=arch, prev_stats_path=save_path,
                                                            aggregation_type=aggregation_type)

                likelihood_te = model.incremental_inference(layers, nodes_dataset_test, edges_dataset_test,
                                                            adjacency_lists_test, sizes_test,
                                                            max_arieties_test, infer_with_posterior, unibigram, add_self_arc,
                                                            save_path, tr_val_test='test',
                                                            architecture=arch, prev_stats_path=save_path,
                                                            aggregation_type=aggregation_type)


                filepaths_tr = [os.path.join(save_path, 'unigrams_train', 'unigrams_' + str(layer) + '.npy')
                                for layer in range(1, layers)]
                filepaths_tr.append(os.path.join(save_path, 'unigrams_train', 'unigrams_' + str(layers) + '.npy'))

                fingerprints_train_dataset = load_to_ZipDataset(filepaths_tr)
                fingerprints_tr = concat_graph_fingerprints(fingerprints_train_dataset)

                filepaths_valid = [os.path.join(save_path, 'unigrams_valid', 'unigrams_' + str(layer) + '.npy')
                                   for layer in range(1, layers)]
                filepaths_valid.append(os.path.join(save_path, 'unigrams_valid', 'unigrams_' + str(layers) + '.npy'))

                fingerprints_valid_dataset = load_to_ZipDataset(filepaths_valid)
                fingerprints_valid = concat_graph_fingerprints(fingerprints_valid_dataset)

                filepaths_test = [os.path.join(save_path, 'unigrams_test', 'unigrams_' + str(layer) + '.npy')
                                  for layer in range(1, layers)]

                filepaths_test.append(os.path.join(save_path, 'unigrams_test', 'unigrams_' + str(layers) + '.npy'))

                fingerprints_test_dataset = load_to_ZipDataset(filepaths_test)
                fingerprints_test = concat_graph_fingerprints(fingerprints_test_dataset)

                train_set = DataLoader(fingerprints_tr, batch_size=l_batch_size)
                train_targets = DataLoader(target_dataset_train, batch_size=l_batch_size)
                valid_set = DataLoader(fingerprints_valid, batch_size=l_batch_size)
                valid_targets = DataLoader(target_dataset_valid, batch_size=l_batch_size)
                test_set = DataLoader(fingerprints_test, batch_size=l_batch_size)
                test_targets = DataLoader(target_dataset_test, batch_size=l_batch_size)

                feature_size = fingerprints_tr.shape[1]
                if classifier == 'logistic':
                    clf = LogisticRegressionModel(feature_size, no_classes)
                    tr_acc, early_stop_epochs = clf.train(train_set, train_targets,
                                                          learning_rate=learning_rate, l2=l2,
                                                          max_epochs=training_epochs,
                                                          vl_loader=valid_set, vl_target_loader=valid_targets,
                                                          te_loader=test_set, te_target_loader=test_targets,
                                                          early_stopping=early_stopping, plot=False)
                    vl_acc, vl_l_mean, vl_l_std = clf.compute_accuracy(valid_set, valid_targets)
                    te_acc, te_l_mean, te_l_std = clf.compute_accuracy(test_set, test_targets)

                elif classifier == 'mlp':
                    clf = SimpleMLPClassifier(feature_size, hidden_units, no_classes)
                    tr_acc, early_stop_epochs = clf.train(train_set, train_targets,
                                                          learning_rate=learning_rate, l2=l2,
                                                          max_epochs=training_epochs,
                                                          vl_loader=valid_set, vl_target_loader=valid_targets,
                                                          te_loader=test_set, te_target_loader=test_targets,
                                                          early_stopping=early_stopping, plot=False)
                    vl_acc, vl_l_mean, vl_l_std = clf.compute_accuracy(valid_set, valid_targets)
                    te_acc, te_l_mean, te_l_std = clf.compute_accuracy(test_set, test_targets)

                print('Layer', layers, 'TR VL TE:', tr_acc, vl_acc, te_acc)
                tr_matrix[run-1, layers-1] = tr_acc
                vl_matrix[run-1, layers-1] = vl_acc
                te_matrix[run-1, layers-1] = te_acc

                # IF use statistics is [1] delete the previous winning statistics also!
                if layers > 1 and use_statistics == [1]:
                    for model_type in model_types:
                        last_tr_stats_path = os.path.join(
                            'stats_train/stats_' + model_type + '_' + str(layers - 1) + '.npy')
                        last_vl_stats_path = os.path.join(
                            'stats_valid/stats_' + model_type + '_' + str(layers - 1) + '.npy')
                        last_te_stats_path = os.path.join(
                            'stats_test/stats_' + model_type + '_' + str(layers - 1) + '.npy')
                        for path in [last_tr_stats_path, last_vl_stats_path, last_te_stats_path]:
                            delete_stats = os.path.join(save_path, path)
                            os.remove(delete_stats)

                run_arch = arch, arch_dict
                layers += 1

            # --------------------------------------------------------------------------------- #

            # DELETE STATISTICS OF EACH RUN, KEEP UNIGRAMS!
            # Move winning statistics to the winner folder
            last_tr_stats_path = os.path.join('stats_train')
            last_vl_stats_path = os.path.join('stats_valid')
            last_te_stats_path = os.path.join('stats_test')
            for path in [last_tr_stats_path, last_vl_stats_path, last_te_stats_path]:
                src = os.path.join(self.exp_path, 'run_' + str(run), path)
                shutil.rmtree(src)
            last_tr_states_path = os.path.join('states_train')
            last_vl_states_path = os.path.join('states_valid')
            last_te_states_path = os.path.join('states_test')
            for path in [last_tr_states_path, last_vl_states_path, last_te_states_path]:
                src = os.path.join(self.exp_path, 'run_' + str(run), path)
                shutil.rmtree(src)

        # Plot fingerprints for each run and layer (on a grid)
        heatgrid_df = pd.DataFrame()
        # Plot differences with the fingerprint before (ordered so that it approximates a mapping)
        diff_df = pd.DataFrame()
        accuracy_df = pd.DataFrame()

        for run in range(1, runs + 1):
            unigrams_path = 'run_' + str(run)
            jsd_matrix = None

            for layer in range(1, max_layers + 1):
                filepath = os.path.join(self.exp_path, unigrams_path, 'unigrams_train', 'unigrams_' + str(layer) + '.npy')
                unigrams = np.array(np.load(filepath, mmap_mode='r'))
                unigrams.sort(axis=1)

                accuracy_df = accuracy_df.append(
                    {'acc': tr_matrix[run-1, layer-1], 'run': run, 'layer': layer, 'tr_val_test': 'train'},
                    ignore_index=True)
                accuracy_df = accuracy_df.append(
                    {'acc': vl_matrix[run-1, layer-1], 'run': run, 'layer': layer, 'tr_val_test': 'valid'},
                    ignore_index=True)
                accuracy_df = accuracy_df.append(
                    {'acc': te_matrix[run-1, layer-1], 'run': run, 'layer': layer, 'tr_val_test': 'test'},
                    ignore_index=True)

                heatgrid_df = heatgrid_df.append({'unigrams': unigrams, 'run': run, 'layer': layer}, ignore_index=True)

                '''
                if layer == 1:
                    old_fingerprints = unigrams
                    jsd_matrix = np.zeros((unigrams.shape[0], 1))
                    diff_matrix = unigrams - old_fingerprints
                else:
                    jsd_divergence, jsd_distance = JSD(old_fingerprints, unigrams)
                    diff_matrix = unigrams - old_fingerprints
                    old_fingerprints = unigrams
                    jsd_divergence = np.reshape(jsd_divergence, (jsd_divergence.shape[0], 1))
                    jsd_matrix = np.concatenate((jsd_matrix, jsd_divergence), axis=1)

                diff_df = diff_df.append({'differences': diff_matrix, 'run': run, 'layer': layer}, ignore_index=True)
                '''

        np.save(os.path.join(self.exp_path, 'tr_matrix.npy'), tr_matrix)
        np.save(os.path.join(self.exp_path, 'vl_matrix.npy'), vl_matrix)
        np.save(os.path.join(self.exp_path, 'te_matrix.npy'), te_matrix)

        pointplot_title = 'CN: ' + str(CN) + ' CA: ' + str(CA) + ' Posterior: ' + str(infer_with_posterior) + \
                          ' stats: ' + str(use_statistics) + ' maxepochs: ' + str(max_epochs)

        sns.pointplot(x="layer", y="acc", data=accuracy_df, hue='tr_val_test', ci='sd', capsize=.2, errwidth=1)\
            .set_title(pointplot_title)
        plt.savefig(os.path.join(self.exp_path, 'accuracies.png'), dpi=600)
        plt.close()

        '''
        g = sns.FacetGrid(diff_df, row="run", col="layer")
        g.map(self.heatmap_plot, 'differences')
        g.savefig(os.path.join(self.exp_path, 'differences.png'))
        plt.close()
        
        g = sns.FacetGrid(heatgrid_df, row="run", col="layer")
        g.map(self.heatmap_plot, 'unigrams')
        g.savefig(os.path.join(self.exp_path, 'fingerprints.png'))
        plt.close()
        '''

        for run in range(1, runs+1):
            save_path = os.path.join(self.exp_path, 'run_' + str(run))

            last_tr_stats_path = os.path.join('unigrams_train')
            last_te_stats_path = os.path.join('unigrams_valid')
            for path in [last_tr_stats_path, last_te_stats_path]:
                src = os.path.join(self.exp_path, 'run_' + str(run), path)
                shutil.rmtree(src)

        return 0., 0.
