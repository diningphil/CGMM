import shutil
from torch.utils.data import DataLoader
from models.CGMM.CGMM import CGMM
from models.Utilities import *
from experiments.Experiment import Experiment


class GraphUnigramsExperiment(Experiment):
    """
    This experiment is used to produce fingerprints. Being the model separate from the classifier,
    we can try to build the fingerprints in advance (without pooling) and then perform a grid search
    over the classifier hyperparams.
    """

    def __init__(self, model_config, exp_path):
        super(GraphUnigramsExperiment, self).__init__(model_config, exp_path)

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

        # ------------- EXTRACT CURRENT CONFIGURATION -------------- #

        threshold = self.model_config['threshold']
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
        aggregation_type = self.model_config['aggregation']

        '''
        unibigram = self.model_config['unibigram']
        infer_with_posterior = self.model_config['infer_with_posterior']
        
        classifier = self.model_config['classifier']
        hidden_units = self.model_config['hidden_units']
        learning_rate = self.model_config['learning_rate']
        l2 = self.model_config['l2']
        l_batch_size = self.model_config['l_batch_size']
        training_epochs = self.model_config['training_epochs']
        '''
        # ------------------------------------------------ #

        # ------------- DATASET PREPARATION -------------- #

        X_train, edges_train, Y_train, adjacency_lists_train, sizes_train, max_arieties_train = unravel(train_data)
        X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, max_arieties_valid = unravel(valid_or_test_data)

        no_classes = len(np.unique(np.concatenate((Y_train, Y_valid))))

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

        # ------------------------------------------------ #

        if model_class == 'CGMM':
            model = CGMM(K, A, CN, use_statistics, node_type)
            model_types = ['node']

        save_path = os.path.join(self.exp_path, 'exp')

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

            # WE ARE PUTTING FALSE BECAUSE WE PASS ALL_COMBINATIONS TO INCREMENTAL INFERENCE!
            # IT GENERATES ALL POSSIBLE COMBINATIONS OF UNIBIGRAM AND INFER WITH POSTERIOR (4 in total)

            likelihood_tr = model.incremental_inference(layers, nodes_dataset_train, edges_dataset_train,
                                                        adjacency_lists_train, sizes_train,
                                                        max_arieties_train, False, False, add_self_arc,
                                                        save_path, tr_val_test='train',
                                                        architecture=arch, prev_stats_path=save_path,
                                                        aggregation_type=aggregation_type, all_combinations=True)

            likelihood_vl = model.incremental_inference(layers, nodes_dataset_valid, edges_dataset_valid,
                                                        adjacency_lists_valid, sizes_valid,
                                                        max_arieties_valid, False, False, add_self_arc,
                                                        save_path, tr_val_test='valid',
                                                        architecture=arch, prev_stats_path=save_path,
                                                        aggregation_type=aggregation_type, all_combinations=True)

            # IF use statistics is [1] delete the previous winning statistics also!
            if layers > 1 and use_statistics == [1]:
                for model_type in model_types:
                    last_tr_stats_path = os.path.join(
                        'stats_train/stats_' + model_type + '_' + str(layers - 1) + '.npy')
                    last_te_stats_path = os.path.join(
                        'stats_valid/stats_' + model_type + '_' + str(layers - 1) + '.npy')
                    for path in [last_tr_stats_path, last_te_stats_path]:
                        delete_stats = os.path.join(save_path, path)
                        os.remove(delete_stats)

            run_arch = arch, arch_dict
            layers += 1

        # --------------------------------------------------------------------------------- #

        # DELETE STATISTICS OF EACH RUN, KEEP UNIGRAMS!
        # Move winning statistics to the winner folder
        last_tr_stats_path = os.path.join('stats_train')
        last_te_stats_path = os.path.join('stats_valid')
        for path in [last_tr_stats_path, last_te_stats_path]:
            src = os.path.join(self.exp_path, 'exp', path)
            shutil.rmtree(src)
        last_tr_stats_path = os.path.join('states_train')
        last_te_stats_path = os.path.join('states_valid')
        for path in [last_tr_stats_path, last_te_stats_path]:
            src = os.path.join(self.exp_path, 'exp', path)
            shutil.rmtree(src)


        np.savez(os.path.join(save_path, 'train_targets.npz'), Y_train)
        np.savez(os.path.join(save_path, 'valid_targets.npz'), Y_valid)


        return 0., 0.
