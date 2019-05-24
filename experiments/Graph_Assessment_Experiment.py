import shutil
from torch.utils.data import DataLoader

from models.CGMM.CGMM import CGMM
from models.LogisticRegression import LogisticRegressionModel
from models.SimpleMLPClassifier import SimpleMLPClassifier
from models.Utilities import *
from experiments.Experiment import Experiment
from utils.utils import inductive_split_dataset


class GraphAssessmentExperiment(Experiment):
    """
    This experiment wants to assess performances
    """

    def __init__(self, model_config, exp_path):
        super(GraphAssessmentExperiment, self).__init__(model_config, exp_path)

    def run_valid(self, train_data, valid_data, other=None):
        """
        This function returns the training and validation accuracy. DO WHATEVER YOU WANT WITH VL SET,
        BECAUSE YOU WILL MAKE PERFORMANCE ASSESSMENT ON A TEST SET
        :return: (training accuracy, validation accuracy)
        """
        return self._helper(train_data, valid_data, is_valid=True, other=None)

    def run_test(self, train_data, test_data, other=None):
        """
        This function returns the training and test accuracy
        :return: (training accuracy, test accuracy)
        """
        return self._helper(train_data, test_data, is_valid=False, other=None)

    def __prepare_datasets(self, X, edges, Y):
        nodes_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))  # It will return a tuple
        edges_dataset = torch.utils.data.TensorDataset(torch.from_numpy(edges))  # It will return a tuple
        target_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Y))  # It will return a tuple
        return nodes_dataset, edges_dataset, target_dataset

    def __build_dataset_of_fingerprints(self, fingerprints_files, batch_size):
        fingerprints_dataset = load_to_ZipDataset(fingerprints_files)
        fingerprints = concat_graph_fingerprints(fingerprints_dataset)
        return DataLoader(fingerprints, batch_size=batch_size), fingerprints.shape[1]

    def __extract_config(self):
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

        unibigram = self.model_config['unibigram']
        aggregation_type = self.model_config['aggregation']
        infer_with_posterior = self.model_config['infer_with_posterior']

        classifier = self.model_config['classifier']
        hidden_units = self.model_config['hidden_units']
        learning_rate = self.model_config['learning_rate']
        l2 = self.model_config['l2']
        early_stopping = self.model_config['early_stopping']  # the number of epochs when to start early stopping, 0 ow
        l_batch_size = self.model_config['l_batch_size']
        training_epochs = self.model_config['training_epochs']
        plot = self.model_config['plot']

        return threshold, max_epochs, max_layers, model_class, K, A, CN, CA, node_type, edge_type,\
               use_statistics, add_self_arc, unibigram, aggregation_type, infer_with_posterior, classifier,\
               hidden_units, learning_rate, l2, early_stopping, l_batch_size, training_epochs, plot

    def _helper(self, train_data, unseen_data, is_valid, other=None):

        # ------------- EXTRACT CURRENT CONFIGURATION -------------- #

        threshold, max_epochs, max_layers, model_class, K, A, CN, CA, node_type, edge_type, \
        use_statistics, add_self_arc, unibigram, aggregation_type, infer_with_posterior, classifier, \
        hidden_units, learning_rate, l2, early_stopping, l_batch_size, training_epochs, plot = self.__extract_config()

        # ------------------------------------------------ #

        # ------------- DATASET PREPARATION -------------- #
        save_path = os.path.join(self.exp_path, 'exp')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train = True  # Use this to reuse previously computed embeddings and to test the classifier only
        if train:
            if not is_valid:
                train, valid, _ = inductive_split_dataset(train_data, 0.9, 0.1, 0., shuffle=True)

                X_train, edges_train, Y_train, adjacency_lists_train, sizes_train, max_ariety_tr = unravel(train)
                if len(valid) != 0:
                    X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, max_ariety_vl = unravel(valid)
                    np.savez(os.path.join(save_path, 'valid_targets.npz'), Y_valid)
                else:
                    X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, max_ariety_vl = None, None, None, None, None, None

                if len(unseen_data) != 0:
                    X_test, edges_test, Y_test, adjacency_lists_test, sizes_test, max_ariety_te = unravel(unseen_data)
                    np.savez(os.path.join(save_path, 'test_targets.npz'), Y_test)
                else:
                    X_test, edges_test, Y_test, adjacency_lists_test, sizes_test, max_ariety_te = None, None, None, None, None, None

                np.savez(os.path.join(save_path, 'train_targets.npz'), Y_train)

            else:
                X_train, edges_train, Y_train, adjacency_lists_train, sizes_train, max_ariety_tr = unravel(train_data)
                if len(unseen_data) != 0:
                    X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, max_ariety_vl = unravel(unseen_data)
                    np.savez(os.path.join(save_path, 'valid_targets.npz'), Y_valid)
                else:
                    X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, max_ariety_vl = None, None, None, None, None, None
                np.savez(os.path.join(save_path, 'train_targets.npz'), Y_train)

            if node_type == 'continuous':
                X_train = X_train.astype(np.float64)
                if X_valid is not None:
                    X_valid = X_valid.astype(np.float64)

                if not is_valid:
                    if X_test is not None:
                        X_test = X_test.astype(np.float64)

            if edge_type == 'continuous':
                if edges_valid is not None:
                    edges_valid = edges_valid.astype(np.float64)
                edges_train = edges_train.astype(np.float64)

                if not is_valid and edges_test is not None:
                    edges_test = edges_test.astype(np.float64)

            nodes_dataset_train, edges_dataset_train, target_dataset_train =\
                self.__prepare_datasets(X_train, edges_train, Y_train)

            nodes_dataset_valid, edges_dataset_valid, target_dataset_valid = \
                self.__prepare_datasets(X_valid, edges_valid, Y_valid)

            if not is_valid:
                nodes_dataset_test, edges_dataset_test, target_dataset_test = \
                    self.__prepare_datasets(X_test, edges_test, Y_test)


            # ------------------------------------------------ #

            if model_class == 'CGMM':
                model = CGMM(K, A, CN, use_statistics,node_type)
                model_types = ['node']

            run_arch = None

            torch.manual_seed(0)

            layers = 1
            while layers <= max_layers:

                # Resume from the best architecture (if any) and add a new level
                arch, arch_dict = model.incremental_training(adjacency_lists_train,
                                                             nodes_dataset_train, edges_dataset_train, sizes_train,
                                                             max_ariety_tr, layers, add_self_arc,
                                                             save_path, threshold=threshold, max_epochs=max_epochs,
                                                             prev_architecture=run_arch,
                                                             prev_stats_path=save_path,
                                                             node_type=node_type, edge_type=edge_type)

                likelihood_tr = model.incremental_inference(layers, nodes_dataset_train, edges_dataset_train,
                                                            adjacency_lists_train, sizes_train, max_ariety_tr,
                                                            infer_with_posterior, unibigram, add_self_arc,
                                                            save_path, tr_val_test='train',
                                                            architecture=arch, prev_stats_path=save_path,
                                                            aggregation_type=aggregation_type)

                likelihood_vl = model.incremental_inference(layers, nodes_dataset_valid, edges_dataset_valid,
                                                            adjacency_lists_valid, sizes_valid, max_ariety_vl,
                                                            infer_with_posterior, unibigram, add_self_arc,
                                                            save_path, tr_val_test='valid',
                                                            architecture=arch, prev_stats_path=save_path,
                                                            aggregation_type=aggregation_type)
                if not is_valid:
                    likelihood_te = model.incremental_inference(layers, nodes_dataset_test, edges_dataset_test,
                                                            adjacency_lists_test, sizes_test, max_ariety_te,
                                                            infer_with_posterior, unibigram, add_self_arc,
                                                            save_path, tr_val_test='test',
                                                            architecture=arch, prev_stats_path=save_path,
                                                            aggregation_type=aggregation_type)

                # IF use statistics is [1] delete the previous winning statistics also!
                if layers > 1 and use_statistics == [1]:
                    for model_type in model_types:
                        last_tr_stats_path = os.path.join(
                            'stats_train/stats_' + model_type + '_' + str(layers - 1) + '.npy')
                        last_vl_stats_path = os.path.join(
                            'stats_valid/stats_' + model_type + '_' + str(layers - 1) + '.npy')
                        last_te_stats_path = os.path.join(
                            'stats_test/stats_' + model_type + '_' + str(layers - 1) + '.npy') if not is_valid else ''
                        for path in [last_tr_stats_path, last_vl_stats_path, last_te_stats_path]:
                            try:
                                delete_stats = os.path.join(save_path, path)
                                os.remove(delete_stats)
                            except Exception:
                                pass

                run_arch = arch, arch_dict
                layers += 1

        layers = max_layers+1

        Y_train = np.load(os.path.join(save_path, 'train_targets.npz'))['arr_0']
        Y_valid = np.load(os.path.join(save_path, 'valid_targets.npz'))['arr_0']

        no_classes = len(np.unique(np.concatenate((Y_train, Y_valid))))

        target_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(Y_train))
        target_dataset_valid = torch.utils.data.TensorDataset(torch.from_numpy(Y_valid))

        if not is_valid:
           Y_test = np.load(os.path.join(save_path, 'test_targets.npz'))['arr_0']
           target_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(Y_test))

        filepaths_tr = [os.path.join(save_path, 'unigrams_train', 'unigrams_' + str(layer) + '.npy')
                        for layer in range(1, layers)]
        train_set, feature_size = self.__build_dataset_of_fingerprints(filepaths_tr, l_batch_size)
        train_targets = DataLoader(target_dataset_train, batch_size=l_batch_size)

        filepaths_valid = [os.path.join(save_path, 'unigrams_valid', 'unigrams_' + str(layer) + '.npy')
                           for layer in range(1, layers)]
        valid_set, _ = self.__build_dataset_of_fingerprints(filepaths_valid, l_batch_size)
        valid_targets = DataLoader(target_dataset_valid, batch_size=l_batch_size)

        if not is_valid:
            filepaths_test = [os.path.join(save_path, 'unigrams_test', 'unigrams_' + str(layer) + '.npy')
                               for layer in range(1, layers)]
            test_set, _ = self.__build_dataset_of_fingerprints(filepaths_test, l_batch_size)
            test_targets = DataLoader(target_dataset_test, batch_size=l_batch_size)
        else:
            test_set = None
            test_targets = None

        print(f'Training with {classifier}')
        if classifier == 'logistic':
            clf = LogisticRegressionModel(feature_size, no_classes)
            tr_acc, early_stop_epochs = clf.train(train_set, train_targets,
                                   learning_rate=learning_rate, l2=l2, max_epochs=training_epochs,
                                   vl_loader=valid_set, vl_target_loader=valid_targets,
                                   te_loader=test_set, te_target_loader=test_targets, early_stopping=early_stopping, plot=plot)
            vl_acc, vl_l_mean, vl_l_std = clf.compute_accuracy(valid_set, valid_targets)

        elif classifier == 'mlp':
            clf = SimpleMLPClassifier(feature_size, hidden_units, no_classes)
            tr_acc, early_stop_epochs = clf.train(train_set, train_targets,
                               learning_rate=learning_rate, l2=l2, max_epochs=training_epochs,
                               vl_loader=valid_set, vl_target_loader=valid_targets,
                               te_loader=test_set, te_target_loader=test_targets, early_stopping=early_stopping, plot=plot)
            vl_acc, vl_l_mean, vl_l_std = clf.compute_accuracy(valid_set, valid_targets)

        '''
        elif classifier == 'deepmultisets':

            clf = DeepMultisets(feature_size, no_classes, {'hidden_units': hidden_units})

            tr_acc, early_stop_epochs = \
                clf.train(train_set, train_targets,
                               learning_rate=learning_rate, l2=l2, max_epochs=training_epochs,
                               vl_loader=valid_set, vl_target_loader=valid_targets,
                               te_loader=test_set, te_target_loader=test_targets, early_stopping=early_stopping, plot=plot)
            vl_acc, vl_l_mean, vl_l_std = clf.compute_accuracy(valid_set, valid_targets)
        '''
        if not is_valid:
            te_acc, te_loss, _, = clf.compute_accuracy(test_set, test_targets)
            print('Layer', layers, 'TR VL TE:', tr_acc, vl_acc, te_acc)
            print('Layer', layers, 'VL loss TE loss:', vl_l_mean, te_loss)
        else:
            print('Layer', layers, 'TR VL:', tr_acc, vl_acc)

        # --------------------------------------------------------------------------------- #

        # DELETE STATISTICS OF EACH RUN, KEEP UNIGRAMS!
        # Move winning statistics to the winner folder
        last_tr_stats_path = os.path.join('stats_train')
        last_vl_stats_path = os.path.join('stats_valid')
        last_te_stats_path = os.path.join('stats_test')
        for path in [last_tr_stats_path, last_vl_stats_path, last_te_stats_path]:
            src = os.path.join(self.exp_path, 'exp', path)
            shutil.rmtree(src, ignore_errors=True)
        last_tr_stats_path = os.path.join('states_train')
        last_vl_stats_path = os.path.join('states_valid')
        last_te_stats_path = os.path.join('states_test')
        for path in [last_tr_stats_path, last_vl_stats_path, last_te_stats_path]:
            src = os.path.join(self.exp_path, 'exp', path)
            shutil.rmtree(src, ignore_errors=True)


        return tr_acc, te_acc if not is_valid else vl_acc
