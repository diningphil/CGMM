import shutil
from torch.utils.data import DataLoader

from models.LogisticRegression import LogisticRegressionModel
from models.SimpleMLPClassifier import SimpleMLPClassifier
from models.Utilities import *
from experiments.Experiment import Experiment
from utils.utils import inductive_split_dataset


class GraphTrivialExperiment(Experiment):
    """
    This experiment wants to assess performances
    """

    def __init__(self, model_config, exp_path):
        super(GraphTrivialExperiment, self).__init__(model_config, exp_path)

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


    def _helper(self, train_data, unseen_data, is_valid, other=None):

        # ------------- EXTRACT CURRENT CONFIGURATION -------------- #

        classifier = self.model_config['classifier']
        hidden_units = self.model_config['hidden_units']
        learning_rate = self.model_config['learning_rate']
        l2 = self.model_config['l2']
        early_stopping = self.model_config['early_stopping']  # the number of epochs when to start early stopping, 0 ow
        l_batch_size = self.model_config['l_batch_size']
        training_epochs = self.model_config['training_epochs']
        plot = self.model_config['plot']

        # ------------------------------------------------ #

        # ------------- DATASET PREPARATION -------------- #

        save_path = os.path.join(self.exp_path, 'exp')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not is_valid:
            train, valid, _ = inductive_split_dataset(train_data, 0.9, 0.1, 0., shuffle=True)

            X_train, edges_train, Y_train, adjacency_lists_train, sizes_train, _ = unravel(train)
            X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, _ = unravel(valid)
            X_test, edges_test, Y_test, adjacency_lists_test, sizes_test, _ = unravel(unseen_data)

            np.savez(os.path.join(save_path, 'train_targets.npz'), Y_train)
            np.savez(os.path.join(save_path, 'valid_targets.npz'), Y_valid)
            np.savez(os.path.join(save_path, 'test_targets.npz'), Y_test)
        else:
            X_train, edges_train, Y_train, adjacency_lists_train, sizes_train, _ = unravel(train_data)
            X_valid, edges_valid, Y_valid, adjacency_lists_valid, sizes_valid, _ = unravel(unseen_data)

            np.savez(os.path.join(save_path, 'train_targets.npz'), Y_train)
            np.savez(os.path.join(save_path, 'valid_targets.npz'), Y_valid)

        if is_valid:
            feature_size = int(np.max(np.concatenate((X_train, X_valid)))) + 1
        else:
            feature_size = int(np.max(np.concatenate((X_train, X_valid, X_test)))) + 1

        trivial_fing_train = np.zeros((len(sizes_train), feature_size))
        trivial_fing_valid = np.zeros((len(sizes_valid), feature_size))


        mean = False

        X_train = X_train.astype(np.int)
        X_valid = X_valid.astype(np.int)
        if not is_valid:
            X_test = X_test.astype(np.int)

        curr_node_size = 0
        g = 0
        for size in sizes_train:
            np.add.at(trivial_fing_train[g, :], X_train[curr_node_size:(curr_node_size + size)], 1)
            curr_node_size += size

            if mean:
                trivial_fing_train[g, :] = np.divide(trivial_fing_train[g, :], size)  # Normalize
            g += 1

        curr_node_size = 0
        g = 0
        for size in sizes_valid:
            np.add.at(trivial_fing_valid[g, :], X_valid[curr_node_size:(curr_node_size + size)], 1)
            curr_node_size += size

            if mean:
                trivial_fing_valid[g, :] = np.divide(trivial_fing_valid[g, :], size)  # Normalize
            g += 1
        if not is_valid:
            trivial_fing_test = np.zeros((len(sizes_test), feature_size))
            g = 0
            curr_node_size = 0
            for size in sizes_test:
                np.add.at(trivial_fing_test[g, :], X_test[curr_node_size:(curr_node_size + size)], 1)
                curr_node_size += size

                if mean:
                    trivial_fing_test[g, :] = np.divide(trivial_fing_test[g, :], size)  # Normalize
                g += 1

            node_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(trivial_fing_test))
            test_set = DataLoader(node_dataset_test, batch_size=l_batch_size)


        node_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(trivial_fing_train))
        node_dataset_valid = torch.utils.data.TensorDataset(torch.from_numpy(trivial_fing_valid))
        train_set = DataLoader(node_dataset_train, batch_size=l_batch_size)
        valid_set = DataLoader(node_dataset_valid, batch_size=l_batch_size)

        # ------------------------------------------------ #

        Y_train = np.load(os.path.join(save_path, 'train_targets.npz'))['arr_0']
        Y_valid = np.load(os.path.join(save_path, 'valid_targets.npz'))['arr_0']

        no_classes = len(np.unique(np.concatenate((Y_train, Y_valid))))
        target_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(Y_train))
        target_dataset_valid = torch.utils.data.TensorDataset(torch.from_numpy(Y_valid))
        train_targets = DataLoader(target_dataset_train, batch_size=l_batch_size)
        valid_targets = DataLoader(target_dataset_valid, batch_size=l_batch_size)

        if not is_valid:
            Y_test = np.load(os.path.join(save_path, 'test_targets.npz'))['arr_0']
            target_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(Y_test))
            test_targets = DataLoader(target_dataset_test, batch_size=l_batch_size)
        else:
            test_targets = None
            test_set = None

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

        if not is_valid:
            te_acc, te_loss, _, = clf.compute_accuracy(test_set, test_targets)
            print('TR VL TE:', tr_acc, vl_acc, te_acc)
            print('VL loss TE loss:', vl_l_mean, te_loss)
        else:
            print('TR VL:', tr_acc, vl_acc)

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
