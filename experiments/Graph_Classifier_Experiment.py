from torch.utils.data import DataLoader
from models.LogisticRegression import LogisticRegressionModel
from models.SimpleMLPClassifier import SimpleMLPClassifier
from models.Utilities import *
from experiments.Experiment import Experiment


class GraphClassifierExperiment(Experiment):
    """
    This experiment wants to assess performances
    """

    def __init__(self, model_config, exp_path):
        super(GraphClassifierExperiment, self).__init__(model_config, exp_path)

    def run_valid(self, train_data, valid_data, other=None):
        """
        This function returns the training and validation accuracy. DO WHATEVER YOU WANT WITH VL SET,
        BECAUSE YOU WILL MAKE PERFORMANCE ASSESSMENT ON A TEST SET
        :return: (training accuracy, validation accuracy)
        """
        return self._helper(train_data, valid_data, other=None)

    def run_test(self, train_data, test_data, other=None):
        """
        This function returns the training and test accuracy
        :return: (training accuracy, test accuracy)
        """
        raise NotImplementedError('This experiment is not designed to work with test data!')
        return None
    def __prepare_datasets(self, Y):
        target_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Y))  # It will return a tuple
        return target_dataset

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


        return threshold, max_epochs, max_layers, model_class, K, A, CN, CA, node_type, edge_type, \
               use_statistics, add_self_arc, unibigram, aggregation_type, infer_with_posterior, classifier,\
               hidden_units, learning_rate, l2, early_stopping, l_batch_size, training_epochs, plot

    def _helper(self, train_data, unseen_data,  other=None):

        # ------------- EXTRACT CURRENT CONFIGURATION -------------- #

        threshold, max_epochs, max_layers, model_class, K, A, CN, CA, node_type, edge_type, \
        use_statistics, add_self_arc, unibigram, aggregation_type, infer_with_posterior, classifier, \
        hidden_units, learning_rate, l2, early_stopping, l_batch_size, training_epochs, plot = self.__extract_config()

        # ------------------------------------------------ #

        save_path = os.path.join(self.exp_path, 'exp')

        Y_train = np.load(os.path.join(save_path, 'train_targets.npz'))['arr_0']
        Y_valid = np.load(os.path.join(save_path, 'valid_targets.npz'))['arr_0']

        no_classes = len(np.unique(np.concatenate((Y_train, Y_valid))))

        target_dataset_train = self.__prepare_datasets(Y_train)
        target_dataset_valid = self.__prepare_datasets(Y_valid)

        layers = max_layers
        str(infer_with_posterior) + '_' + str(unibigram) + '_'

        filepaths_tr = [os.path.join(save_path, 'unigrams_train', 'unigrams_' +
                                     str(infer_with_posterior) + '_' + str(unibigram) + '_' + str(layer) + '.npy')
                        for layer in range(1, layers)]
        train_set, feature_size = self.__build_dataset_of_fingerprints(filepaths_tr, l_batch_size)
        train_targets = DataLoader(target_dataset_train, batch_size=l_batch_size)

        filepaths_valid = [os.path.join(save_path, 'unigrams_valid', 'unigrams_' +
                                        str(infer_with_posterior) + '_' + str(unibigram) + '_' + str(layer) + '.npy')
                           for layer in range(1, layers)]
        valid_set, _ = self.__build_dataset_of_fingerprints(filepaths_valid, l_batch_size)
        valid_targets = DataLoader(target_dataset_valid, batch_size=l_batch_size)

        if classifier == 'logistic':
            clf = LogisticRegressionModel(feature_size, no_classes)
            tr_acc, early_stop_epochs = clf.train(train_set, train_targets,
                                   learning_rate=learning_rate, l2=l2, max_epochs=training_epochs,
                                   vl_loader=valid_set, vl_target_loader=valid_targets, early_stopping=early_stopping, plot=plot)
            vl_acc, vl_l_mean, vl_l_std = clf.compute_accuracy(valid_set, valid_targets)

        elif classifier == 'mlp':
            clf = SimpleMLPClassifier(feature_size, hidden_units, no_classes)
            tr_acc, early_stop_epochs = clf.train(train_set, train_targets,
                               learning_rate=learning_rate, l2=l2, max_epochs=training_epochs,
                               vl_loader=valid_set, vl_target_loader=valid_targets, early_stopping=early_stopping, plot=plot)
            vl_acc, vl_l_mean, vl_l_std = clf.compute_accuracy(valid_set, valid_targets)

        print('Layer', layers, 'TR VL:', tr_acc, vl_acc)

        return tr_acc, vl_acc
