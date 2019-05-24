import os
import json
import concurrent.futures
import pickle
from time import sleep

from log.Logger import Logger
from utils.utils import shuffle_dataset, generate_grid


class SimpleModelSelector:
    """
    Class implementing a sufficiently general framework to do model selection
    """

    def __init__(self, max_processes, train_perc):
        self.max_processes = max_processes
        self.train_perc = train_perc

        # Create the experiments folder straight away
        self._CONFIG_BASE           = 'config_'
        self._CONFIG_FILENAME       = 'config_results.json'
        self.WINNER_CONFIG_FILENAME = 'winner_config.json'

    def process_results(self, HOLDOUT_MS_FOLDER, no_configurations):

        best_vl = 0.

        for i in range(1, no_configurations+1):
            try:
                config_filename = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(i),
                                               self._CONFIG_FILENAME)

                with open(config_filename, 'r') as fp:
                    config_dict = json.load(fp)

                vl = config_dict['VL_score']

                if best_vl <= vl:
                    best_i = i
                    best_vl = vl
                    best_config = config_dict

            except Exception as e:
                print(e)

        print('Model selection winner for experiment', HOLDOUT_MS_FOLDER, 'is config ', best_i, ':')
        for k in best_config.keys():
            print('\t', k, ':', best_config[k])

        return best_config

    def model_selection(self, data, experiment_class, exp_path, model_configs, shuffle=False, debug=False, other=None):
        """
        :param train: List of examples. It will be processed by the experiment
        :param valid: List of examples. It will be processed by the experiment
        :param experiment_class: the kind of experiment used
        :param shuffle:
        :param debug:
        :return: the best performing configuration on average over the k folds. TL;DR RETURNS A MODEL, NOT AN ESTIMATE!
        """
        HOLDOUT_MS_FOLDER = os.path.join(exp_path, 'HOLDOUT_MS')

        if not os.path.exists(HOLDOUT_MS_FOLDER):
            os.makedirs(HOLDOUT_MS_FOLDER)

        config_id = 0

        if shuffle:
            shuffle_dataset(data)

        # Split in train and validation
        last_tr_idx = int(len(data) * self.train_perc)
        train, valid = data[:last_tr_idx], data[last_tr_idx:]

        with open(os.path.join(HOLDOUT_MS_FOLDER, 'train'), 'wb') as f:
            pickle.dump(train, f)
        with open(os.path.join(HOLDOUT_MS_FOLDER, 'valid'), 'wb') as f:
            pickle.dump(valid, f)

        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_processes)

        for config in generate_grid(model_configs):

            # I need to make a copy of this dictionary
            # It seems it gets shared between processes!
            config = dict(config)

            # Create a separate folder for each experiment
            exp_config_name = os.path.join(HOLDOUT_MS_FOLDER, self._CONFIG_BASE + str(config_id + 1))
            if not os.path.exists(exp_config_name):
                os.makedirs(exp_config_name)

            if not debug:
                pool.submit(self._model_selection_helper, experiment_class, config, HOLDOUT_MS_FOLDER,
                            exp_config_name, other)
            else:  # DEBUG
                self._model_selection_helper(experiment_class, config, HOLDOUT_MS_FOLDER, exp_config_name, other)

            config_id += 1

        pool.shutdown()  # wait the batch of configs to terminate

        best_config = self.process_results(HOLDOUT_MS_FOLDER, config_id)

        with open(os.path.join(HOLDOUT_MS_FOLDER, self.WINNER_CONFIG_FILENAME), 'w') as fp:
            json.dump(best_config, fp)

        return best_config

    def _model_selection_helper(self, experiment_class, config, HOLDOUT_MS_FOLDER, exp_config_name, other=None):

        with open(os.path.join(HOLDOUT_MS_FOLDER, 'train'), 'rb') as f:
            train = pickle.load(f)
        with open(os.path.join(HOLDOUT_MS_FOLDER, 'valid'), 'rb') as f:
            valid = pickle.load(f)

        # Create the experiment object which will be responsible for running a specific experiment
        experiment = experiment_class(config, exp_config_name)

        # Set up a log file for this experiment (run in a separate process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        logger.log('Configuration: ' + str(experiment.model_config))

        config_filename = os.path.join(experiment.exp_path, self._CONFIG_FILENAME)

        # ------------- PREPARE DICTIONARY TO STORE RESULTS -------------- #

        selection_dict = {
            'config': experiment.model_config,
            'TR_score': 0.,
            'VL_score': 0.,
        }

        training_score, validation_score = experiment.run_valid(train, valid, other)

        selection_dict['TR_score'] = float(training_score)
        selection_dict['VL_score'] = float(validation_score)

        logger.log('TR Accuracy: ' + str(training_score) + ' VL Accuracy: ' + str(validation_score))

        with open(config_filename, 'w') as fp:
            json.dump(selection_dict, fp)
