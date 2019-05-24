import os
import json
import pickle

from log.Logger import Logger
from model_selection.SimpleModelSelector import SimpleModelSelector
from utils.utils import shuffle_dataset


class HoldOut:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """

    def __init__(self, max_processes, model_selector, train_perc, exp_path, model_configs):
        self.max_processes   = max_processes
        self.model_configs   = model_configs  # Dictionary with key:list of possible values
        self.model_selector  = model_selector
        self.train_perc      = train_perc

        # Create the experiments folder straight away
        self.exp_path               = exp_path
        self._HOLDOUT_FOLDER        = os.path.join(exp_path, 'HOLDOUT_ASS')
        self._ASSESSMENT_FILENAME   = 'assessment_results.json'

    def risk_assessment(self, data, experiment_class, shuffle=True, debug=False, other=None):
        """
        :param train: List of examples. It will be processed by the experiment
        :param valid: List of examples. It will be processed by the experiment
        :param test: List of examples. It will be processed by the experiment
        :param experiment_class: the kind of experiment used
        :param shuffle:
        :param debug:
        :return: An average over the outer test folds. RETURNS AN ESTIMATE, NOT A MODEL!!!
        """
        if not os.path.exists(self._HOLDOUT_FOLDER):
            os.makedirs(self._HOLDOUT_FOLDER)
        else:
            print("Folder already present! Shutting down to prevent loss of previous experiments")
            #return

        if shuffle:
            shuffle_dataset(data)

        self._risk_assessment_helper(data, experiment_class, self._HOLDOUT_FOLDER, shuffle, debug, other)

    def _risk_assessment_helper(self, data, experiment_class, exp_path, shuffle=True, debug=False, other=None):

        # Split in train and test, pass train to model selector
        last_tr_idx = int(len(data) * self.train_perc)
        train, test = data[:last_tr_idx], data[last_tr_idx:]

        with open(os.path.join(self._HOLDOUT_FOLDER, 'train'), 'wb') as f:
            pickle.dump(train, f)
        with open(os.path.join(self._HOLDOUT_FOLDER, 'test'), 'wb') as f:
            pickle.dump(test, f)

        best_config = self.model_selector.model_selection(train, experiment_class, exp_path, self.model_configs,
                                                          shuffle, debug, other)

        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], exp_path)

        # Set up a log file for this experiment (I am in a forked process)
        logger = Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        training_score, test_score = experiment.run_test(train, test, other)
        print(training_score, test_score)

        logger.log('TR score: ' + str(training_score) + ' TS score: ' + str(test_score))

        with open(os.path.join(self._HOLDOUT_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump({'HOLDOUT_TR': training_score, 'HOLDOUT_TS': test_score}, fp)




