import os
import json
import pickle

import numpy as np
import concurrent.futures

from log import Logger
from model_selection.K_Fold import KFold
from utils.utils import shuffle_dataset, build_folds


class NestedCV:
    """
    Class implementing a sufficiently general framework to do model ASSESSMENT
    """

    def __init__(self, outer_folds, model_selector, outer_processes,
                 exp_path, model_configs):
        self.outer_folds     = outer_folds
        self.outer_processes = outer_processes
        self.model_selector = model_selector
        self.model_configs = model_configs  # Dictionary with key:list of possible values

        # Create the experiments folder straight away
        self.exp_path               = exp_path
        self.__NESTED_FOLDER        = os.path.join(exp_path, str(self.outer_folds) + '_NESTED_CV')
        self.__OUTER_FOLD_BASE      = 'OUTER_FOLD_'
        self._OUTER_RESULTS_FILENAME = 'outer_results.json'
        self._ASSESSMENT_FILENAME    = 'assessment_results.json'


    def process_results(self):

        outer_TR_scores = []
        outer_TS_scores = []
        assessment_results = {}

        for i in range(1, self.outer_folds+1):
            try:
                config_filename = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(i),
                                               self._OUTER_RESULTS_FILENAME)

                with open(config_filename, 'r') as fp:
                    outer_fold_scores = json.load(fp)

                    outer_TR_scores.append(outer_fold_scores['OUTER_TR'])
                    outer_TS_scores.append(outer_fold_scores['OUTER_TS'])

            except Exception as e:
                print(e)

        outer_TR_scores = np.array(outer_TR_scores)
        outer_TS_scores = np.array(outer_TS_scores)

        assessment_results['avg_TR_score'] = outer_TR_scores.mean()
        assessment_results['std_TR_score'] = outer_TR_scores.std()
        assessment_results['avg_TS_score'] = outer_TS_scores.mean()
        assessment_results['std_TS_score'] = outer_TS_scores.std()

        with open(os.path.join(self.__NESTED_FOLDER, self._ASSESSMENT_FILENAME), 'w') as fp:
            json.dump(assessment_results, fp)

    def risk_assessment(self, data, experiment_class, shuffle=True, debug=False, other=None):
        """
        :param data: List of examples. It will be processed by the experiment
        :param experiment_class: the kind of experiment used
        :param shuffle:
        :param debug:
        :param other: anything you want to share across processes
        :return: An average over the outer test folds. RETURNS AN ESTIMATE, NOT A MODEL!!!
        """
        if not os.path.exists(self.__NESTED_FOLDER):
            os.makedirs(self.__NESTED_FOLDER)
        else:
            print("Folder already present! Shutting down to prevent loss of previous experiments")
            #return
        '''
        if shuffle:
            shuffle_dataset(data)

        splits = build_folds(self.outer_folds, data)

        # Store splits separately
        for outer_k in range(self.outer_folds):
            with open(os.path.join(self.__NESTED_FOLDER, 'outer_split_' + str(outer_k + 1)), 'wb') as f:
                pickle.dump(splits[outer_k], f)
        '''
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.outer_processes)
        for outer_k in range(self.outer_folds):

            # Create a separate folder for each experiment
            kfold_folder = os.path.join(self.__NESTED_FOLDER, self.__OUTER_FOLD_BASE + str(outer_k + 1))
            if not os.path.exists(kfold_folder):
                os.makedirs(kfold_folder)

            if not debug:
                pool.submit(self._risk_assessment_helper, outer_k, experiment_class, kfold_folder, debug, other)

            else:  # DEBUG
                self._risk_assessment_helper(outer_k, experiment_class, kfold_folder,
                                             shuffle, debug, other)

        pool.shutdown()  # wait the batch of configs to terminate

        self.process_results()

    def _risk_assessment_helper(self, outer_k, experiment_class, exp_path, shuffle=True, debug=False, other=None):

        # Load specific outer split
        with open(os.path.join(self.__NESTED_FOLDER, 'outer_split_' + str(outer_k + 1)), 'rb') as f:
            train, test = pickle.load(f)

        best_config = self.model_selector.model_selection(train, experiment_class, exp_path, self.model_configs,
                                                          shuffle, debug, other)

        # Retrain with the best configuration and test
        experiment = experiment_class(best_config['config'], exp_path)

        # Set up a log file for this experiment (run in a separate process)

        logger = Logger.Logger(str(os.path.join(experiment.exp_path, 'experiment.log')), mode='a')

        training_score, test_score = experiment.run_test(train, test, other)

        logger.log('TEST FOLD ' + str(outer_k) + ' TR score: ' + str(training_score) + ' TS score: ' + str(test_score))

        with open(os.path.join(exp_path, self._OUTER_RESULTS_FILENAME), 'w') as fp:
            json.dump({'OUTER_TR': training_score, 'OUTER_TS': test_score}, fp)




