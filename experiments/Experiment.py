import random

from models import TestModel


class Experiment:
    """
    Experiment provides a layer of abstraction to avoid that all models implement the same interface
    """

    def __init__(self, model_configuration, exp_path):
        self.model_config = model_configuration
        self.exp_path     = exp_path

    def run_valid(self, train_data, test_data, other=None):
        """
        This function returns the training and validation accuracy. DO WHATEVER YOU WANT WITH VL SET,
        BECAUSE YOU WILL MAKE PERFORMANCE ASSESSMENT ON A TEST SET
        :return: (training accuracy, validation accuracy)
        """
        raise NotImplementedError('You must implement this function!')

    def run_test(self, train_data, test_data, other=None):
        """
        This function returns the training and test accuracy
        :return: (training accuracy, test accuracy)
        """
        raise NotImplementedError('You must implement this function!')


class ToyExperiment(Experiment):

    def __init__(self, model_configuration, exp_path):
        super(ToyExperiment, self).__init__(model_configuration, exp_path)
        self.model_class = TestModel

    def run_valid(self, train_data, valid_data, other=None):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """
        return random.uniform(0, 100), random.uniform(0, 100)

    def run_test(self, train_data, test_data, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR ANY REASON
        :return: (training accuracy, test accuracy)
        """
        return random.uniform(0, 100), random.uniform(0, 100)
