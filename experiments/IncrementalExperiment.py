from models.incremental_architectures.IncrementalModel import IncrementalModel

from experiments.Experiment import Experiment


class IncrementalExperiment(Experiment):

    def __init__(self, model_configuration, exp_path):
        super(IncrementalExperiment, self).__init__(model_configuration, exp_path)

    def run_valid(self, dataset_getter, logger, other=None):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        dataset_class = self.model_config.dataset  # dataset_class()
        dataset = dataset_class()
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping
        device = self.model_config.device

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)

        architecture = IncrementalModel(model_class, dataset.dim_features, dataset.dim_target, self.exp_path,
                                        loss_class, optim_class, sched_class, stopper_class, clipping=clipping, device=device)

        res = architecture.incremental_training(train_loader, self.model_config['max_layers'], self.model_config,
                                                val_loader, test_loader=None, concatenate_axis=1, save=False,
                                                resume=False, logger=logger, device=self.model_config['device'])

        # Use last training and validation scores
        return res[-1]['train_score'], res[-1]['validation_score']

    def run_test(self, dataset_getter, logger, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR ANY REASON
        :return: (training accuracy, test accuracy)
        """

        dataset_class = self.model_config.dataset  # dataset_class()
        dataset = dataset_class()
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping
        device = self.model_config.device

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size']
                                                                , shuffle=shuffle)
        test_loader = dataset_getter.get_test(dataset, self.model_config['batch_size'], shuffle=shuffle)

        architecture = IncrementalModel(model_class, dataset.dim_features, dataset.dim_target, self.exp_path,
                                        loss_class, optim_class, sched_class, stopper_class, clipping=clipping, device=device)

        res = architecture.incremental_training(train_loader, self.model_config['max_layers'], self.model_config,
                                                val_loader, test_loader=test_loader, concatenate_axis=1,
                                                save=False, resume=False, logger=logger)

        # Use last training and test scores
        return res[-1]['train_score'], res[-1]['test_score']
