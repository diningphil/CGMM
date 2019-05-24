import copy


class EarlyStopper:

    def stop(self, val, epoch):
        raise NotImplementedError("Implement this method!")


class GLStopper(EarlyStopper):

    '''
    Implement Generalization Loss technique (Prechelt 1997)
    '''

    def __init__(self, model, alpha=5, less_is_best=True):
        self.local_optimum = float("inf") if less_is_best else -float("inf")
        self.less_is_best = less_is_best
        self.alpha = alpha
        self.model = model
        self.best_state_dict = None
        self.best_epoch = None

    def stop(self, val, epoch):
        if self.less_is_best:
            if val <= self.local_optimum:
                self.local_optimum = val
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                return False
            else:
                return 100*(val/self.local_optimum - 1) > self.alpha
        else:
            if val >= self.local_optimum:
                self.local_optimum = val
                self.best_state_dict = copy.deepcopy(self.model.state_dict())  # get a copy
                self.best_epoch = epoch
                return False
            else:
                return 100*(self.local_optimum/val - 1) > self.alpha

        assert False, 'You cannot be here'

    def get_best_params(self):
        return self.best_state_dict

    def get_best_epoch(self):
        return self.best_epoch

    def get_best_validation_score(self):
        return self.local_optimum
