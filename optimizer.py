from pydgn.training.callback.optimizer import Optimizer
from pydgn.training.event.handler import EventHandler


class CGMMOptimizer(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_eval_epoch_start(self, state):
        """
        Use the "return_node_embeddings" field of the state to decide whether to compute statistics or not during
        this evaluation epoch
        :param state: the shared State object
        """
        state.model.return_node_embeddings = state.return_node_embeddings

    # Not necessary, but it may help to debug
    def on_eval_epoch_end(self, state):
        """
        Reset the "return_node_embeddings" field to False
        :param state:
        :return:
        """
        state.model.return_node_embeddings = False

    def on_training_epoch_end(self, state):
        """
        Calls the M_step to update the parameters
        :param state: the shared State object
        :return:
        """
        state.model.m_step()
