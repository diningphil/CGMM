from typing import List

import torch
from pydgn.training.callback.metric import Metric


class CGMMCompleteLikelihoodScore(Metric):

    @property
    def name(self) -> str:
        return 'Complete Log Likelihood'

    def __init__(self, use_as_loss=False, reduction='mean', use_nodes_batch_size=True):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)

    def on_training_batch_end(self, state):
        self.batch_metrics.append(state.batch_score[self.name].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def on_eval_epoch_end(self, state):
        state.update(epoch_score={self.name: torch.tensor(self.batch_metrics).sum() / self.num_samples})
        self.batch_metrics = None
        self.num_samples = None

    def on_eval_batch_end(self, state):
        self.batch_metrics.append(state.batch_score[self.name].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return outputs[2]

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return outputs[2]


class CGMMTrueLikelihoodScore(Metric):

    @property
    def name(self) -> str:
        return 'True Log Likelihood'

    def __init__(self, use_as_loss=False, reduction='mean', use_nodes_batch_size=True):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)

    def on_training_batch_end(self, state):
        self.batch_metrics.append(state.batch_score[self.name].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def on_eval_batch_end(self, state):
        self.batch_metrics.append(state.batch_score[self.name].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def _score_fun(self, targets, *outputs, batch_loss_extra):
        return outputs[3]

    def forward(self, targets: torch.Tensor, *outputs: List[torch.Tensor], batch_loss_extra: dict = None) -> dict:
        return outputs[3]
