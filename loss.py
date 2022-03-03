from pydgn.training.callback.metric import Metric


class CGMMLoss(Metric):
    @property
    def name(self) -> str:
        return 'CGMM Loss'

    def __init__(self, use_as_loss=True, reduction='mean', use_nodes_batch_size=True):
        super().__init__(use_as_loss=use_as_loss, reduction=reduction, use_nodes_batch_size=use_nodes_batch_size)
        self.old_likelihood = -float('inf')
        self.new_likelihood = None

    def on_training_batch_end(self, state):
        self.batch_metrics.append(state.batch_loss[self.name].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    def on_training_epoch_end(self, state):
        super().on_training_epoch_end(state)

        if (state.epoch_loss[self.name].item() - self.old_likelihood) < 0:
            pass
            # tate.stop_training = True
        self.old_likelihood = state.epoch_loss[self.name].item()

    def on_eval_batch_end(self, state):
        self.batch_metrics.append(state.batch_loss[self.name].item())
        if state.model.is_graph_classification:
            self.num_samples += state.batch_num_targets
        else:
            # This works for unsupervised CGMM
            self.num_samples += state.batch_num_nodes

    # Simply ignore targets
    def forward(self, targets, *outputs):
        likelihood = outputs[2]
        return likelihood

    def on_backward(self, state):
        pass

