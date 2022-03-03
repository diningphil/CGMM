import torch
from pydgn.training.engine import TrainingEngine
from util import extend_lists, to_tensor_lists


class IncrementalTrainingEngine(TrainingEngine):
    def __init__(self, engine_callback, model, loss, **kwargs):
        super().__init__(engine_callback, model, loss, **kwargs)

    def _to_list(self, data_list, embeddings, batch, edge_index, y):

        if isinstance(embeddings, tuple):
            embeddings = tuple([e.detach().cpu() if e is not None else None for e in embeddings])
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu()
        else:
            raise NotImplementedError('Embeddings not understood, should be Tensor or Tuple of Tensors')

        data_list = extend_lists(data_list, to_tensor_lists(embeddings, batch, edge_index))
        return data_list