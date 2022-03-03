import random

import numpy as np
from pydgn.data.dataset import ZipDataset
from pydgn.data.provider import DataProvider
from pydgn.data.sampler import RandomSampler
from torch.utils.data import Subset


def seed_worker(exp_seed, worker_id):
    np.random.seed(exp_seed + worker_id)
    random.seed(exp_seed + worker_id)


class IncrementalDataProvider(DataProvider):
    """
    An extension of the DataProvider class to deal with the intermediate outputs produced by incremental architectures
    Used by CGMM to deal with node/graph classification/regression.
    """

    def _get_loader(self, indices, **kwargs):
        """
        Takes the "extra" argument from kwargs and zips it together with the original data into a ZipDataset
        :param indices: indices of the subset of the data to be extracted
        :param kwargs: an arbitrary dictionary
        :return: a DataLoader
        """
        dataset = self._get_dataset()
        dataset = Subset(dataset, indices)
        dataset_extra = kwargs.pop("extra", None)

        if dataset_extra is not None and isinstance(dataset_extra, list) and len(dataset_extra) > 0:
            assert len(dataset) == len(dataset_extra), (dataset, dataset_extra)
            datasets = [dataset, dataset_extra]
            dataset = ZipDataset(datasets)
        elif dataset_extra is None or (isinstance(dataset_extra, list) and len(dataset_extra) == 0):
            pass
        else:
            raise NotImplementedError("Check that extra is None, an empty list or a non-empty list")

        shuffle = kwargs.pop("shuffle", False)

        assert self.exp_seed is not None, 'DataLoader seed has not been specified! Is this a bug?'
        kwargs['worker_init_fn'] = lambda worker_id: seed_worker(worker_id, self.exp_seed)
        kwargs.update(self.data_loader_args)

        if shuffle is True:
            sampler = RandomSampler(dataset)
            dataloader = self.data_loader_class(dataset, sampler=sampler,
                                                **kwargs)
        else:
            dataloader = self.data_loader_class(dataset, shuffle=False,
                                                **kwargs)

        return dataloader

