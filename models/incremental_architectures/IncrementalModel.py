import os
import shutil

import torch
from torch.utils.data import SequentialSampler

from matplotlib import pyplot as plt
import seaborn as sns


class IncrementalLayer(torch.nn.Module):

    def __init__(self, dim_features, dim_target, depth, layer_config, checkpoint=None,
                 loss_class=None, optim_class=None, sched_class=None, stopper_class=None, clipping=None, device='cpu', **kwargs):
        super().__init__()

        self.is_first_layer = depth == 1
        self.depth = depth
        self.device = device

        self.dim_features = dim_features
        self.dim_target = dim_target

        self.layer_config = layer_config

        self.loss_class = loss_class
        self.optim_class = optim_class
        self.sched_class = sched_class
        self.stopper_class = stopper_class
        self.clipping = clipping

        self.checkpoint = checkpoint

    def train_layer(self, data_loader, L, concat_axis, validation_loader=None, device='cpu'):
        raise NotImplementedError()

    def infer(self, data_loader, device='cpu'):
        raise NotImplementedError()

    def checkpoint(self):
        """
        :return: layer state dict to store
        """
        raise NotImplementedError()

    def restore(self, checkpoint):
        raise NotImplementedError()

    def arbitrary_logic(self, train_loader, layer_config, is_last_layer, validation_loader=None, test_loader=None,
                        logger=None):
        return {}

    def stopping_criterion(self, depth, dict_per_layer, layer_config, logger=None):
        return False


class IncrementalModel:

    def __init__(self, layer_class, dim_features, dim_target, exp_path, loss_class, optim_class, sched_class,
                 stopper_class, clipping=None, device='cpu'):
        self.device = device
        self.layers = []
        self.model = layer_class
        self.l_prec = None
        self._concat_axis = None
        self.main_folder = exp_path
        self.dim_features = dim_features
        self.dim_target = dim_target
        self.output_folder = os.path.join(self.main_folder, 'outputs')
        self.checkpoint_folder = os.path.join(self.main_folder, 'checkpoint')

        self.loss_class = loss_class
        self.optim_class = optim_class
        self.sched_class = sched_class
        self.stopper_class = stopper_class
        self.clipping = clipping

    def _load_outputs(self, mode, prev_outputs_to_consider):

        outs_dict = {
            'vertex_outputs': None,
            'edge_outputs': None,
            'graph_outputs': None,
            'other_outputs': None
        }

        for prev in prev_outputs_to_consider:
            for path, o_key in [(os.path.join(self.output_folder, mode, f'vertex_output_{prev}.pt'), 'vertex_outputs'),
                                (os.path.join(self.output_folder, mode, f'edge_output_{prev}.pt'), 'edge_outputs'),
                                (os.path.join(self.output_folder, mode, f'graph_output_{prev}.pt'), 'graph_outputs'),
                                (os.path.join(self.output_folder, mode, f'other_output_{prev}.pt'), 'other_outputs')]:
                if os.path.exists(path):
                    out = torch.load(path)
                    outs = outs_dict[o_key]

                    if outs is None:
                        # print('None!')
                        outs = [None] * len(out)
                    else:
                        pass
                        # print('Not none!')

                    for graph_id in range(len(out)):  # iterate over graphs
                        outs[graph_id] = out[graph_id] if outs[graph_id] is None \
                            else torch.cat((out[graph_id], outs[graph_id]), self._concat_axis)

                    outs_dict[o_key] = outs

        return outs_dict['vertex_outputs'], outs_dict['edge_outputs'],\
            outs_dict['graph_outputs'], outs_dict['other_outputs']

    def _store_outputs(self, mode, depth, v_tensor_list, e_tensor_list=None, g_tensor_list=None, o_tensor_list=None):

        if not os.path.exists(os.path.join(self.output_folder, mode)):
            os.makedirs(os.path.join(self.output_folder, mode))

        if v_tensor_list is not None:
            vertex_filepath = os.path.join(self.output_folder, mode, f'vertex_output_{depth}.pt')
            torch.save([torch.unsqueeze(v_tensor, self._concat_axis) for v_tensor in v_tensor_list], vertex_filepath)
        if e_tensor_list is not None:
            edge_filepath = os.path.join(self.output_folder, mode, f'edge_output_{depth}.pt')
            torch.save([torch.unsqueeze(e_tensor, self._concat_axis) for e_tensor in e_tensor_list], edge_filepath)
        if g_tensor_list is not None:
            graph_filepath = os.path.join(self.output_folder, mode, f'graph_output_{depth}.pt')
            torch.save([torch.unsqueeze(g_tensor, self._concat_axis) for g_tensor in g_tensor_list], graph_filepath)
        if o_tensor_list is not None:
            other_filepath = os.path.join(self.output_folder, mode, f'other_output_{depth}.pt')
            torch.save([torch.unsqueeze(o_tensor, self._concat_axis) for o_tensor in o_tensor_list], other_filepath)

    def checkpoint(self):
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        for i, layer in enumerate(self.layers):
            torch.save(layer.checkpoint(), os.path.join(self.checkpoint_folder, f'layer_{i+1}.pt'))

    def restore(self, no_layers):
        self.layers = [self.layer(checkpoint=torch.load(os.path.join(self.checkpoint_folder, f'layer_{i+1}.pt')))
                       for i in range(no_layers)]

    def incremental_training(self, train_loader, max_layers, layer_config, validation_loader=None, test_loader=None,
                             concatenate_axis=1, save=True, resume=False, logger=None, device='cpu'):

        assert concatenate_axis > 0, 'You cannot concat on the first axis for design reasons.'
        if resume:
            raise NotImplementedError()

        self.layers = []
        self.l_prec = layer_config['l_prec']  # a list of previous layers to consider e.g. [1,2] === the prev 2 layers
        self._concat_axis = concatenate_axis

        dict_per_layer = []

        stop = False
        depth = 1
        while not stop and depth <= max_layers:

            prev_outputs_to_consider = [(depth - x) for x in self.l_prec if (depth - x) > 0]

            new_layer = self.model(self.dim_features, self.dim_target, depth, layer_config, None, self.loss_class,
                                   self.optim_class, self.sched_class, self.stopper_class, self.clipping)

            for data_loader, mode in [(train_loader, 'train'), (validation_loader, 'validation'), (test_loader, 'test')]:
                if data_loader is not None:

                    # Needed to reset variables, DO NOT remove it
                    v_outs, e_outs, g_outs, o_outs = None, None, None, None
                    v_out, e_out, g_out, o_out = None, None, None, None

                    # Load previous outputs if any according to prev. layers to consider (ALL TENSORS)
                    v_outs, e_outs, g_outs, o_outs = self._load_outputs(mode, prev_outputs_to_consider)

                    data_loader.dataset.augment(v_outs, e_outs, g_outs, o_outs)

                    if mode == 'train':  # must be executed before validation and test
                        # Train new Layer
                        new_layer.train_layer(data_loader, len(prev_outputs_to_consider), self._concat_axis,
                                              validation_loader, device=device)

                    # Infer and produce outputs
                    v_out, e_out, g_out, o_out = new_layer.infer(data_loader, device=device)

                    plt.figure()
                    sns.heatmap(v_out[0].detach().cpu().numpy())
                    plt.savefig(f'prova_{depth}.png', dpi=200)
                    plt.close()

                    # Reorder outputs, which are produced in shuffled order, to the original arrangement of the dataset.
                    v_out, e_out, g_out, o_out = self._reorder_shuffled_objects(v_out, e_out, g_out, o_out, data_loader)

                    # Store outputs
                    self._store_outputs(mode, depth, v_out, e_out, g_out, o_out)

            prev_outputs_to_consider = [l for l in range(1, depth+1)]

            v_outs, e_outs, g_outs, o_outs = self._load_outputs('train', prev_outputs_to_consider)



            train_loader.dataset.augment(v_outs, e_outs, g_outs, o_outs)
            if validation_loader is not None:
                v_outs, e_outs, g_outs, o_outs = self._load_outputs('validation', prev_outputs_to_consider)
                validation_loader.dataset.augment(v_outs, e_outs, g_outs, o_outs)
            if test_loader is not None:
                v_outs, e_outs, g_outs, o_outs = self._load_outputs('test', prev_outputs_to_consider)
                test_loader.dataset.augment(v_outs, e_outs, g_outs, o_outs)

            # Execute arbitrary logic e.g. to compute scores
            d = new_layer.arbitrary_logic(train_loader, layer_config, is_last_layer=(depth == max_layers),
                                          validation_loader=validation_loader, test_loader=test_loader, logger=logger, device=device)

            # Stopping criterion
            stop = new_layer.stopping_criterion(depth, dict_per_layer, layer_config, logger=logger)
            d['stop'] = stop

            # Append layer
            self.layers.append(new_layer)

            dict_per_layer.append(d)
            depth += 1

        # CLEAR OUTPUTS TO SAVE SPACE
        for mode in ['train', 'validation', 'test']:
            shutil.rmtree(os.path.join(self.output_folder, mode), ignore_errors=True)

        # Potential checkpoint
        if save:
            self.checkpoint()

        return dict_per_layer

    '''
    def incremental_inference(self, dataset, name, layer_config):
        """
        Use it when you want to infer from scratch
        :param dataset:
        :param name:
        :return:
        """

        dict_per_layer = []

        for depth in range(1, len(self.layers)+1):

            # Load previous outputs if any according to Lprec
            v_outs, e_outs, g_outs, o_outs = self._load_outputs(name, depth)

            # TODO embed previous outputs in geometric dataloaders, and then iterate over them in parallel.

            # Infer and take output
            v_out, e_out, g_out, o_out, score = self.layers[depth-1].infer(dataset, v_outs, e_outs, g_outs, o_outs)

            # Store outputs
            self._store_outputs(name, depth, v_out, e_out, g_out, o_out)

            # Execute arbitrary logic  e.g. to compute scores
            dict_per_layer.append(self.arbitrary_logic(depth, name, layer_config))

            # Potential outputs cleanup (user specifies with bool) of non-used stats if l_prec = 1
            # TODO if the user wants it we can clean things that are not relevant

        return dict_per_layer
    '''

    @staticmethod
    def _reorder_shuffled_objects(v_out, e_out, g_out, o_out, data_loader):
        if type(data_loader.sampler) == SequentialSampler:  # No permutation
            return v_out, e_out, g_out, o_out

        idxs = data_loader.sampler.permutation  # permutation of the last data_loader iteration

        def reorder(obj, perm):
            assert len(obj) == len(perm) and len(obj) > 0
            return [y for (x, y) in sorted(zip(perm, obj))]

        if g_out is not None:
            #print(len(g_out))
            g_out = reorder(g_out, idxs)

        if o_out is not None:
            #print(len(o_out))
            o_out = reorder(o_out, idxs)

        if v_out is not None:
            #print(len(v_out))
            v_out = reorder(v_out, idxs)

        if e_out is not None:
            raise NotImplementedError('This feature has not been implemented yet!')
            # e_out = reorder(e_out, idxs)

        return v_out, e_out, g_out, o_out
