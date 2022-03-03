from typing import Optional, Tuple, List

import torch
import torch_geometric


def extend_lists(data_list: Optional[Tuple[Optional[List[torch.Tensor]]]],
                 new_data_list: Tuple[Optional[List[torch.Tensor]]]) -> Tuple[Optional[List[torch.Tensor]]]:
    r"""
    Extends the semantic of Python :func:`extend()` over lists to tuples
    Used e.g., to concatenate results of mini-batches in incremental architectures such as :obj:`CGMM`

    Args:
        data_list: tuple of lists, or ``None`` if there is no list to extend.
        new_data_list: object of the same form of :obj:`data_list` that has to be concatenated

    Returns:
        the tuple of extended lists
    """
    if data_list is None:
        return new_data_list

    assert len(data_list) == len(new_data_list)

    for i in range(len(data_list)):
        if new_data_list[i] is not None:
            data_list[i].extend(new_data_list[i])

    return data_list


def to_tensor_lists(embeddings: Tuple[Optional[torch.Tensor]],
                    batch: torch_geometric.data.batch.Batch,
                    edge_index: torch.Tensor) -> Tuple[Optional[List[torch.Tensor]]]:
    r"""
    Reverts batched outputs back to a list of Tensors elements.
    Can be useful to build incremental architectures such as :obj:`CGMM` that store intermediate results
    before training the next layer.

    Args:
        embeddings (tuple): a tuple of embeddings :obj:`(vertex_output, edge_output, graph_output, vertex_extra_output, edge_extra_output, graph_extra_output)`.
                            Each embedding can be a :class:`torch.Tensor` or ``None``.
        batch (:class:`torch_geometric.data.batch.Batch`): Batch information used to split the tensors.

        edge_index (:class:`torch.Tensor`): a :obj:`2 x num_edges` tensor as defined in Pytorch Geometric.
                                            Used to split edge Tensors graph-wise.

    Returns:
        a tuple with the same semantics as the argument ``embeddings``, but this time each element holds a list of
        Tensors, one for each graph in the dataset.
    """
    # Crucial: Detach the embeddings to free the computation graph!!
    # TODO this code can surely be made more compact, but leave it as is until future refactoring or removal from PyDGN.
    v_out, e_out, g_out, vo_out, eo_out, go_out = embeddings

    v_out = v_out.detach() if v_out is not None else None
    v_out_list = [] if v_out is not None else None

    e_out = e_out.detach() if e_out is not None else None
    e_out_list = [] if e_out is not None else None

    g_out = g_out.detach() if g_out is not None else None
    g_out_list = [] if g_out is not None else None

    vo_out = vo_out.detach() if vo_out is not None else None
    vo_out_list = [] if vo_out is not None else None

    eo_out = eo_out.detach() if eo_out is not None else None
    eo_out_list = [] if eo_out is not None else None

    go_out = go_out.detach() if go_out is not None else None
    go_out_list = [] if go_out is not None else None

    _, node_counts = torch.unique_consecutive(batch, return_counts=True)
    node_cumulative = torch.cumsum(node_counts, dim=0)

    if e_out is not None or eo_out is not None:
        edge_batch = batch[edge_index[0]]
        _, edge_counts = torch.unique_consecutive(edge_batch, return_counts=True)
        edge_cumulative = torch.cumsum(edge_counts, dim=0)

    if v_out_list is not None:
        v_out_list.append(v_out[:node_cumulative[0]])

    if e_out_list is not None:
        e_out_list.append(e_out[:edge_cumulative[0]])

    if g_out_list is not None:
        g_out_list.append(g_out[0].unsqueeze(0))  # recreate batch dimension by unsqueezing

    if vo_out_list is not None:
        vo_out_list.append(vo_out[:node_cumulative[0]])

    if eo_out_list is not None:
        eo_out_list.append(eo_out[:edge_cumulative[0]])

    if go_out_list is not None:
        go_out_list.append(go_out[0].unsqueeze(0))  # recreate batch dimension by unsqueezing

    for i in range(1, len(node_cumulative)):
        if v_out_list is not None:
            v_out_list.append(v_out[node_cumulative[i - 1]:node_cumulative[i]])

        if e_out_list is not None:
            e_out_list.append(e_out[edge_cumulative[i - 1]:edge_cumulative[i]])

        if g_out_list is not None:
            g_out_list.append(g_out[i].unsqueeze(0))  # recreate batch dimension by unsqueezing

        if vo_out_list is not None:
            vo_out_list.append(vo_out[node_cumulative[i - 1]:node_cumulative[i]])

        if eo_out_list is not None:
            eo_out_list.append(eo_out[edge_cumulative[i - 1]:edge_cumulative[i]])

        if go_out_list is not None:
            go_out_list.append(go_out[i].unsqueeze(0))  # recreate batch dimension by unsqueezing

    return v_out_list, e_out_list, g_out_list, vo_out_list, eo_out_list, go_out_list


def compute_unigram(posteriors: torch.Tensor, use_continuous_states: bool) -> torch.Tensor:
    r"""
    Computes the unigram representation of nodes as defined in https://www.jmlr.org/papers/volume21/19-470/19-470.pdf

    Args:
        posteriors (torch.Tensor): tensor of posterior distributions of nodes with shape `(#nodes,num_latent_states)`
        use_continuous_states (bool): whether or not to use the most probable state (one-hot vector) or a "soft" version

    Returns:
        a tensor of unigrams with shape `(#nodes,num_latent_states)`
    """
    num_latent_states = posteriors.shape[1]

    if use_continuous_states:
        node_embeddings_batch = posteriors
    else:
        node_embeddings_batch = make_one_hot(posteriors.argmax(dim=1), num_latent_states)

    return node_embeddings_batch.double()


def compute_bigram(posteriors: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                   use_continuous_states: bool) -> torch.Tensor:
    r"""
    Computes the bigram representation of nodes as defined in https://www.jmlr.org/papers/volume21/19-470/19-470.pdf

    Args:
        posteriors (torch.Tensor): tensor of posterior distributions of nodes with shape `(#nodes,num_latent_states)`
        edge_index (torch.Tensor): tensor of edge indices with shape `(2,#edges)` that adheres to PyG specifications
        batch (torch.Tensor): vector that assigns each node to a graph id in the batch
        use_continuous_states (bool): whether or not to use the most probable state (one-hot vector) or a "soft" version

    Returns:
        a tensor of bigrams with shape `(#nodes,num_latent_states*num_latent_states)`
    """
    C = posteriors.shape[1]
    device = posteriors.get_device()
    device = 'cpu' if device == -1 else device

    if use_continuous_states:
        # Code provided by Daniele Atzeni to speed up the computation!
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]).to(device),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors.float()).repeat(1, C)
        tmp2 = posteriors.reshape(-1, 1).repeat(1, C).reshape(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)
    else:
        # Convert into one hot
        posteriors_one_hot = make_one_hot(posteriors.argmax(dim=1), C).float()

        # Code provided by Daniele Atzeni to speed up the computation!
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]).to(device),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors_one_hot).repeat(1, C)
        tmp2 = posteriors_one_hot.reshape(-1, 1).repeat(1, C).reshape(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)

    return node_bigram_batch.double()


def make_one_hot(labels: torch.Tensor, num_unique_ids: torch.Tensor) -> torch.Tensor:
    r"""
    Converts a vector of ids into a one-hot matrix

    Args:
        labels (torch.Tensor): the vector of ids
        num_unique_ids (torch.Tensor): number of unique ids

    Returns:
        a one-hot tensor with shape `(samples,num_unique_ids)`
    """
    device = labels.get_device()
    device = 'cpu' if device == -1 else device
    one_hot = torch.zeros(labels.size(0), num_unique_ids).to(device)
    one_hot[torch.arange(labels.size(0)).to(device), labels] = 1
    return one_hot
