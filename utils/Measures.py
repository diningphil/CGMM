from scipy.stats import entropy
import numpy as np


def JSD(p, q, base=2):
    '''
    Jensen-Shannon Divergence. It is symmetric. Also bounded between 0 and 1 when the base of the logarithm is 2
    :param p: matrix of distributions, (no_examples, distribution_size)
    :param q: matrix of distributions, (no_examples, distribution_size)
    :param base:
    :return: JS divergence and JS distance (square root of JS divergence). The latter is a metric
    '''
    p /= np.expand_dims(p.sum(axis=1), axis=1)
    q /= np.expand_dims(q.sum(axis=1), axis=1)
    m = (p + q) / 2

    js_divergence = []
    js_distance = []
    for i in range(p.shape[0]):
        js_divergence.append((entropy(p[i, :], m[i, :], base=base) + entropy(q[i, :], m[i, :], base=base)) / 2)
        js_distance.append(np.sqrt(js_divergence))

    js_divergence, js_distance = np.array(js_divergence), np.array(js_distance)
    return js_divergence, js_distance


def compute_total_graphs_entropy(CN, new_fingerprints, old_fingerprints, sizes):
    total_entropy, entropy_per_graph = 0., []
    no_graphs = len(sizes)
    u = 0
    for size in sizes:

        matrix = np.zeros((CN, CN))
        for i in range(len(new_fingerprints[u:u+size])):
            matrix[new_fingerprints[u+i], old_fingerprints[u+i]] += 1

        # Normalize the graph entropy between 0 and 1
        e = graph_entropy(matrix, base=CN)

        total_entropy += (e/no_graphs)
        entropy_per_graph.append(e)

        u += size

    return total_entropy, entropy_per_graph


def graph_entropy(matrix, base=2):
    '''
    Computes the entropy of a graph
    :param matrix: A matrix indexed by i,j, where j represents the previous state of a vertex
    and i stands for the newly generated state of the same vertex. When considering an entire graph,
    as in this case, vertex matrices are summed up.
    :param base:
    :return:
    '''

    g_entropy = 0.
    tot_vertexes = np.sum(matrix)

    for j in range(matrix.shape[1]):
        # j represents the starting state

        col = matrix[:, j]

        # get the number of vertexes in the graph that started from j
        starting_v = np.sum(col)

        if starting_v != 0:
            # create a distribution over the arrival states starting from j
            p = col/np.sum(col)

            # the contribution to the graph entropy is weighted according to the ratio betw.
            # the no of vertexes starting in j and the total number of vertexes
            j_contribution = entropy(p, base=base)*(starting_v/tot_vertexes)
            g_entropy += j_contribution

    return g_entropy
