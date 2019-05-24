import re
import os
import numpy as np
import pickle

from utils.utils import shuffle_dataset


def parse_node_labels(filename):
    # ----------------------- PARSE NODE LABELS -------------------------- #
    X = []

    if os.path.exists(filename):
        with open(filename) as f:
            for line in f.readlines():
                # Pass a list, so that the array will have shape (?, 1)
                X.append([float(line)])
        
        X = np.array(X, dtype='int')
        if np.min(X) == 1:
            X = X - 1  # Node labels start from 1
            
        # Compute the maximum number of the labels. Used for categorical distributions
        K = np.max(X) + 1

        return X, K
    return None, None



def parse_targets(filename):
    Y = []
    with open(filename) as f:
        for line in f.readlines():
            if float(line) == -1:  # targets are -1 and 1, binary classification
                Y.append([0.])
            else:
                Y.append([float(line)])
            
    Y = np.array(Y, dtype='int')  # We are assuming classification tasks
    
    if np.min(Y) == 1:
        Y = Y - 1  # Targets start from 1
        
    return Y


def parse_sizes(filename):
    # ----------------------- PARSE SIZES -------------------------- #
    sizes = []
    last_val = None
    with open(filename) as f:
        # If the id found does not change, increase the size, ow stop and start counting the size of the next graph
        for line in f.readlines():
            if last_val is None:
                last_val = float(line)
                size = 1
            else:
                if last_val != float(line):
                    last_val = float(line)
                    sizes.append(size)
                    size = 1
                else:
                    size += 1
        sizes.append(size)
    sizes = np.array(sizes, dtype='int')
    
    return sizes


def parse_structure(adj_filename, edges_filename, no_vertices):
    adj_lists = [[] for _ in range(no_vertices)]

    # ------------------- PARSE EDGES (if present) ------------------- #
    edges = []
    edges_found = False
    if os.path.exists(edges_filename):
        edges_found = True
        with open(edges_filename) as f:
            for line in f.readlines():
                edges.append(float(line))
        edges = np.array(edges, dtype='int')

        if np.min(edges) == 1:
            edges = edges - 1
        A = np.max(edges) + 1
    else:
        A = 1
    # ---------------------------------------------------------------- #

    # NOTE: ALL GRAPHS ARE UNDIRECTED! DS_A HAS 2 ENTRIES FOR EACH EDGE
    with open(adj_filename) as f:
        # I need to order them by destination and source. Because that is the order in which I will read the edges from the adj matrix/lists
        arcs_ordered_by_dest_and_source = []
        count = 0  # to take the associated edge value
        for line in f.readlines():
            el0, el1 = line.split(',')
            src, dst = int(el0)-1, int(el1)-1  # idxes start from 1

            if not edges_found:
                # Just add a dummy edge
                arcs_ordered_by_dest_and_source.append((dst, src, 0))
            else:
                # Append first the destination, then the source and finally the associated edge
                arcs_ordered_by_dest_and_source.append((dst, src, edges[count]))
                count += 1

        # print(arcs_ordered_by_dest_and_source)

        # Re-init edges, because the order of values will change
        edges = []
        
        # Sorts each field of the tuple, starting from the first one.
        arcs_ordered_by_dest_and_source.sort()
        
        for dst, src, val in arcs_ordered_by_dest_and_source:
            adj_lists[dst].append(src)  # add incoming edge
            edges.append(val)    
        
        edges = np.array(edges, dtype='int')       
        if np.min(edges) == 1:
            edges = edges - 1

    # print(arcs_ordered_by_dest_and_source)
    return adj_lists, edges, A


def parse(task_name):
    node_labels_filename = os.path.join(task_name + '/' + task_name + '_node_labels.txt')
    node_attrib_filename = os.path.join(task_name + '/' + task_name + '_node_attributes.txt')
    graph_labels_filename = os.path.join(task_name + '/' + task_name + '_graph_labels.txt')
    sizes_filename = os.path.join(task_name + '/' + task_name + '_graph_indicator.txt')
    adj_mat_filename = os.path.join(task_name + '/' + task_name + '_A.txt')
    edge_labels_filename = os.path.join(task_name + '/' + task_name + '_edge_labels.txt')

    # ----------------------- PARSE TARGETS -------------------------- #
    Y = parse_targets(graph_labels_filename)
    # ---------------------------------------------------------------- #
    # ----------------------- PARSE SIZES -------------------------- #
    sizes = parse_sizes(sizes_filename)
    # ---------------------------------------------------------------- #

    # ----------------------- PARSE NODE LABELS ---------------------- #
    X, K = parse_node_labels(node_labels_filename)
    if os.path.exists(node_attrib_filename):
        X_att, K_att = parse_node_labels(node_attrib_filename)
    # ---------------------------------------------------------------- #

    # ----------------------- PARSE ADJ MAT -------------------------- #
    no_vertices = sum(sizes)

    # ------------------- PARSE EDGES (if present) ------------------- #
    adj_lists, edges, A = parse_structure(adj_mat_filename, edge_labels_filename, no_vertices)
    # ---------------------------------------------------------------- #
    
    # Now construct the dataset of graphs
    graphs = []
    no_graphs = len(Y)

    start_node_idx = 0
    start_edge_idx = 0
    parsed = 0

    # If we do not have node labels, add structural features!
    neighb = []
    for adj_list in adj_lists:
        neighb.append([len(adj_list)])
    neighb = np.array(neighb, dtype='float64')

    if not os.path.exists(node_labels_filename):
        X = neighb
        K = 1

    '''
    if task_name == 'PROTEINS':
        X = X_att
        K = 1
    '''

    # Now convert the idxes values to work between 0 and no_vertices. We will not distinguish between graphs
    # when learning distributions!
    for i in range(no_graphs):
        size = sizes[i]

        X_graph = np.array(X[start_node_idx: start_node_idx + size])
        Y_graph = np.array([Y[i]])
        adjacency_graph = []
        edges_graph = []

        curr = 0
        max_ariety = 0
        for adj_list in adj_lists[start_node_idx:start_node_idx + size]:

            if K == 1:
                X_graph[curr] = len(adj_list)

            curr += 1

            if len(adj_list) > max_ariety:
                max_ariety = len(adj_list)

            adjacency_graph.append([idx - start_node_idx for idx in adj_list])

            edge_size = len(adj_list)
            edges_graph.extend(edges[start_edge_idx:start_edge_idx + edge_size])

        # print(adjacency_graph)
        graphs.append((X_graph, edges_graph, Y_graph, adjacency_graph, size, max_ariety))
        start_node_idx += size
        start_edge_idx += edge_size
        parsed += 1

    print("Parsed ", parsed, " structures")
    
    # ---------------------------------------------------------------- #

    print('K is', K, 'A is', A)

    shuffle_dataset(graphs)

    with open(os.path.join(task_name + '/' + task_name + '_dataset'), 'wb') as f:
        pickle.dump([graphs, A, K], f)


for task_name in ['MUTAG']:

    print("Parsing", task_name, "...")
    parse(task_name)
    

