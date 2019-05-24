import numpy as np

# CONTEXTUAL SIMPLY GENERATES DOUBLE ARCS WITH THE SAME LABEL. NO NEED TO USE 2 DIFFERENT LABELS HERE

def generateWrappedDataset(quantity, max_str_size, data_type, contextual=True, fixed_size=False):
    
    if data_type == 'sequence':
        graphs = generateSequences(quantity, max_str_size, contextual, fixed_size)
    elif data_type == 'tree':
        graphs = generateRootedTree(quantity, max_str_size, 0, None, contextual, fixed_size)
    elif data_type == 'dpag':
        graphs = generateDPAG(quantity, max_str_size, 0, None, contextual, fixed_size)
    else:
        raise Exception("Type not understood")

    
    # Now the adjacency list is shifted
    wrapped_graphs = []
    for graph in graphs:

        X_g, edges_g, _, adj_lists_g, size_g, max_ariety_g = graph
        
        new_X_g = X_g
        new_edges_g = list(np.array(edges_g, copy=True))
        
        new_Y_g = np.array([[1]], dtype=np.int)   
        
        new_adj_lists_g = [[u for u in l] for l in adj_lists_g]
        
        new_adj_lists_g[-1].extend([0])
        new_edges_g.append(0)
        if contextual:
            new_adj_lists_g[0].extend([size_g-1])
            new_edges_g.append(0)

        wrapped_graphs.append((new_X_g, new_edges_g, new_Y_g, new_adj_lists_g, size_g, max_ariety_g))

    return graphs + wrapped_graphs

def generateSequences(quantity, maxSize, contextual=True, fixed_size=False):
    sequences = []
    
    for _ in range(0, quantity):
        # Choose a size for a sequence
        if fixed_size:
            size = maxSize
        else:
            size = np.random.randint(1, maxSize) + 1
            
        X_g = np.zeros(size, dtype=np.int)
        Y_g = np.zeros((1,1), dtype=np.int)
        edges_g = []
        adjacency_lists = []
        # print(f'size is {size}')
        # Used to connect beginning and end of a sequence if requested
        for u in range(size):
            if u == 0:
                l = [u+1]
                edges_g.append(0)
            elif u == size-1:
                l = [u-1]
                edges_g.append(0)
            else:  # This way I can capture the context of all my neighbours
                if contextual:
                    l = [u + 1, u - 1]
                    edges_g.append(0)
                    edges_g.append(0)
                else:
                    l = [u + 1]
                    edges_g.append(0)

            adjacency_lists.append(l)

        max_ariety = 0
        for l in adjacency_lists:
            if len(l) > max_ariety:
                max_ariety = len(l)

        # Append the dimension of this structure
        sequences.append((X_g, edges_g, Y_g, adjacency_lists, size, max_ariety))
    
    # print(sequences)
    return sequences


def generateRootedTree(quantity, maxSize, u_start, prevAdLists=None, contextual=True, fixed_size=False):
    if prevAdLists is None:
        adjacency_lists = []
    else:
        adjacency_lists = prevAdLists
    sizes = []

    # tmp = u_start

    u = u_start
    for _ in range(0, quantity):
        # Choose a size for a tree
        if fixed_size:
            size = maxSize
        else:
            size = np.random.randint(5, maxSize) + 1

        adjacency_lists.extend([[] for _ in range(0, size)])

        is_a_tree = False  # Check if a father as >= 2 children
        fathers = [False for _ in range(0, size)]

        u_start = u
        for __ in range(0, size-1):  # The last node has no incoming arcs

            father = np.random.randint(low=u+1, high=u_start+size)
            if fathers[father - u_start]:
                is_a_tree = True
            fathers[father - u_start] = True

            adjacency_lists[u].append(father)
            if contextual:
                adjacency_lists[father].append(u)
            u = u+1

        # The root has already all the outcoming arcs (if contextual)
        u = u+1

        # Append the dimension of this structure
        sizes.append(size)

        assert is_a_tree

    # print(u, u-tmp, np.cumsum(sizes)[-1]+tmp, len(adjacency_lists))
    new_u_start = u
    return adjacency_lists, new_u_start, sizes


def generateDPAG(quantity, maxSize, u_start, prevAdLists=None, contextual=True, fixed_size=False):
    if prevAdLists is None:
        adjacency_lists = []
    else:
        adjacency_lists = prevAdLists

    sizes = []
    u = u_start
    # tmp = u_start

    for _ in range(0, quantity):
        # Choose a size for a dpag
        if fixed_size:
            size = maxSize
        else:
            size = np.random.randint(5, maxSize) + 1

        adjacency_lists.extend([[] for _ in range(0, size)])

        is_a_dpag = False

        u_start = u
        for __ in range(0, size-1):  # The last node has no incident arcs

            # Choose a number of fathers
            no_fathers = np.random.randint(u_start+size-u-1) + 1

            if no_fathers > 1:
                is_a_dpag = True

            # Select the range of ids in which to choose the father
            possible = np.arange(u+1, u_start+size)
            # Shuffle and select the first "no_fathers" fathers
            np.random.shuffle(possible)

            adlist = []
            for i in range(0, no_fathers):
                father = possible[i]
                adlist.append(father)
                if contextual:
                    adjacency_lists[father].append(u)

            adjacency_lists[u].extend(adlist)
            u = u+1

        # The root has already all the outcoming arcs (if contextual)
        u = u+1

        # Append the dimension of this structure
        sizes.append(size)

        assert is_a_dpag

    # print(u, u - tmp, np.cumsum(sizes)[-1] + tmp, len(adjacency_lists))
    new_u_start = u
    return adjacency_lists, new_u_start, sizes


def generateDirectedCyclic(quantity, maxSize, u_start, prevAdLists=None, contextual=True, fixed_size=False):
    if prevAdLists is None:
        adjacency_lists = []
    else:
        adjacency_lists = prevAdLists

    sizes = []
    u = u_start

    # tmp = u_start

    for _ in range(0, quantity):
        # Choose a size for a directed cyclic graph
        if fixed_size:
            size = maxSize
        else:
            size = np.random.randint(2, maxSize) + 1

        adjacency_lists.extend([[] for _ in range(0, size)])

        u_start = u
        for __ in range(0, size):
            # Choose a number of incident nodes
            adlist = []
            no_incident_nodes = np.random.randint(1, size)
            seen = []
            for ___ in range(0, no_incident_nodes):

                incident_node = np.random.randint(u_start, u_start+size)

                if incident_node != u and not incident_node in seen:
                    adlist.append(incident_node)
                    if contextual:
                        adjacency_lists[incident_node].append(u)
                    seen.append(incident_node)

            adjacency_lists[u].extend(adlist)
            u = u+1
        sizes.append(size)

    # print(u, u-tmp, np.cumsum(sizes)[-1]+tmp, len(adjacency_lists))
    new_u_start = u
    return adjacency_lists, new_u_start, sizes
