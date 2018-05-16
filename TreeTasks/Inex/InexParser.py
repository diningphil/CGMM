import numpy as np
import random

# This is not strictly necessary, one could parse the tree as bottom up losing the symmetric spreading of the context
# (since the neighbourhood is defined by incoming arcs)
# The alternative is simply to parse the tree as undirected, without doubling the number of arc labels
INCOMING = 0
OUTGOING = 1


# If undirected is true, the parser will consider the tree as undirected instead of Bottom-Up
def bu_parse(filename, max_ariety, shuffle=True, undirected=False):

    graphs = []

    lines = open(filename).readlines()
    if shuffle:
        random.shuffle(lines)

    for line in lines:
        u = 0
        children_size = []
        parents = []

        adjacency_lists = []
        X = []
        Y = []

        line = line.strip()
        dim = len(line)
        y = -1  # Use it as y for all the nodes of the tree
        if line[1] == ":":
            y = int(line[0]) - 1
            idx = 2
        else:
            y = int(line[0:2]) - 1  # 2 not included
            idx = 3

        newRoot = True

        stop = False
        while not stop:
            oldidx = idx
            while idx < dim and line[idx] != '(' and line[idx] != ')':
                idx = idx+1

            if idx < dim and line[idx] == "(":

                x = int(line[oldidx:idx]) - 1  # idx not included

                if newRoot:
                    size = 1
                    newRoot = False
                    adjacency_lists.append([])  # no incident arcs
                    X.append(x)
                    Y.append(y)
                    parents.append(-1)
                    children_size.append(0)
                    curr = u
                    u = u+1
                else:
                    size += 1
                    # curr now is the father!
                    father = curr
                    l = children_size[father]

                    # the position in the children list in this case corresponds to the true position
                    adjacency_lists.append([(father, OUTGOING * max_ariety + l)])

                    # Append Incoming edge on the father
                    adjacency_lists[father].append((u, INCOMING * max_ariety + l))

                    # If I want to consider the tree as undirected --> graph
                    if undirected:
                        adjacency_lists[father].append((u, OUTGOING * max_ariety + l))
                        adjacency_lists[u].append((father, INCOMING * max_ariety + l))

                    X.append(x)
                    Y.append(y)
                    parents.append(curr)
                    children_size[curr] += 1
                    curr = u  # curr becomes the node I have processed now
                    u = u+1
                    children_size.append(0)

                if line[idx+1] == "$" and line[idx+2] == ")":
                    curr = parents[curr]
                    idx = idx + 3  # positioning the cursor after the ")"

                else:
                    idx = idx + 1

            elif idx < dim and line[idx] == ")":
                curr = parents[curr]
                idx = idx + 1

            if idx >= dim:
                stop = True

        # A graph is a tuple (X,Y,adj_lists,dim), where Y==X in unsup. tasks
        graphs.append((np.array(X, dtype='int'), np.array(Y, dtype='int'), adjacency_lists, size))

    print("Parsed ", len(graphs), " graphs")

    return graphs
