import numpy as np
import pickle

#task_name = 'CPDB'
task_name = 'MUTAG'

# ----------------------- PARSE TARGETS -------------------------- #
target = []
with open(task_name + "_task/" + task_name + "_labels.txt") as f:
    for line in f.readlines():
        target.append(int(line))
target = np.array(target, dtype='int')
# ---------------------------------------------------------------- #

# ----------------------- PARSE DATASET -------------------------- #
graphs_0 = []
graphs_1 = []
A = 0

with open(task_name + "_task/" + task_name + ".mol") as f:
    end = False
    c = 0  # a counter of examples
    # Parsing a graph in the loop
    while not end:

        X_graph = []
        Y_graph = []

        skip = f.readline()
        skip_id = f.readline()
        skip = f.readline()
        counts = f.readline().split()  # default delimiter is space
        if len(counts) == 0:
            end = True
        if not end:
            no_atoms, no_edges = int(counts[0]), int(counts[1])
            # print(no_atoms, no_edges)

            size = no_atoms
            adjacency_lists_graph = [[] for _ in range(0, no_atoms)]

            for _ in range(0, no_atoms):
                atom_symbol = f.readline().split()[3]
                X_graph.append(atom_symbol)
                Y_graph.append(target[c])

            for _ in range(0, no_edges):
                edge_data = f.readline().split()
                # node ids shall start from 0
                from_who, to_who, bond_type = int(edge_data[0]) - 1, int(edge_data[1]) - 1, int(edge_data[2])
                assert bond_type != 0
                if A < bond_type:
                    A = bond_type

                bond_type -= 1  # arc labels start from 0 to A-1

                adjacency_lists_graph[from_who].append((to_who, bond_type))
                adjacency_lists_graph[to_who].append((from_who, bond_type))

            if task_name == 'MUTAG':
                f.readline()
            m_end = f.readline()
            dollars = f.readline()

            if target[c] == 0:
                graphs_0.append((X_graph, Y_graph, adjacency_lists_graph, size))
            else:
                graphs_1.append((X_graph, Y_graph, adjacency_lists_graph, size))
            c += 1

print("Parsed ", c, " structures")
# ---------------------------------------------------------------- #

return_graphs_0 = []
return_graphs_1 = []
# ------------ Convert symbols to categorical inputs ------------ #
seen = []
for graph in graphs_0:
    for x in graph[0]:
        if x not in seen:
            seen.append(x)
for graph in graphs_1:
    for x in graph[0]:
        if x not in seen:
            seen.append(x)
map = {}
for i in range(0, len(seen)):
    map[seen[i]] = i

for graph in graphs_0:
    new_X = np.array([map[x] for x in graph[0]])
    return_graphs_0.append((new_X, graph[1], graph[2], graph[3]))
for graph in graphs_1:
    new_X = np.array([map[x] for x in graph[0]])
    return_graphs_1.append((new_X, graph[1], graph[2], graph[3]))
# ---------------------------------------------------------------- #

M = len(seen)

with open(task_name + '_datasetPerClass', 'wb') as f:
    pickle.dump([return_graphs_0, return_graphs_1, A, M], f)
