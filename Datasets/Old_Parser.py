import re

import numpy as np
import pickle

# THIS PARSER CAN PARSE MUTAG, CPDB, AIDS
task_name = 'AIDS'


# NOTE: this parser assumes MOL files (.mol) have been correctly generated using OpenBabel

# ----------------------- PARSE TARGETS -------------------------- #
target = []
with open(task_name + '/' + task_name + '_labels.txt') as f:
    for line in f.readlines():
        target.append(float(line))
target = np.array(target, dtype='int')
# ---------------------------------------------------------------- #

# ----------------------- PARSE DATASET -------------------------- #
graphs = []
A = 0

with open(task_name + '/' + task_name + '.mol') as f:
    end = False
    c = 0  # a counter of examples
    # Parsing a graph in the loop
    while not end:

        X_graph = []
        edge_graph = []
        Y_graph = []

        skip = f.readline()
        skip_id = f.readline()
        skip = f.readline()
        counts = f.readline().split()  # default delimiter is space

        # print(counts)

        if len(counts) == 0:
            end = True

        elif len(counts) == 1:
            string_to_split = counts[0]

            if len(string_to_split) == 5:
                counts = [string_to_split[0:2], string_to_split[2:5]]
            elif len(string_to_split) == 6:
                counts = [string_to_split[0:3], string_to_split[3:6]]

            # print('Node data to split: ', string_to_split, ' result: ', counts)

        if not end:
            no_atoms, no_edges = int(counts[0]), int(counts[1])
            # print(no_atoms, no_edges)
            # print(no_atoms, no_edges)

            size = no_atoms
            adjacency_lists_graph = [[] for _ in range(0, no_atoms)]
            edges = []
            edges_to_order = []

            for _ in range(0, no_atoms):
                l = f.readline()
                atom_symbol = l.split()[3]

                X_graph.append(atom_symbol)
                Y_graph.append([target[c]])

            for _ in range(0, no_edges):
                edge_data = f.readline().split()

                if len(edge_data) == 5:
                    # node ids shall start from 0
                    string_to_split, bond_type = edge_data[0], int(edge_data[1])
                    if len(string_to_split) == 5:
                        from_who, to_who = int(string_to_split[0:2]) - 1, int(string_to_split[2:5])-1
                    elif len(string_to_split) == 6:
                        from_who, to_who = int(string_to_split[0:3])-1, int(string_to_split[3:6])-1

                    # print('Edge data to split: ', string_to_split, ' result: ', [from_who, to_who])

                else:
                    # node ids shall start from 0
                    from_who, to_who, bond_type = int(edge_data[0]) - 1, int(edge_data[1]) - 1, int(edge_data[2])
                    # print(from_who, to_who, bond_type)
                assert bond_type != 0

                if A < bond_type:
                    A = bond_type

                bond_type -= 1  # arc labels start from 0 to A-1

                adjacency_lists_graph[from_who].append(to_who)
                adjacency_lists_graph[to_who].append(from_who)
                # THIS TUPLE ORDER IS IMPORTANT! PYTHON WILL ORDER FIRST BY DEST, THEN BY SOURCE (And also by value)!!!
                edges_to_order.append((to_who, from_who, bond_type))
                edges_to_order.append((from_who, to_who, bond_type))

            ordered_edges = sorted(edges_to_order)

            for dest, source, val in ordered_edges:
                edges.append(val)

            edge_graph.extend(edges)

            if task_name == 'AIDS':
                for _ in range(0, 6):
                    m_end = f.readline()
            else:
                m_end = f.readline()
                while re.search('END', m_end) == None:
                    m_end = f.readline()

            dollars = f.readline()

            c += 1

            max_ariety = 0
            for l in adjacency_lists_graph:
                ariety = len(l)
                if ariety > max_ariety:
                    max_ariety = ariety
            # print(f'max_ariety {max_ariety}')
            graphs.append((X_graph, edge_graph, Y_graph, adjacency_lists_graph, size, max_ariety))

print('Parsed ', c, ' structures')
# ---------------------------------------------------------------- #

return_graphs = []
# ------------ Convert symbols to categorical inputs ------------ #
seen = []
for graph in graphs:
    for x in graph[0]:
        if x not in seen:
            seen.append(x)
map = {}
for i in range(0, len(seen)):
    map[seen[i]] = i

for graph in graphs:
    new_X = np.array([map[x] for x in graph[0]])
    return_graphs.append((new_X, graph[1], graph[2], graph[3], graph[4], graph[5]))
# ---------------------------------------------------------------- #
K = len(seen)
print(K, A)
with open(task_name + '/' + task_name + '_dataset', 'wb') as f:
    pickle.dump([return_graphs, A, K], f)
