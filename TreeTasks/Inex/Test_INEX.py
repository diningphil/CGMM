from utils.CGMMUtilities import *
from TreeTasks.Inex.InexParser import bu_parse

INEX = '2005'
name = 'INEX_test'+INEX
undirected = False

C = 40
layer = 4
concatenate_fingerprints = True

svmC = 100
gamma = 5
unibigram = True

threshold = 0.
max_epochs = 15
Lprec = np.array([1], dtype='int')

if INEX == '2005':
    A = 32
    M = 366
    K = 11
    graphs_train = bu_parse("./inex05.train.elastic.tree", max_ariety=A,
                                      undirected=undirected)
    graphs_test = bu_parse("./inex05.test.elastic.tree", max_ariety=A,
                                     undirected=undirected)
elif INEX == '2006':
    A = 66
    M = 65
    K = 18
    graphs_train = bu_parse("./inex06.train.elastic.tree", max_ariety=A,
                                      undirected=undirected)
    graphs_test = bu_parse("./inex06.test.elastic.tree", max_ariety=A,
                                     undirected=undirected)

X_train, Y_train, adjacency_lists_train, sizes_train = unravel(graphs_train, one_target_per_graph=True)
X_test, Y_test, adjacency_lists_test, sizes_test = unravel(graphs_test, one_target_per_graph=True)

A = 2 * A  # see ModelSelection_INEX

# PERFORM TRAINING over the entire training set
runs = 1
for run in range(0, runs):
    architecture = None
    prevStats = None
    lastStates = None
    architecture, prevStats, lastStates = incremental_training(C, M, A, Lprec, adjacency_lists_train, X_train,
                                                               layers=layer,
                                                               threshold=threshold, max_epochs=max_epochs,
                                                               architecture=architecture,
                                                               prev_statistics=prevStats,
                                                               last_states=lastStates)

    unigram_train, allStates_train = compute_input_matrix(architecture, C, X_train, adjacency_lists_train, sizes_train,
                                                          concatenate=concatenate_fingerprints, return_all_states=True)
    unigram_test, allStates_test = compute_input_matrix(architecture, C, X_test, adjacency_lists_test, sizes_test,
                                                        concatenate=concatenate_fingerprints, return_all_states=True)

    with open("fingerprints/" + name + '_' + str(run) + ' ' + str(C) + ' ' + str(layer) + ' ' + str(Lprec)
                      + ' ' + str(concatenate_fingerprints), 'wb') as f:
        pickle.dump(
            [unigram_train, unigram_test, allStates_train, allStates_test, adjacency_lists_train, adjacency_lists_test,
             sizes_train, sizes_test, Y_train, Y_test], f)

useSVM = True
useJaccard = False

# OPEN A LOG FILE WHERE TO STORE RESULTS
logging.basicConfig(
    filename=name + '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.log',
    level=logging.DEBUG, filemode='a')

if useSVM:
    fingerprints_to_svm_accuracy(C, Lprec, layer, runs, svmC, gamma, concatenate_fingerprints, name)

if useJaccard:
    for unibigram in [False, True]:
        fingerprints_to_jaccard_accuracy(C, Lprec, layer, runs, unibigram, concatenate_fingerprints, name)