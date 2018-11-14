from utils.CGMMUtilities import *
from utils.DatasetUtilities import *
from TreeTasks.Inex.InexParser import bu_parse

INEX = '2005'

name = 'INEX'+INEX

if INEX == '2005':
    A = 32
    undirected = False

    M = 366
    K = 11

    graphs = bu_parse("./inex05.train.elastic.tree", max_ariety=A, undirected=undirected)

elif INEX == '2006':
    A = 66
    undirected = False

    M = 65
    K = 18

    graphs = bu_parse("./inex06.train.elastic.tree", max_ariety=A, undirected=undirected)


A = 2*A  # Need to double edge labels because of how we performed parsing on INEX datasets


# ######### SPLIT TRAINING INTO TR E VL SETS ######### #
train_perc = 0.8
valid_perc = 0.2
test_perc = 0.0
graphs_train, graphs_valid, _ = \
    split_dataset(graphs, train_perc, valid_perc, test_perc, shuffle=False)

X_TR, Y_TR, adjacency_lists_TR, sizes_train = unravel(graphs_train, one_target_per_graph=True)
X_VL, Y_VL, adjacency_lists_VL, sizes_valid = unravel(graphs_valid, one_target_per_graph=True)

# ############################################################################################ #

# HYPER-PARAMS:
Lprecs = [np.array([1], dtype='int')]  #, np.array([1,2,3,4], dtype='int')]
svmCs = [5, 50, 100]
gammas = [5, 50, 100]
Cs = [20, 40]
max_epochs = 15
runs = 3
layers = [1, 2]
max_workers = 4


# THE HOLDOUT PROCESS IS DIVIDED IN 2 PHASES: 1) COMPUTE FINGERPRINTS (UNSUPERVISED); 2) COMPUTE ACCURACIES
# REQUIREMENTS: a folder in the current path named "fingerprints"

# NOTE: We have not implemented pooling nor incremental construction of the network.
#       These have been used with cross validation
# TODO Implement this holdout technique adding pooling and automatic construction

'''
# PHASE 1:
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
    # Store fingerprints for SVM/Jaccard/whatever
    for C in Cs:
        for concatenate_fingerprints in [False, True]:
            for Lprec in Lprecs:
                for run in range(0, runs):
                    # Stores fingerprints for subsequent SVM/Jaccard/whatever and as checkpoint
                    pool.submit(compute_fingerprints, C, M, A, Lprec, adjacency_lists_TR, X_TR, Y_TR, sizes_train,
                                adjacency_lists_VL, X_VL, Y_VL, sizes_valid,
                                layers, 0., max_epochs, run, concatenate_fingerprints, name)

    print("Submitted jobs")
    pool.shutdown(wait=True)

'''
# PHASE 2:
useSVM = True
useJaccard = False

# OPEN A LOG FILE WHERE TO STORE RESULTS
logging.basicConfig(
    filename=name + '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.log',
    level=logging.DEBUG, filemode='a')

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
    for C in Cs:
        for concatenate_fingerprints in [False, True]:
            for Lprec in Lprecs:
                for layer in layers:
                    if useSVM:
                        for svmC in svmCs:
                            for gamma in gammas:

                                pool.submit(fingerprints_to_svm_accuracy, C, Lprec, layer, runs, svmC, gamma,
                                            concatenate_fingerprints, name)

                                # FOR DEBUG PURPOSES
                                #fingerprints_to_svm_accuracy(C, Lprec, layer, runs, svmC, gamma,
                                #            concatenate_fingerprints, name)

                    if useJaccard:
                        for unibigram in [False, True]:

                            pool.submit(fingerprints_to_jaccard_accuracy, C, Lprec, layer, runs, unibigram,
                                        concatenate_fingerprints, name)

                            # FOR DEBUG PURPOSES
                            #fingerprints_to_jaccard_accuracy(C, A, Lprec, layer, runs, unibigram,
                            #            concatenate_fingerprints, name)

    pool.shutdown(wait=True)