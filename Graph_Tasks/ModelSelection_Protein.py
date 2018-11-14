from utils.CGMMUtilities import *

task_name = 'MUTAG'

with open(task_name + '_data/' + task_name + '_dataset', 'rb') as f:
    [graphs, A, K] = pickle.load(f)

    shuffle_dataset(graphs)

    # Use entire dataset
    graphs_train = graphs

max_workers = 4
folds = 3
Lprecs = [np.arange(1, dtype='int')+1]  # Lprec must start from 1
Cs = [20]
max_epochs = 5

layers = [1,2]

pool_dim = 3
svmCs = [5, 10, 20]
gammas = [5, 10, 15, 20]

logging.basicConfig(
    filename=str(task_name) + '_' + datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + '.log',
    level=logging.DEBUG, filemode='a')


useSVM_pool = True

# PHASE 1) COMPUTE FINGERPRINTS
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
    for C in Cs:
        for appendPrevVectors in [False, True]:
            for Lprec in Lprecs:
                pool.submit(k_fold_model_selection,
                            folds, pool_dim, svmCs, gammas, C, K, A, Lprec, graphs_train, layers,
                            0., max_epochs, appendPrevVectors, useSVM_pool, task_name)

    print("Submitted jobs")
    pool.shutdown(wait=True)

# PHASE 2) COMPUTE AVERAGE RESULTS FOR EACH CONFIGURATION
useSVM = True
useJaccard = False

# OPEN A LOG FILE WHERE TO STORE RESULTS
logging.basicConfig(
    filename=task_name + '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.log',
    level=logging.DEBUG, filemode='a')

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
    for C in Cs:
        for concatenate_fingerprints in [False, True]:
            for Lprec in Lprecs:
                for layer in layers:
                    if useSVM:
                        for svmC in svmCs:
                            for gamma in gammas:

                                pool.submit(fingerprints_to_svm_accuracy, C, Lprec, layer, folds, svmC, gamma,
                                            concatenate_fingerprints, task_name)

                    if useJaccard:
                        for unibigram in [False, True]:

                            pool.submit(fingerprints_to_jaccard_accuracy, C, Lprec, layer, folds, unibigram,
                                        concatenate_fingerprints, task_name)

    pool.shutdown(wait=True)
