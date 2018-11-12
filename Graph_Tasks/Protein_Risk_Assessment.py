from utils.TrainingUtilities import *

task_name = 'MUTAG'

with open(task_name + '_data/' + task_name + '_dataset', 'rb') as f:
    [graphs, A, K] = pickle.load(f)
    shuffle_dataset(graphs)

folds = 3
inner_folds = 5
Lprec = np.array([1], dtype='int') #, np.array([1,2,3,4], dtype='int')]
svmCs = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100]
gammas = [1, 5, 10, 15, 20, 30, 40, 50, 80, 100]

layers = [2]
Cs = [20]

appends = [True]
unibigrams = [False, True]
max_epochs = 3
pool_dim = 3

risk_estimate = double_cross_validation(folds, inner_folds, pool_dim, Cs, svmCs, gammas, K, A, Lprec, graphs,
                        layers, appends, unibigrams, threshold=0., max_epochs=max_epochs, name=task_name)
