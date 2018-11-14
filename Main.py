import pickle
from torch.utils.data import DataLoader
from TreeTasks.Inex.InexParser import bu_parse
from utils.CGMMUtilities import *
from utils.DatasetUtilities import unravel, split_dataset

'''
INEX = '2005'
task_name = 'INEX'+INEX

if INEX == '2005':
    A = 32
    undirected = False

    K = 366
    O = 11

    graphs = bu_parse("TreeTasks/Inex/inex05.train.elastic.tree", max_ariety=A, undirected=undirected)

elif INEX == '2006':
    A = 66
    undirected = False

    K = 65
    O = 18

    graphs = bu_parse("TreeTasks/Inex/inex06.train.elastic.tree", max_ariety=A, undirected=undirected)


A = 2*A  # Need to double edge labels because of how we performed parsing on INEX datasets


# ######### SPLIT SET INTO TRAIN and VALID SETS ######### #
train_perc = 0.8
valid_perc = 0.2
test_perc = 0.0
graphs_train, graphs_valid, graphs_test = \
    split_dataset(graphs, train_perc, valid_perc, test_perc, shuffle=True)

X_train, Y_train, adjacency_lists_train, sizes_train = unravel(graphs_train, one_target_per_graph=True)
X_valid, Y_valid, adjacency_lists_valid, sizes_valid = unravel(graphs_valid, one_target_per_graph=True)
X_test, Y_test, adjacency_lists_test, sizes_test = unravel(graphs_valid, one_target_per_graph=True)

# ############################################################################################ #
'''
task_name = 'AIDS'

with open('Graph_Tasks/' + task_name + '_data/' + task_name + '_dataset', 'rb') as f:
    [graphs, A, K] = pickle.load(f)
    graphs_train, graphs_valid, graphs_test = split_dataset(graphs, 0.9, 0.1, 0., shuffle=True)


X_train, Y_train, adjacency_lists_train, sizes_train = unravel(graphs_train, one_target_per_graph=True)
X_valid, Y_valid, adjacency_lists_valid, sizes_valid = unravel(graphs_valid, one_target_per_graph=True)
X_test, Y_test, adjacency_lists_test, sizes_test = unravel(graphs_test, one_target_per_graph=True)
# '''

label_dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train))  # It will return a tuple
label_dataset_valid = torch.utils.data.TensorDataset(torch.from_numpy(X_valid))  # It will return a tuple
label_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test))  # It will return a tuple

batch_size = 2000

# ---------- Hyper-Parameters ---------- #
# C and layers will be (possibly) be automatically determined
C = 10
layers = 4  # How many layers you will train

# This should be investigated but could be left to [1] in our experiments (keep complexity low?)
use_statistics = [1]  # np.arange(layers)+1

# This can be searched as well
max_epochs = 10

# 8 possible configurations of these hyperparams
infer_with_posterior = True  # Use the posterior to construct the fingerprints
bigram = False  # Use
radius = 1  # can be 1 or 2
add_self_arc = False  # Considers self at preceding layers (with a special recurrent arc)
# ---------------------------------------- #


exp_name = 'Experiments/' + task_name
save_name = 'checkpoint.pth'

# NOTE: PLEASE BE CAREFUL AND NOTICE IF YOU ARE SHUFFLING THE GRAPHS (in that case do not use saved unigrams)
# '''
# Training and inference phase
incremental_training(C, K, A, use_statistics, adjacency_lists_train, label_dataset_train, layers, radius, add_self_arc,
                     exp_name, threshold=0., max_epochs=max_epochs, save_name=save_name)


incremental_inference(A, layers, use_statistics, label_dataset_train, adjacency_lists_train, sizes_train,
                      infer_with_posterior, bigram, add_self_arc,
                      exp_name, tr_val_test='train', architecture=None, save_name=save_name)


incremental_inference(A, layers, use_statistics, label_dataset_valid, adjacency_lists_valid, sizes_valid,
                      infer_with_posterior, bigram, add_self_arc,
                      exp_name, tr_val_test='valid', architecture=None, save_name=save_name)

# '''
# You should take care of information about A and C for each layer
unigrams_train_dataset = load_unigrams_or_statistics(exp_name, 'unigrams_layer_train', 'unigrams',
                                             layers=[i for i in range(layers)])
# Concatenate the fingerprints (e.g. C=10, then [?, 10], [?,10] becomes [?, 20]
fingerprints_tr = concat_graph_fingerprints(unigrams_train_dataset)

# You should take care of information about A and C for each layer
unigrams_valid_dataset = load_unigrams_or_statistics(exp_name, 'unigrams_layer_valid', 'unigrams',
                                             layers=[i for i in range(layers)])
# Concatenate the fingerprints (e.g. C=10, then [?, 10], [?,10] becomes [?, 20]
fingerprints_val = concat_graph_fingerprints(unigrams_valid_dataset)

for svmC in [0.01, 0.1, 0.5, 1, 10, 20, 60, 80]:
    for gamma in [0.01, 0.5, 1, 5, 15, 30, 45, 80]:
        tr_acc, vl_acc = compute_svm_accuracy(fingerprints_tr, Y_train, svmC, gamma, fingerprints_val, Y_valid)
        print(svmC, gamma, tr_acc, vl_acc)
