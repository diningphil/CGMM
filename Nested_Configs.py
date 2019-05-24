possible_CN = [5, 10, 20]
possible_CA = [None]

# This should be investigated but could be left to [1] in our experiments (keep complexity low?)
possible_use_statistics = [[1]]  # np.arange(layers)+1

# This can be searched as well
possible_max_epochs = [10]

# ---------- Construct possible configurations ---------- #

# These are the keys tht my experiment will use
model_configurations = {
    'model_class': ['CGMM'],
    'max_layers': [20],
    'threshold': [0.],
    'CN': possible_CN,
    'CA': possible_CA,
    'max_epochs': possible_max_epochs,
    'use_statistics': possible_use_statistics,
    'add_self_arc': [False],
    'aggregation': ['sum'],

}

classifier_configurations = {
    'classifier': ['mlp'],
    'max_layers': [5, 10, 15, 20],  # max value must be less than the one used in Nested_CV_OnlyUnigrams
    'l2': [1e-2, 5e-2, 5e-3],
    'learning_rate': [1e-3, 1e-4],
    'l_batch_size': [100],
    'training_epochs': [20000],
    'early_stopping': [10000],
    # This is redundant for logistic!
    'hidden_units': [8, 16, 32, 128],
}


outer_folds = 10
inner_folds = 5
task_names = ['DD'] 
