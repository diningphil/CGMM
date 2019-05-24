import json
import os
import glob
import numpy as np
from sklearn.svm import SVC, LinearSVC


def compute_accuracy(predictions, ground_truth):
    assert len(predictions) == len(ground_truth)
    return 100 * (np.sum(ground_truth == predictions) / len(predictions))


def compute_svm_accuracy(C, gamma, X, Y, X_test=None, Y_test=None):
    # SVC performs a AVA approach for multiclass classification
    if gamma is None:
        svm = LinearSVC(C=C, max_iter=1000)
    else:
        svm = SVC(C=C, kernel='rbf', gamma=gamma)

    # Train on train set
    svm.fit(X, Y)

    # Compute train accuracy
    tr_acc = compute_accuracy(svm.predict(X), Y)

    vl_acc = -1
    if X_test is not None and Y_test is not None:
        vl_acc = compute_accuracy(svm.predict(X_test), Y_test)

    return tr_acc, vl_acc, svm


def process_results(task_name, configs_folder):

    best_avg_vl_acc = 0.

    for i in range(0, 300):
        try:
            json_name = os.path.join(configs_folder, 'config_' + str(i), 'best_config.json')

            with open(json_name, 'r') as fp:
                results = json.load(fp)

            if best_avg_vl_acc < results['avg_vl_acc']:
                best_i = i
                best_avg_vl_acc = results['avg_vl_acc']
                best_config = results
        except Exception as e:
            pass
    print('Model selection winner for task', task_name, 'is config ', best_i, ':')
    for k in best_config.keys():
        print('\t', k, ':', best_config[k])

    return best_config
