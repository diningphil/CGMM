import concurrent
import json
import os
import pickle
from multiprocessing import Lock

import torch
import numpy as np

from experiments.Graph_Assessment_Experiment import GraphAssessmentExperiment
from experiments.Graph_Classifier_Experiment import GraphClassifierExperiment
# C and layers will be (possibly) be automatically determined
from utils.utils import generate_grid

# ---------- Hyper-Parameters ---------- #
from Nested_Configs import model_configurations, classifier_configurations, outer_folds, inner_folds, task_names

all_configs = {}
all_configs.update(model_configurations)
all_configs.update(classifier_configurations)

# ---------------------------------------#

# Needed to avoid thread spawning
torch.set_num_threads(1)

for task_name in task_names:
    
    exp_path = os.path.join('Results', model_configurations['model_class'][0], task_name + '_assessmentUnigrams')
    nested_path = os.path.join(exp_path, str(outer_folds) + '_NESTED_CV')

    results_obj = [{} for _ in range(outer_folds)]
    # Per ogni outer fold
    for out_k in range(outer_folds):
        results_obj[out_k] = {'TR_ACC': 0.0, 'TE_ACC': 0.0, 'best_config': None,
                              'kfold': [{} for _ in range(inner_folds)]}
        # Per ogni inner fold
        for inn_k in range(inner_folds):
            # Costruisci le possibili configurazioni
            for layers in classifier_configurations['max_layers']:
                for unibigram in [False, True]:
                    for infer_with_posterior in [False, True]:
                        for model_config in generate_grid(all_configs):
                            res = results_obj[out_k]['kfold'][inn_k]

                            model_class = model_config['model_class']
                            threshold = model_config['threshold']
                            CN = model_config['CN']
                            CA = model_config['CA']
                            max_epochs = model_config['max_epochs']
                            use_statistics = model_config['use_statistics']
                            add_self_arc = model_config['add_self_arc']
                            aggregation = model_config['aggregation']

                            classifier = model_config['classifier']
                            l2 = model_config['l2']
                            learning_rate = model_config['learning_rate']
                            l_batch_size = model_config['l_batch_size']
                            training_epochs = model_config['training_epochs']
                            early_stopping = model_config['early_stopping']
                            hidden_units = model_config['hidden_units']

                            for key in [model_class, layers, threshold, unibigram, infer_with_posterior,
                                        CN, CA, max_epochs, use_statistics,
                                        add_self_arc, aggregation, classifier, l2, learning_rate, l_batch_size,
                                        training_epochs, early_stopping, hidden_units]:
                                if str(key) not in res:
                                    res[str(key)] = {}
                                res = res[str(key)]

                            res['TR_ACC'] = {}
                            res['VL_ACC'] = {}

    # print(results_obj)

    # Write results
    if not os.path.exists(os.path.join(nested_path, 'classifier_results.json')):
        with open(os.path.join(nested_path, 'classifier_results.json'), 'w') as f:
            json.dump(results_obj, f)
    else:
        print("Not overwriting already present json file, do you need to do smth about it?")

    lock = Lock()

    def execute_classifier(out_k, curr_config, kfold_path):
        try:
            model_class = str(curr_config['model_class'])
            max_layers = str(curr_config['max_layers'])
            threshold = str(curr_config['threshold'])
            CN = str(curr_config['CN'])
            CA = str(curr_config['CA'])
            max_epochs = str(curr_config['max_epochs'])
            use_statistics = str(curr_config['use_statistics'])
            add_self_arc = str(curr_config['add_self_arc'])
            aggregation = str(curr_config['aggregation'])

            classifier = str(curr_config['classifier'])
            l2 = str(curr_config['l2'])
            learning_rate = str(curr_config['learning_rate'])
            l_batch_size = str(curr_config['l_batch_size'])
            training_epochs = str(curr_config['training_epochs'])
            early_stopping = str(curr_config['early_stopping'])
            hidden_units = str(curr_config['hidden_units'])

            unibigram =  str(curr_config['unibigram'])
            infer_with_posterior =  str(curr_config['infer_with_posterior'])

            val_results = []

            lock.acquire()
            with open(os.path.join(nested_path, 'classifier_results.json'), 'r') as f:
                results_obj = json.load(f)
            lock.release()

            for inn_k in range(inner_folds):
                curr_exp_path = os.path.join(config_path, 'FOLD_' + str(inn_k+1))
                #kfold_split = pickle.load(open(os.path.join(kfold_path, 'inner_split_' + str(inn_k+1)), 'rb'))

                if results_obj[out_k]['kfold'][inn_k][model_class][max_layers][threshold][unibigram][infer_with_posterior][
                    CN][CA][max_epochs][use_statistics][
                    add_self_arc][aggregation][classifier][l2][learning_rate][l_batch_size][training_epochs][
                    early_stopping][hidden_units]['TR_ACC'] != {}:

                    # AVOID RECOMPUTING THINGS, RESUME COMPUTATION FROM WHERE IT WAS LEFT
                    print("Avoiding recomputing things for outer fold", out_k, results_obj[out_k]['kfold'][inn_k][model_class][max_layers][threshold][unibigram][infer_with_posterior][
                    CN][CA][max_epochs][use_statistics][
                    add_self_arc][aggregation][classifier][l2][learning_rate][l_batch_size][training_epochs][
                    early_stopping][hidden_units]['VL_ACC'])


                    val_results.append(results_obj[out_k]['kfold'][inn_k][model_class][max_layers][threshold][unibigram][infer_with_posterior][
                        CN][CA][max_epochs][use_statistics][
                        add_self_arc][aggregation][classifier][l2][learning_rate][l_batch_size][training_epochs][
                        early_stopping][hidden_units]['VL_ACC'])

                else:
                    experiment = GraphClassifierExperiment(curr_config, curr_exp_path)
                    train, valid = list(range(1000)), list(range(100))# fake splits
                    tr_acc, vl_acc = experiment.run_valid(train, valid)

                    lock.acquire()

                    with open(os.path.join(nested_path, 'classifier_results.json'), 'r') as f:
                        results_obj = json.load(f)

                    results_obj[out_k]['kfold'][inn_k][model_class][max_layers][threshold][unibigram][infer_with_posterior][
                        CN][CA][max_epochs][use_statistics][
                        add_self_arc][aggregation][classifier][l2][learning_rate][l_batch_size][training_epochs][
                        early_stopping][hidden_units]['TR_ACC'] = tr_acc
                    results_obj[out_k]['kfold'][inn_k][model_class][max_layers][threshold][unibigram][infer_with_posterior][
                        CN][CA][max_epochs][use_statistics][
                        add_self_arc][aggregation][classifier][l2][learning_rate][l_batch_size][training_epochs][
                        early_stopping][hidden_units]['VL_ACC'] = vl_acc

                    with open(os.path.join(nested_path, 'classifier_results.json'), 'w') as f:
                        json.dump(results_obj, f)

                    lock.release()

                    val_results.append(vl_acc)

            val_results = np.array(val_results)

            lock.acquire()

            if os.path.exists(os.path.join(kfold_path, 'best_config')):
                with open(os.path.join(kfold_path, 'best_config'), 'rb') as f:
                    best_val, best_config = pickle.load(f)
            else:
                best_val = 0.0
                best_config = {}

            if val_results.mean() > best_val:
                best_config['config'] = curr_config
                print('New best conf with VL', val_results.mean(), 'best was', best_val)
                best_val = val_results.mean()

                with open(os.path.join(kfold_path, 'best_config'), 'wb') as f:
                    pickle.dump([best_val, best_config], f)

            lock.release()
        except Exception as e:
            print(e)

    # For each outer fold
    #outer_splits = pickle.load(open(os.path.join(nested_path, 'outer_splits'), 'rb'))

    
    no_processes = 120
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=no_processes)
        
    for out_k in range(outer_folds):
        outer_path = exp_path = os.path.join(nested_path, 'OUTER_FOLD_' + str(out_k+1))
        kfold_path = os.path.join(outer_path, str(inner_folds) + '_FOLD_MS')

        best_config = {}
        best_val = [0.0]

        # Per ogni config (itera sulle cartelle e prendi la config associata)
        for root, dirs, files in os.walk(kfold_path):
            for _dir in dirs:
                if 'config' in _dir:
                    config_path = os.path.join(kfold_path, _dir)
                    # Per ogni unigram type
                    curr_config = json.load(open(os.path.join(config_path, 'config_results.json'), 'r'))
                    curr_config = curr_config['config']
                    curr_config['plot'] = False

                    for unibigram in [False, True]:
                        curr_config['unibigram'] = unibigram
                        for infer_with_posterior in [False, True]:
                            curr_config['infer_with_posterior'] = infer_with_posterior
                            for classifier_config in generate_grid(classifier_configurations):
                                curr_config.update(classifier_config)

                                pool.submit(execute_classifier, out_k, dict(curr_config), kfold_path)
    
    try:
        pool.shutdown()
    except Exception as e:
        print(e)
    
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=outer_folds)


    def execute(out_k):

        outer_path = exp_path = os.path.join(nested_path, 'OUTER_FOLD_' + str(out_k + 1))
        outer_split = pickle.load(open(os.path.join(nested_path, 'outer_split_' + str(out_k + 1)), 'rb'))
        kfold_path = os.path.join(outer_path, str(inner_folds) + '_FOLD_MS')
        # kfold_splits = pickle.load(open(os.path.join(kfold_path, 'inner_splits'), 'rb'))

        with open(os.path.join(kfold_path, 'best_config'), 'rb') as f:
            best_val, best_config = pickle.load(f)
            print(out_k, best_val, best_config)
        
        final_tr, final_te = [], []
        for run in range(3):
            experiment = GraphAssessmentExperiment(best_config['config'], outer_path)
            train, test = outer_split
            tr_acc, te_acc = experiment.run_test(train, test)

            final_tr.append(tr_acc)
            final_te.append(te_acc)

        final_tr = np.array(final_tr)
        final_te = np.array(final_te)

        lock.acquire()

        with open(os.path.join(nested_path, 'classifier_results.json'), 'r') as f:
            results_obj = json.load(f)

        results_obj[out_k]['best_config'] = best_config['config']
        results_obj[out_k]['TR_ACC'] = float(final_tr.mean())
        results_obj[out_k]['TE_ACC'] = float(final_te.mean())

        print(out_k, 'TE_ACC', results_obj[out_k]['TE_ACC'])
        # Write results
        with open(os.path.join(nested_path, 'classifier_results.json'), 'w') as f:
            json.dump(results_obj, f)
        lock.release()
	
    try:
        for out_k in range(outer_folds):
            pool.submit(execute, out_k)
            #execute(out_k)
    except Exception as e:
        print(e)
