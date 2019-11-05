import argparse
from EndToEnd_Evaluation import main as endtoend
from Incremental_Evaluation import main as incremental

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='incremental')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--dataset-name', dest='dataset_name', default='none')
    parser.add_argument('--outer-folds', dest='outer_folds', default=10)
    parser.add_argument('--outer-processes', dest='outer_processes', default=2)
    parser.add_argument('--inner-folds', dest='inner_folds', default=5)
    parser.add_argument('--inner-processes', dest='inner_processes', default=1)
    parser.add_argument('--debug', action="store_true", dest='debug')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.dataset_name != 'none':
        datasets = [args.dataset_name]
    else:
        datasets = ['PPI']  # TODO refactor this

    config_file = args.config_file
    experiment = args.experiment

    if 'None' in args.outer_folds:
        outer_k = None
    else:
        outer_k = int(args.outer_folds)

    if 'None' in args.inner_folds:
        inner_k = None
    else:
        inner_k = int(args.inner_folds)

    for dataset_name in datasets:
        try:
            if experiment == 'incremental':
                incremental(config_file, dataset_name,
                            outer_k=outer_k, outer_processes=int(args.outer_processes),
                            inner_k=inner_k, inner_processes=int(args.inner_processes),
                            debug=args.debug)
        except Exception as e:
            raise e  # print(e)
