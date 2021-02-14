# Contextual Graph Markov Model (CGMM)

## Summary
CGMM is a generative approach to learning contexts in graphs. It combines information diffusion and local computation through the use of a deep architecture and stationarity assumptions. The model does NOT preprocess the graph into a fixed structure before learning. Instead, it works with graphs of any size and shape while retaining scalability. Experiments show that this model works well compared to expensive kernel methods that extensively analyse the entire input structure in order to extract relevant features. In contrast, CGMM extract more abstract features as the architecture is built (incrementally). 

We hope that the exploitation of the proposed framework, which can be extended in many directions, can contribute to the extensive use of both generative and discriminative approaches to the adaptive processing of structured data.

## This repo
The library includes data and scripts to reproduce the tree/graph classification experiments reported in the paper describing the method.

This research software is provided as is. If you happen to use or modify this code, please remember to cite the foundation papers:

[*Bacciu Davide, Errica Federico, Micheli Alessio: Probabilistic Learning on Graphs via Contextual Architectures. Journal of Machine Learning Research, 21(134):1−39, 2020.*
](http://jmlr.org/papers/v21/19-470.html)

[*Bacciu Davide, Errica Federico, Micheli Alessio: Contextual Graph Markov Model: A Deep and Generative Approach to Graph Processing. Proceedings of the 35th International Conference on Machine Learning, 80:294-303, 2018.*
](http://proceedings.mlr.press/v80/bacciu18a.html)

### 27th of July 2020: Paper accepted at JMLR!
Please see the reference above.

### 3rd of March 2020 UPDATE

Thanks to the amazing work of [Daniele Atzeni](https://github.com/daniele-atzeni) we have dramatically increased the performance of bigram computation. With ``C=4``, continuous posteriors and matrix operations in place of nested for loops, we have been able to get a speedup of 900x (yes.. 900x) on NCI1 with a single core. Bravo Daniele!

### 5th of November 2019 UPDATE
We refactored the whole repository to allow for easy experimentation with incremental architectures. New efficiency improvements are coming soon. Stay tuned!

### 24th of May 2019 UPDATE
We provide an extended and refactored version of CGMM, implemented in Pytorch. There are additional experimental routines to try some common graph classification tasks. Please refer to the "Paper Version" Release tag for the original code of the paper.

### Create Data Sets

We first need to create a data set. Let's try to parse NCI1
`python PrepareDatasets.py DATA --dataset-name NCI1`
In the config file, specify node_type "discrete", as features are represented as atom types

For social datasets such as IMDB-MULTI:
`python PrepareDatasets.py DATA --dataset-name IMDB-BINARY --use-degree`
In the config file, specify node_type "continuous", as the degree should be treated as a continuous value

### Replicate Experiments

To replicate our experiments on graph classification, first modify the *config_CGMM.yml* file accordingly (use CGMM as model), then execute:
`python Launch_Experiments.py --config-file config_CGMM.yml --inner-folds None --outer-folds 10 --inner-processes [processes to use for internal cross validation] --outer-processes [processes to use for external cross validation] --dataset [DATASET STRING]`

By default, datasets are created to implement external 10-fold CV for model assessment, i.e. random splits between train and TEST, and an internal hold-out split of the training set (10% as VALIDATION set for model selection). If you change the number of data splits, you have to modify the **--inner-folds** and **--outer-folds** arguments accordingly. NOTE: a hold-out technique is associate to **--inner(outer)-folds = None**. Reproducibility is not hampered by different random splits in our case.

For node classification on PPI, use CGMMPPI in the config file instead of CGMM (to be refactored. In this case, you have to preprocess PPI before running on multiprocessing. You can do this by appending the `--debug` argument the very first time you try to train on PPI with CGMM).
