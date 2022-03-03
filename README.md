# Contextual Graph Markov Model (CGMM)

## Summary
CGMM is a generative approach to learning contexts in graphs. It combines information diffusion and local computation through the use of a deep architecture and stationarity assumptions. The model does NOT preprocess the graph into a fixed structure before learning. Instead, it works with graphs of any size and shape while retaining scalability. Experiments show that this model works well compared to expensive kernel methods that extensively analyse the entire input structure in order to extract relevant features. In contrast, CGMM extract more abstract features as the architecture is built (incrementally). 

We hope that the exploitation of the proposed framework, which can be extended in many directions, can contribute to the extensive use of both generative and discriminative approaches to the adaptive processing of structured data.

## This repo
The library includes data and scripts to reproduce the tree/graph classification experiments reported in the paper describing the method (**please look at previous releases**).

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

### Usage

This repo builds upon [PyDGN](https://github.com/diningphil/PyDGN), a framework to easily develop and test new DGNs.
See how to construct your dataset and then train your model there.

This repo assumes PyDGN 1.0.3 is used. Compatibility with future versions is not guaranteed.

The evaluation is carried out in two steps:
- Generate the unsupervised graph embeddings
- Apply a classifier on top

We designed two separate experiments to avoid recomputing the embeddings each time. First, use the `config_CGMM_Embedding.yml` config file to create the embeddings,
specifying the folder where to store them in the parameter `embeddings_folder`. Then, use the `config_CGMM_Classifier.yml` config file to launch
the classification experiments.

## Launch Exp:

#### Build dataset and data splits (follow PyDGN tutorial)
You can use the data splits we provided for graph classification tasks, taken from our [ICLR 2020](https://arxiv.org/abs/1912.09893) paper on reproducibility.

For instance:

    pydgn-dataset --config-file DATA_CONFIGS/config_PROTEINS.yml

#### Train the model

    pydgn-train  --config-file MODEL_CONFIGS/config_CGMM_Embedding.yml 
    pydgn-train  --config-file MODEL_CONFIGS/config_CGMM_Classifier.yml 
