# Contextual Graph Markov Model (CGMM)

## Summary
CGMM is a generative approach to learning contexts in graphs. It combines information diffusion and local computation through the use of a deep architecture and stationarity assumptions. The model does NOT preprocess the graph into a fixed structure before learning. Instead, it works with graphs of any size and shape while retaining scalability. Experiments show that this model works well compared to expensive kernel methods that extensively analyse the entire input structure in order to extract relevant features. In contrast, CGMM extract more abstract features as the architecture is built (incrementally). 

We hope that the exploitation of the proposed framework, which can be extended in many directions, can contribute to the extensive use of both generative and discriminative approaches to the adaptive processing of structured data.

## This repo
The library includes data and scripts to reproduce the tree/graph classification experiments reported in the paper describing the method.

This research software is provided as is. If you happen to use or modify this code, please remember to cite the foundation papers:

[*Bacciu Davide, Errica Federico, Micheli Alessio: Contextual Graph Markov Model: A Deep and Generative Approach to Graph Processing. Proceedings of the 35th International Conference on Machine Learning. Vol. 80. pp. 294-303.*
](http://proceedings.mlr.press/v80/bacciu18a.html)

### 24/5/19 UPDATE
We provide an extended and refactored version of CGMM, implemented in Pytorch. There are additional experimental routines to try some common graph classification tasks.
For INEX tasks, please refer to the original numpy branch.
