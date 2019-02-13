This folder contains software included with "Using Embeddings to Correct for Unobserved Confounding".

Instructions for running the network and language examples are given in the respective subdirectories.

The three modules are \
*language*: a semi-supervised text classifier based on a BERT embedding model.\
*relational_ERM*: a semi-supervised vertex (graph) classifier using relational empirical minimization.\
*semi_parametric_estimation*: implementations of semi-parametric estimators used to compute final treatment effect estimates.

To reproduce the experiments:\
1. Run the relevant predictor, which will produce a file like `../output/language/my_experiment.tsv` containing predicted values for nuisance parameters.\
2. Run the relevant module in `semi_parametric_estimation` with `my_experiment.tsv` as input.
