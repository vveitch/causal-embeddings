This module contains code to train a relational ERM predictor,
and then use this predictor to compute expected values of treatments and conditional outcomes.

This software builds on relational ERM https://github.com/wooden-spoon/relational-ERM

We include pre-processed Pokec network data for convenience.


# Requirements
1. Python 3.6 with numpy and pandas
2. Tensorflow *1.11*
3. gcc


# Setup
Run the following command in relational_ERM to build the graph samplers:

python setup.py build_ext --inplace


# Reproducing the experiments
The default settings for the code match the settings used in the software.
These match the default settings used by relational ERM.

You'll run the code as 
`./submit_scripts/run_classifier.sh`

The treatment and outcome simulation setting is controlled by
`--simulation_setting A` 
Possible values are A: Linear, B: Trig., C: High Var., and D: t-noise

The output of the run is ../../output/pokec_prediction/settingA/test_results.tsv

# Misc.
The experiments in the paper initialize from node embeddings that were pre-trained using a purely unsupervised objective.
To recreate the initalization embeddings, run `run_unsupervised.sh`. Then, uncomment `--init_checkpoint=$INIT_FILE` in `run_classifer.sh`

The code allows for more general simulation settings than we report in the paper, 
including not discretizing the covariates, and using a more complex model for the treatment simulation.
The conclusions from the experiments were the same under all experimental setting so we reported only the simplest one. 
