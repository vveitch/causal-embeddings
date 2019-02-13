This module contains code to train a BERT based text predictor and then use this predictor to compute expected values of treatments and conditional outcomes.

This software builds on
1. Bert: https://github.com/google-research/bert, and on
2. PeerRead: https://github.com/allenai/PeerRead

We include pre-processed PeerRead arxiv data for convenience.

# Requirements and setup
1. You'll need to download a pre-trained BERT model (following the above github link). We use `uncased_L-12_H-768_A-12`.
2. Install Tensorflow 1.12

# Reproducing the experiments
The default settings for the code match the settings used in the software.
These match the default settings used by BERT, except
1. we reduce batch size to allow training on a Titan X, and
2. we adjust the learning rate to account for this.

You'll run the code as 
`./submit_scripts/run_classifier.sh`
Before doing this, you'll need to edit `run_classifier.sh` to change 
`export BERT_BASE_DIR=../../BERT_pre-trained/uncased_L-12_H-768_A-12`
to
`export BERT_BASE_DIR=[path to BERT_pre-trained]/uncased_L-12_H-768_A-12`.

The flag 
`--treatment=theorem_referenced`
controls the experiment. 
Possible values are `buzzy_title`, `contains_appendix`, `theorem_referenced`, and `equation_referenced`.

The output of the run is ../../output/PeerRead/theorem_referenced/test_results.tsv

# Misc.

The experiments in the paper use a version of BERT that was further pre-trained on the PeerRead corpus
using an unsupervised objective.
This can be reproduced by running `run_unsupervised.sh`. This takes about 24 hours on a single Titan Xp.
After this, uncomment the `INIT_DIR` options in `run_classifier.sh`.

(This is really just a detail though.)