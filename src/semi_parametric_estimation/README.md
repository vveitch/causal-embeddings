This module computes and prints the ATE estimates, given the outputs of the predictors.

Run `pokec_errors.py` for MSE of treatment effect estimate on simulated Pokec data.
Run `PeerRead_estimates.py` for ATE estimate of the treatment on PeerRead data.

Scripts should be run from the src directory as, e.g.,
`python -m semi_parametric_estimation/PeerRead_estimates`

Both scripts assume the output directory specified in the demo `run_classifer.py` scripts.
This can be changed editing the directory name in the main function of the scripts.