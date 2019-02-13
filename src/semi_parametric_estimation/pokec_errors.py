import os
import glob
import numpy as np
import pandas as pd

from .ate import psi_very_naive, psi_naive, psi_iptw, psi_aiptw, psi_tmle_cont_outcome

def rerm_psis_in_output_dir(output_dir):

    tsv_path = os.path.join(output_dir, 'test_results.tsv')
    data_path = os.path.join(output_dir, 'simulated_data.npz')

    output = pd.read_csv(tsv_path, '\t')
    data = np.load(data_path)
    outcomes = data['outcomes']
    treatments = data['treatments']
    y_0 = data['y_0']
    y_1 = data['y_1']
    t_prob = data['t_prob']
    t_latent = data['t_latent']
    y_latent = data['y_latent']

    ground_truth = y_1.mean()-y_0.mean()
    # print(ground_truth)

    # continuous response data output from model was scaled for training
    # for some stupid reason, I'm doing the descaling here instead of when the results are written
    # todo: fix me
    def _descale(x):
        return x*outcomes.std() + outcomes.mean()

    # y = _descale(_clean_input(output['outcome']))
    # t = _clean_input(output['treatment'])
    y = _descale(output['outcome'].values)
    t = output['treatment'].values
    q_t0 = _descale(output['expected_outcome_st_no_treatment'].values)
    q_t1 = _descale(output['expected_outcome_st_treatment'].values)
    g = output['treatment_probability'].values
    in_test = output['in_test'].values==1
    in_train = np.logical_not(in_test)

    psi_estimates = {}

    q_t0_test = q_t0[in_test]
    q_t1_test = q_t1[in_test]
    g_test = g[in_test]
    t_test = t[in_test]
    y_test = y[in_test]

    psi_estimates['very_naive'] = psi_very_naive(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.1)
    psi_estimates['naive'] = psi_naive(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.1)
    psi_estimates['iptw'] = psi_iptw(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.1)
    psi_estimates['aiptw'] = psi_aiptw(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.1)
    psi_estimates['tmle_cont_outcome'] = psi_tmle_cont_outcome(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.1)

    # print(psi_estimates)
    # q_t0_train = q_t0[in_train]
    # q_t1_train = q_t1[in_train]
    # g_train = g[in_train]
    # t_train = t[in_train]
    # y_train = y[in_train]
    #
    # psi_estimates['very_naive'] = psi_very_naive(q_t0_train, q_t1_train, g_train, t_train, y_train)
    # psi_estimates['naive'] = psi_naive(q_t0_train, q_t1_train, g_train, t_train, y_train)
    # psi_estimates['iptw'] = psi_iptw(q_t0_train, q_t1_train, g_train, t_train, y_train)
    # psi_estimates['aiptw'] = psi_aiptw(q_t0_train, q_t1_train, g_train, t_train, y_train)
    # psi_estimates['tmle_cont_outcome'] = psi_tmle_cont_outcome(q_t0_train, q_t1_train, g_train, t_train, y_train)
    #
    # print(psi_estimates)

    return psi_estimates, ground_truth


def rerm_psi_errors(setting_dir):
    output_dirs = sorted(glob.glob(os.path.join(setting_dir+'/*')))
    print(setting_dir)
    print(output_dirs.__len__())
    estimates = []
    for output_dir in output_dirs:
        # print("\n{}".format(output_dir))
        try:
            psi_estimates, ground_truth = rerm_psis_in_output_dir(output_dir)
            estimates += [(psi_estimates, ground_truth)]
        except:
            print('PSI computation failed... did you misspecify the output directory?')
            print(output_dir)

    avg_errors={}
    for k in psi_estimates.keys():
        k_errors = []
        for estimate, ground_truth in estimates:
            k_errors += [estimate[k]-ground_truth]

        errors = np.square(k_errors)
        avg_errors[k] = (np.round(np.mean(errors), 4),
                         np.round(np.std(errors)/np.sqrt(errors.shape[0]), 4))

    print(avg_errors)


def main():
    output_dir = '../output/pokec_prediction/'

    setting_base = os.path.join(output_dir, 'setting')
    rerm_psi_errors(setting_base + 'A')

    # if you ran training sweeping over all settings, this will sweep over the outputs
    # for setting in ['A', 'B', 'C', 'D', 'E']:
    #     print("*****************************************")
    #     print(setting)
    #     print("*****************************************")
    #
    #     rerm_psi_errors(setting_base + setting)


if __name__ == '__main__':
    main()