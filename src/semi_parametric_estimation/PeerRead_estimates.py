import os
import glob
import numpy as np
import pandas as pd

from .ate import psi_very_naive, psi_naive, psi_iptw, psi_aiptw, psi_tmle_cont_outcome, psi_tmle_bin_outcome


def bert_psis_from_tsv(tsv_path):

    output = pd.read_csv(tsv_path, '\t')

    y = output['outcome'].values
    t = output['treatment'].values
    q_t0 = output['expected_outcome_st_no_treatment'].values
    q_t1 = output['expected_outcome_st_treatment'].values
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
    psi_estimates['tmle_bin_outcome'] = psi_tmle_bin_outcome(q_t0_test, q_t1_test, g_test, t_test, y_test, truncate_level=0.1)

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

    return psi_estimates


def bert_psi(output_dir):
    """
    Output_dir should contain one folder per data split, where each folder contains a tsv
    with the predictions from the Bert training
    """

    data_files = sorted(glob.glob('{}/*/*.tsv'.format(output_dir)))
    estimates = []
    for data_file in data_files:
        try:
            psi_estimates = bert_psis_from_tsv(data_file)
            # print(psi_estimates)
            estimates += [psi_estimates]
        except:
            print('wtf')
            print(data_file)

    avg_estimates={}
    for k in psi_estimates.keys():
        k_estimates = []
        for estimate in estimates:
            k_estimates += [estimate[k]]

        avg_estimates[k] = (np.round(np.mean(k_estimates), 4),
                            np.round(np.std(k_estimates) / np.sqrt(len(k_estimates)), 4))

    print(avg_estimates)


def main():
    output_dir = '../output/PeerRead/theorem_referenced/'
    print("output_dir: {}".format(output_dir))
    bert_psi(output_dir)


if __name__ == '__main__':
    main()