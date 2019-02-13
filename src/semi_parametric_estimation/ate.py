"""
Logic:
1. load data from tsv
2. construct covariate
3. tmle
4. causal output
"""
import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize


def load_predictor_outputs(tsv_filename):
    pass


def truncate_by_g(ite, g, level=0.1):
    keep_these = np.logical_and(g > level, g < 1.-level)
    return ite[keep_these]


def _cross_entropy(y, p):
    return -np.mean((y*np.log(p) + (1.-y)*np.log(1.-p)))


def _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps):
    """
    Returns q_\eps (t,x)
    (i.e., value of perturbed predictor at t, eps, x; where q_t0, q_t1, g are all evaluated at x
    """
    h = t * (1./g) - (1.-t) / (1. - g)
    full_lq = (1.-t)*logit(q_t0) + t*logit(q_t1)  # logit predictions from unperturbed model
    logit_perturb = full_lq + eps * h
    return expit(logit_perturb)


def psi_tmle_bin_outcome(q_t0, q_t1, g, t, y, truncate_level=0.1):
    # solve the perturbation problem

    orig_g = np.copy(g)

    q_t0 = truncate_by_g(np.copy(q_t0), orig_g, truncate_level)
    q_t1 = truncate_by_g(np.copy(q_t1), orig_g, truncate_level)
    g = truncate_by_g(np.copy(g), orig_g, truncate_level)
    t = truncate_by_g(np.copy(t), orig_g, truncate_level)
    y = truncate_by_g(np.copy(y), orig_g, truncate_level)

    eps_hat = minimize(lambda eps: _cross_entropy(y, _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps))
                       , 0.)
    eps_hat = eps_hat.x[0]

    # # sanity check
    # this always ends up agreeing in practice
    # eps_hat2 = minimize(lambda eps: _cross_entropy(y, _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps))
    #                     , 0.1)
    # eps_hat2 = eps_hat2.x[0]
    # print("outputs of tmle epsilon estimation: \n")
    # print(eps_hat)
    # print(eps_hat2)

    def q1(t_cf):
        return _perturbed_model_bin_outcome(q_t0, q_t1, g, t_cf, eps_hat)

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    return np.mean(ite)


def _nonneg_scale(x):
    max = x.max()
    min = x.min()

    scale_x = (x - min) / (max - min)
    scale_x = np.clip(scale_x, 0.005, 1-0.005)
    return scale_x, max, min


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, truncate_level=0.1):
    q_t0 = truncate_by_g(q_t0, g, truncate_level)
    q_t1 = truncate_by_g(q_t1, g, truncate_level)
    t = truncate_by_g(t, g, truncate_level)
    y = truncate_by_g(y, g, truncate_level)
    g = truncate_by_g(g, g, truncate_level)

    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1  # predictions from unperturbed model

    eps_hat = np.sum(h*(y-full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_iptw(q_t0, q_t1, g, t, y, truncate_level=0.1):
    ite=(t / g - (1-t) / (1-g))*y
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_aiptw(q_t0, q_t1, g, t, y, truncate_level=0.1):
    full_q = q_t0 * (1 - t) + q_t1 * t
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    ite = h * (y - full_q) + q_t1 - q_t0

    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_very_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    return y[t == 1].mean() - y[t == 0].mean()


def main():
    pass

if __name__ == "__main__":
    main()