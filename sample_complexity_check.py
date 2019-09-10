"""
This script tests the result of Lemma 1 using toy constants
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
rcParams['mathtext.default'] = 'regular'
rcParams['font.size'] = 20
matplotlib.rcParams.update({'errorbar.capsize': 0})


# Global constants. The regularization lambda is not defined here, it is taken as parameter by the functions
gamma = 0.2    # discount factor
delta = 0.1    # confidence parameter
K = 2          # number of actions


def get_derived_constants(lambda_reg):
    """
    Computes the following constants:
    - kappa
    - beta (defined in Lemma 1)
    - alpha (defined in Proposition 1)
    - L = 1/lambda
    - M_lambda = lambda*log(K)

    :param lambda_reg: regularization parameter
    :return:
    """
    L = 1.0 / lambda_reg
    M_lambda = lambda_reg * np.log(K)
    kappa = (1 - np.sqrt(gamma)) / (K * L)

    # \beta(\delta) defined in Lemma 1
    beta = np.log(2 * K / delta) * L * 2 * np.power((3 + M_lambda)*K, 2)
    beta = beta/(np.power(1 - gamma, 4) * np.power(1 - np.sqrt(gamma), 3))

    # \alpha(\delta) defined in Proposition 1
    alpha = np.log(2 * K / delta)*2*K*np.power(3+M_lambda, 2)
    alpha = alpha / (np.power(1 - gamma, 4) * np.power(1 - np.sqrt(gamma), 2))

    return kappa, beta, alpha, L, M_lambda


def sample_complexity_sparse_sampling(epsilon, lambda_reg):
    """
    Computes the sample complexity of Sparse Sampling strategy for regularized problems. See Kearns et al., 1999.
    :param epsilon: accuracy
    :param lambda_reg: regularization parameter
    :return: the number of calls to the generative model required to achieve epsilon accuracy using the Sparse Sampling
             strategy.
    """
    kappa, beta, alpha, L, M_lambda = get_derived_constants(lambda_reg)

    aux = epsilon*(1-gamma)/(1+M_lambda)
    H_epsilon = 2*np.log(aux)/np.log(gamma) + 1

    n_samplev = np.power(gamma, 0.5*H_epsilon*(H_epsilon-1))*np.power(alpha/(epsilon**2.0), H_epsilon)
    return n_samplev


def simulate_sample_complexity(epsilon, lambda_reg):
    """
    Returns n_samplev(epsilon)
    :param epsilon: accuracy
    :param lambda_reg: regularization parameter
    :return: the number of calls made by the function sampleV()
    """
    kappa, beta, _, _, _ = get_derived_constants(lambda_reg)
    if epsilon >= kappa:
        return sample_complexity_sparse_sampling(epsilon, lambda_reg)
    else:
        return 1 + simulate_sample_complexity(epsilon/np.sqrt(gamma), lambda_reg) + \
               (beta/epsilon)*simulate_sample_complexity(np.sqrt(kappa*epsilon/gamma), lambda_reg)


def compute_theoretical_bound(epsilon, lambda_reg):
    """
    Returns theoretical bound on n_samplev(epsilon)
    """
    kappa, beta, alpha, L, M_lambda = get_derived_constants(lambda_reg)

    if epsilon >= kappa:
        return sample_complexity_sparse_sampling(epsilon, lambda_reg)

    else:
        eta1 = (kappa**2.0)*sample_complexity_sparse_sampling(kappa, lambda_reg)
        eta2 = np.log2(gamma * 2*beta / kappa) + np.log2(1.0 / (1.0 - gamma))
        aux = np.log(kappa / (epsilon * gamma)) / np.log(1.0 / gamma)
        bound = (eta1/(epsilon**2.0))*np.power(aux, eta2)
        return bound


# -------------------------------------------------------------------------
# Simulation for fixed lambda - Testing the theoretical bound
# -------------------------------------------------------------------------
lambd = 0.1
kappa, beta, alpha, L, M_lambda = get_derived_constants(lambd)
epsilon_array = np.logspace(-9, -1, 100)
nsamplev_array = [simulate_sample_complexity(epsilon, lambd) for epsilon in epsilon_array]
theoretical_bound = [compute_theoretical_bound(epsilon, lambd) for epsilon in epsilon_array]
theoretical_bound_unif = [sample_complexity_sparse_sampling(epsilon, lambd) for epsilon in epsilon_array]

# Plot
plt.figure()
plt.loglog(1.0/epsilon_array, nsamplev_array, '-.', label='SmoothCruiser (simulated)')
plt.loglog(1.0/epsilon_array, theoretical_bound, '-', label='SmoothCruiser (theoretical bound)')
plt.loglog(1.0/epsilon_array, theoretical_bound_unif, '--', label='Sparse Sampling')
plt.axvline(x=1/kappa, color='r', linestyle=':', label='$\epsilon = \kappa$')
plt.xlabel('inverse accuracy $1/\epsilon$')
plt.ylabel('$n_{\mathrm{sampleV}}(\epsilon, \delta)$')
# plt.title('Sample complexity simulation')
plt.legend(loc='upper right')
plt.draw()

# -------------------------------------------------------------------------
# Lambda versus sample complexity
# -------------------------------------------------------------------------
lambda_list = np.logspace(-4, 4, 100)

kappa_ref_list = []
for ii in range(len(lambda_list)):
    kappa_ref, _, _, _, _ = get_derived_constants(lambda_list[ii])
    kappa_ref_list.append(kappa_ref)

# epsilon_ref = np.array(kappa_ref_list)*0.25
relative_error = 1e-2
epsilon_ref = relative_error*(1+lambda_list*np.log(K))/(1-gamma)

sample_complexity_list = np.array([compute_theoretical_bound(epsilon_ref[ii], lambda_list[ii]) for ii in range(len(lambda_list))])
sim_sample_complexity_list = np.array([simulate_sample_complexity(epsilon_ref[ii], lambda_list[ii]) for ii in range(len(lambda_list))])
sample_complexity_unif_list = np.array([sample_complexity_sparse_sampling(epsilon_ref[ii], lambda_list[ii]) for ii in range(len(lambda_list))])


plt.figure()
plt.title('Samples required to achieve %.2f relative error' % relative_error)
plt.loglog(lambda_list, sim_sample_complexity_list, label='simulated')
plt.loglog(lambda_list, sample_complexity_list, '--', label='theoretical bound')
plt.xlabel('regularization $\lambda$')
plt.legend()
# plt.ylabel('sample complexity bound')

plt.figure()
plt.title('Ratio wrt Sparse Sampling for %.2f rel. error' % relative_error)
plt.loglog(lambda_list, sim_sample_complexity_list/sample_complexity_unif_list, label='simulated')
plt.loglog(lambda_list, sample_complexity_list/sample_complexity_unif_list, '--', label='theoretical bound')
plt.xlabel('regularization $\lambda$')
plt.legend()
# plt.ylabel('sample complexity bound')


# -------------------------------------------------------------------------
# Show plots
# -------------------------------------------------------------------------
plt.show()