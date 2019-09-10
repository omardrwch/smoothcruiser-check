import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from scipy.special import logsumexp, softmax
from envs import GridWorld, Chain, TwoRoomSparse
from regularized_value_iteration import SmoothValueIterator
import multiprocessing


np.random.seed(42)


class SmoothCruiser:
    """
    Simplified version of SmoothCruiser (see Appendix F)
    """
    def __init__(self, env, gamma, eta):
        self.env = deepcopy(env)
        self.gamma = gamma
        self.eta = eta
        self.K = env.action_space.n
        self.L = 1 / eta

        # Environment seed
        self.env.seed(np.random.randint(10000))

        # Computing constants
        self.epsilon_bar = (1.0 - np.sqrt(gamma)) / (self.K * self.L)  # reference precision

        # True Q function
        smooth_vi = SmoothValueIterator(self.env, eta, gamma=gamma)
        smooth_vi.run()
        self.trueQ = smooth_vi.Q
        self.trueV = smooth_vi.V

        self.count = 0  # for debug

    def F(self, state, q):
        eta = self.eta
        return logsumexp(q/eta)*eta

    def gradF(self, state, q):
        eta = self.eta
        return softmax(q/eta)

    def estimateQ(self, state, epsilon):
        return self.trueQ[state, :] + epsilon*np.random.uniform(-1.0, 1.0, size=self.K)

    def sampleV(self, state, epsilon):
        if epsilon >= (1.0+self.eta*np.log(self.K))/(1.0-self.gamma):
            return 0
        elif epsilon >= self.epsilon_bar:
            Q = self.estimateQ(state, epsilon)
            return self.F(state, Q)
        elif epsilon < self.epsilon_bar:
            Q = self.estimateQ(state, np.sqrt(epsilon*self.epsilon_bar))
            # compute probability
            grad = self.gradF(state, Q)
            prob = grad / grad.sum()
            # choose action
            action = np.random.choice(self.env.action_space.n, p=prob)
            # take step
            self.env.reset(state)
            z, r, done, _ = self.env.step(action)

            v = self.sampleV(z, epsilon/np.sqrt(self.gamma))

            self.count += 1

            out = self.F(state, Q) - Q.dot(grad) + (r + self.gamma * v) * grad.sum()

            # tests
            assert np.abs(Q-self.trueQ[state]).max() <= np.sqrt(epsilon*self.epsilon_bar) + 1e-15

            return out


def get_error(args):
    smoothcruiser, state, target_acc = args
    smoothcruiser.count = 0
    v_estimate = smoothcruiser.sampleV(state, target_acc)
    error = v_estimate - smoothcruiser.trueV[state]
    assert smoothcruiser.count > 1, "no calls to the generative model!"  # making sure SmoothCruiser is being used!
    # print(smoothcruiser.count)
    return error, v_estimate


def check(env, gamma, eta, target_accuracy, N_sim):
    # SmoothCruiser
    sc = SmoothCruiser(env, gamma, eta)
    # target accuracy
    M_lambda = eta * np.log(env.action_space.n)
    V_lambda_max = (1.0+M_lambda)/(1-gamma)

    # Compute error mean and standard deviation (using joblib)
    njobs = 1  # max(1, multiprocessing.cpu_count() - 1)
    args = (sc, env.reset(), target_accuracy)
    arg_instances = [deepcopy(args) for ii in range(N_sim)]
    outputs = Parallel(n_jobs=njobs, backend="threading")(map(delayed(get_error), arg_instances))

    error, v_estimate = zip(*outputs)
    error = np.array(error)
    v_estimate = np.array(v_estimate)

    error_mean = error.mean()
    error_std = error.std()

    # Checking item (iii) of Lemma 2
    C_gamma = (3+2*M_lambda)/np.power(1.0-gamma, 2)

    print('--- Vmax = ', V_lambda_max)
    print('--- target acc = %0.5E' % target_accuracy)
    print('--- error = (%0.5E +- %0.5E)' % (error_mean, error_std))
    print('')
    print("--- C_gamma = ", C_gamma)
    print("--- max of v_estimate = ", np.abs(v_estimate).max())
    print('')

    assert np.abs(error_mean) <= target_accuracy, "accuracy error"
    assert np.abs(v_estimate).max() <= C_gamma, "bound error"

    print("\n ... passed!")
    return error, v_estimate, sc


if __name__ == '__main__':
    # Environment and parameters
    env_names = ['5 Chain', '10 Chain', '5x5 GridWorld', '10x10 GridWorld']
    env_list = [Chain(5),
                Chain(10),
                TwoRoomSparse(5, 5, success_probability=0.75, enable_render=False),
                TwoRoomSparse(10, 10, success_probability=0.75, enable_render=False)]

    gamma_ = 0.2  # discount factor
    eta_ = 10  # regularization

    K_ref = 4  # largest number of actions among environments in env_list
    M_lambda_ref = eta_ * np.log(K_ref)
    C_gamma_ref = (3 + 2 * M_lambda_ref) / np.power(1.0 - gamma_, 2)  # constant in item (iii) of Lemma 2

    # Computing target accuracy (epsilon)
    kappa = eta_*(1.0 - np.sqrt(gamma_))/K_ref
    target_accuracy = kappa/4.0

    # Number of simulations required so that $\hat{\Delta}(s, \epsilon)$ (appendix F.2) is close to its mean
    # This uses Hoeffding's inequality and the fact that the estimates are bounded by C_gamma_ref
    confidence = 0.2
    N_sim = int((C_gamma_ref**2.0)/(2*(target_accuracy**2))*np.log(1/confidence))

    print('N_sim = ', N_sim)

    # Run check
    for ii, env_ in enumerate(env_list):
        print("-------------------")
        print(env_names[ii])
        error, v_estimate, sc = check(env_, gamma_, eta_, target_accuracy, N_sim)

