import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from scipy.special import logsumexp, softmax
from envs import GridWorld, Chain, TwoRoomDense, TwoRoomSparse
from regularized_value_iteration import SmoothValueIterator
import multiprocessing


class SmoothCruiser:
    def __init__(self, env, gamma, eta):
        self.env = deepcopy(env)
        self.gamma = gamma
        self.eta = eta
        self.K = env_.action_space.n
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

        # self.count = 0  # for debug

    def F(self, state, q):
        eta = self.eta
        return logsumexp(q/eta)*eta

    def gradF(self, state, q):
        eta = self.eta
        return softmax(q/eta)

    def estimateQ(self, state, epsilon):
        return self.trueQ[state, :] + epsilon*np.random.uniform(-1.0, 1.0, size=self.K)
        # return self.trueQ[state, :] + epsilon*np.ones(self.K)

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

            # self.count += 1

            out = self.F(state, Q) - Q.dot(grad) + (r + self.gamma * v) * grad.sum()

            # tests
            assert np.abs(Q-self.trueQ[state]).max() <= np.sqrt(epsilon*self.epsilon_bar) + 1e-15

            return out


def get_error(args):
    smoothcruiser, state, target_acc = args
    v_estimate = smoothcruiser.sampleV(state, target_acc)
    error = v_estimate - smoothcruiser.trueV[state]
    return error, v_estimate


def check(env, gamma, eta, target_rel_error, N_sim):
    # SmoothCruiser
    sc = SmoothCruiser(env, gamma, eta)

    #
    print(sc.trueV)

    # target accuracy
    M_lambda = eta * np.log(env.action_space.n)
    target_acc = target_rel_error*(1.0+M_lambda)/(1-gamma)

    # Compute error mean and standard deviation (using joblib)
    njobs = max(1, multiprocessing.cpu_count() - 1)
    args = (sc, env.reset(), target_acc)
    arg_instances = [deepcopy(args) for ii in range(N_sim)]
    outputs = Parallel(n_jobs=njobs, backend="multiprocessing")(map(delayed(get_error), arg_instances))

    error, v_estimate = zip(*outputs)
    error = np.array(error)
    v_estimate = np.array(v_estimate)

    error_mean = error.mean()
    error_std = error.std()

    # Checking item (iii) of Lemma 2
    C_gamma = (3+2*M_lambda)/np.power(1.0-gamma, 2)

    print('Vmax = ', (1.0+M_lambda)/(1-gamma))
    print('target acc = %0.5E' % target_acc)
    print('error = (%0.5E +- %0.5E)' % (error_mean, error_std))
    print('')
    print("C_gamma = ", C_gamma)
    print("max of v_estimate = ", np.abs(v_estimate).max())

    assert np.abs(error_mean) <= target_acc, "accuracy error"
    assert np.abs(v_estimate).max() <= C_gamma, "bound error"

    return error, v_estimate, sc


if __name__ == '__main__':
    # Environment and parameters
    env_names = ['5x5 GridWorld', '10x10 GridWorld', '5 Chain', '10 Chain']
    env_list = []
    env_ = Chain(5)
    # env_ = TwoRoomSparse(10, 10, success_probability=0.75, enable_render=False)
    gamma_ = 0.5  # discount factor
    eta_ = 0.001  # regularization

    # Number of simulations
    N_sim = 200

    # Target relative error
    target_rel_error = 1e-3

    # Run check
    error, v_estimate, sc = check(env_, gamma_, eta_, target_rel_error, N_sim)
