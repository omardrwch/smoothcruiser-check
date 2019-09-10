"""
Implements value iteration for entropy regularized MDPs
"""

import numpy as np
import warnings
from scipy.special import logsumexp


class SmoothValueIterator:
    def __init__(self, env, reg, gamma=0.95):
        self.env = env
        self.reg = reg    # regularization constant
        self.gamma = gamma
        self.Q = None
        self.V = None

    def run(self, val_it_tol=1e-8, val_it_max_it=1e6):
        V = np.zeros(self.env.observation_space.n)
        it = 1
        while True:
            TV, Q, err = self.value_iteration_step(V)

            if it > val_it_max_it:
                warnings.warn("Value iteration: Maximum number of iterations exceeded.")
            if err < val_it_tol or it > val_it_max_it:
                self.Q = Q
                self.V = TV
                return TV
            V = TV
            it += 1

    def soft_bellman_operator(self, V):
        Ns = self.env.observation_space.n
        Na = self.env.action_space.n
        lambd = self.reg

        Q = np.zeros((Ns, Na))
        TV = np.zeros(Ns)

        for s in self.env.states:
            for a in self.env.available_actions(s):
                prob = self.env.P[s, a, :]
                rewards = np.array([self.env.reward_fn(s, a, s_) for s_ in self.env.states])
                Q[s, a] = np.sum(prob * (rewards + self.gamma * V))
            TV[s] = logsumexp(Q[s, self.env.available_actions(s)]/lambd)*lambd
        return TV, Q

    def value_iteration_step(self, V):
        TV, Q = self.soft_bellman_operator(V)
        err = np.abs(TV - V).max()
        return TV, Q, err

