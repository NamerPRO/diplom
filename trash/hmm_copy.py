from typing import Dict, Final

import numpy as np

from acoustic_model.gmm import GMM
from acoustic_model.hmm_state import HMMState


class HMMGMM:

    __states_n: Final[int]

    def __init__(self, states: [HMMState], transition_probabilities = None):
        self.states = states
        self.__states_n = len(states)

        if transition_probabilities:
            if len(transition_probabilities) != self.__states_n:
                raise ValueError("transition_probabilities matrix must be N*N where N is number of states!")
            for line in transition_probabilities:
                if not np.isclose(sum(line), 1.0):
                    raise ValueError("transition_probabilities matrix consist of incorrect probabilities!")
        self.transition_probabilities = transition_probabilities or [[1. / self.__states_n for i in range(self.__states_n)] for i in range(self.__states_n)]

    def __alpha(o, a, b, start):
        """
        Computes probabilities of being in hidden state 'j'
        at time 't' given all the observations that have come
        before for each time 't' and each hidden state 'j'.

        Formally:
            α_t(j)=P(o_1, o_2, ..., o_t, q_t=j|λ) ∀t, j

        Args:
            o (numpy array): Observations.
            a (numpy array): Transition probabilities.
            b (numpy array): Emission probabilities.
            start (numpy array): Initial probabilities.

        Returns:
            2D numpy array: Computed α_t(j) for each 't' and 'j'.
        """
        T = o.shape[0]
        N = a.shape[0]

        dp = np.zeros((T, N), dtype=np.float64)

        for j in range(N):
            dp[0, j] = start[j] * b[j, o[0]]

        for t in range(1, T):
            for j in range(N):
                for i in range(N):
                    dp[t, j] += dp[t - 1, i] * a[i, j] * b[j, o[t]]

        return dp


    def beta(o, a, b):
        """
        Computes probabilities of being in hidden state 'j'
        at time 't' knowing future observations, but not
        knowing observations in the past for each time 't'
        and each hidden state 'j'.

        Formally:
            β_t(j)=P(o_{t+1}, o_{t+2}, ..., o_T|q_t=j, λ) ∀t, j

        Args:
            o (numpy array): Observations.
            a (numpy array): Transition probabilities.
            b (numpy array): Emission probabilities.

        Returns:
            2D numpy array: Computed β_t(j) for each 't' and 'j'.
        """
        T = o.shape[0]
        N = a.shape[0]

        dp = np.zeros((T, N), dtype=np.float64)

        for j in range(N):
            dp[-1, j] = 1

        for t in reversed(range(T - 1)):
            for j in range(N):
                for i in range(N):
                    dp[t, j] += dp[t + 1, i] * a[j][i] * b[i, o[t + 1]]

        return dp


    def gamma(alphas, betas):
        """
        Computes probabilities of being in hidden state 'j'
        at time 't' knowing observations that came before and
        observations that will come after for each time 't'
        and each hidden state 'j'.

        Formally:
            γ_t(j)=P(q_t=j|o, λ) ∀t, j

        Args:
            alphas (2D numpy array): data produced by 'alpha' method.
            betas (2D numpy array): data produced by 'beta' method.

        Returns:
            2D numpy array: Computed γ_t(j) for each 't' and 'j'.
        """
        T = alphas.shape[0]
        N = alphas.shape[1]

        gammas = np.zeros((T, N), dtype=np.float64)
        for t in range(T):
            denom_prob = np.float64(0)
            for j in range(N):
                denom_prob += alphas[t, j] * betas[t, j]
            for j in range(N):
                num_prob = alphas[t, j] * betas[t, j]
                gammas[t, j] = num_prob / denom_prob

        return gammas


    def ksi(alphas, betas, a, b, o):
        """
        For each 't', 'i', 'j' computes probability of
        being in hidden state 'i' at time 't' and in hidden
        state 'j' at time 't+1' given set of the observations.

        Formally:
            ξ_t(i,j)=P(q_t=i,q_{t+1}=j|o, λ) ∀t, i, j

        Args:
            alphas (2D numpy array): data produced by 'alpha' method.
            betas (2D numpy array): data produced by 'beta' method.
            a (numpy.ndarray): A 2D numpy array of shape '(N,N)' where N is the number
                of hidden states. Consists of transition probabilities.
            b (numpy.ndarray): A 2D nupy array of shape '(N,M)' where M is the number
                of observable states. Consists of emission probabilities.
            o (numpy.ndarray): A 1D numpy array of shape '(T,)' where T is the number
                of time steps. Consists of observations.

        Returns:
            3D numpy array: Computed ξ_t(i,j) for each 't', 'i' and 'j'.
        """
        T = o.shape[0]
        N = a.shape[0]

        ksis = np.zeros((T, N, N), dtype=np.float64)

        for t in range(T - 1):
            denom_prob = np.float64(0)
            for k in range(N):
                for w in range(N):
                    denom_prob += alphas[t, k] * a[k, w] * b[w, o[t + 1]] * betas[t + 1, w]
            for i in range(N):
                for j in range(N):
                    num_prob = alphas[t, i] * a[i, j] * b[j, o[t + 1]] * betas[t + 1, j]
                    ksis[t, i, j] = num_prob / denom_prob

        return ksis


    def forward(o, a, b, start):
        """
        Implementation of forward algorithm.

        Args:
            o (numpy.ndarray): A 1D numpy array of shape '(T,)' where T is the number
                of time steps. Consists of observations.
            a (numpy.ndarray): A 2D numpy array of shape '(N,N)' where N is the number
                of hidden states. Consists of transition probabilities.
            b (numpy.ndarray): A 2D nupy array of shape '(N,M)' where M is the number
                of observable states. Consists of emission probabilities.
            start (numpy.ndarray): A 1D numpy array of shape '(N,)'.
                Consists of Initial probabilities.

        Returns:
            np.float64: Computed probability.
        """
        probs = alpha(o, a, b, start)
        return np.sum(probs[-1, :])


    def backward(o, a, b, start):
        """
        Implementation of backward algorithm.

        Args:
            o (numpy.ndarray): A 1D numpy array of shape '(T,)' where T is the number
                of time steps. Consists of observations.
            a (numpy.ndarray): A 2D numpy array of shape '(N,N)' where N is the number
                of hidden states. Consists of transition probabilities.
            b (numpy.ndarray): A 2D nupy array of shape '(N,M)' where M is the number
                of observable states. Consists of emission probabilities.
            start (numpy.ndarray): A 1D numpy array of shape '(N,)'.
                Consists of Initial probabilities.

        Returns:
            np.float64: Computed probability.
        """
        probs = beta(o, a, b)

        total = np.float64(0)
        for i in range(a.shape[0]):
            total += probs[0, i] * start[i] * b[i, o[0]]

        return total


    def viterbi(o, a, b, start):
        """
        Implementation of Viterbi algorithm.

        Args:
            o (numpy.ndarray): A 1D numpy array of shape '(T,)' where T is the number
                of time steps. Consists of observations.
            a (numpy.ndarray): A 2D numpy array of shape '(N,N)' where N is the number
                of hidden states. Consists of transition probabilities.
            b (numpy.ndarray): A 2D nupy array of shape '(N,M)' where M is the number
                of observable states. Consists of emission probabilities.
            start (numpy.ndarray): A 1D numpy array of shape '(N,)'.
                Consists of Initial probabilities.

        Returns:
            Tuple[list, numpy.float64]: A tuple containing:
                - Most probable sequence of hidden states.
                - Probability of this sequence.
        """
        T = o.shape[0]
        N = a.shape[0]

        dp = np.zeros((T, N), dtype=np.float64)
        q = np.full((T, N), -1, dtype=int)

        for j in range(N):
            dp[0, j] = start[j] * b[j, o[0]]

        for t in range(1, T):
            for j in range(N):
                for i in range(N):
                    end_in_j_prob = dp[t - 1, i] * a[i, j] * b[j, o[t]]
                    if dp[t, j] < end_in_j_prob:
                        dp[t, j] = end_in_j_prob
                        q[t, j] = i

        max_prob = -np.inf
        max_prob_i = 0
        for i in range(N):
            if dp[-1, i] > max_prob:
                max_prob = dp[-1, i]
                max_prob_i = i

        answ, time = [max_prob_i], -1
        while q[time][max_prob_i] != -1:
            answ.append(q[time][max_prob_i])
            max_prob_i = q[time][max_prob_i]
            time -= 1

        return reversed(answ), max_prob


    def log_likelihood(x):
        """
        Computes log likelihood.

        Args:
            x (numpy.ndarray): A 2D numpy array for which to compute log likelihood.

        Returns:
            float: Log likelihood.
        """
        return np.sum(np.log(np.sum(x[-1, :])))


    def baum_welch(o, a, b, start, eps=0.00001, max_iters=1000, ignore_eps=False):
        """
        Implementation of Baum-Welch algorithm.

        Args:
            o (numpy.ndarray): A 1D numpy array of shape '(T,)' where T is the number
                of time steps. Consists of observations.
            a (numpy.ndarray): A 2D numpy array of shape '(N,N)' where N is the number
                of hidden states. Consists of transition probabilities.
            b (numpy.ndarray): A 2D nupy array of shape '(N,M)' where M is the number
                of observable states. Consists of emission probabilities.
            start (numpy.ndarray): A 1D numpy array of shape '(N,)'.
                Consists of Initial probabilities.
            eps (float): Epsilon parameter. Default is 0.00001.
            max_iters (int): Maximum number of iterations Baum-Welch algorithm is
                limited with. Defaults to 1000.
            ignore_eps (bool): If set to True will do 'max_iters' iterations
                ignoring 'eps' parameter. Defaults to False.

        Return:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing:
                - The updated initial probabilities.
                - The updated transition probabilities.
                - The updated emission probabilities.

        Example:
            >>> a = np.array([[0.5, 0.5],
            >>>               [0.3, 0.7]])
            >>> b = np.array([[0.3, 0.7],
            >>>               [0.8, 0.2]])
            >>> start = np.array([0.2, 0.8])
            >>> o = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
            >>> baum_welch(o, a, b, start)
            (array([7.92078023e-137, 1.00000000e+000]),
             array([[0.50002742, 0.49997258],
                    [0.14287281, 0.85712719]]),
            array([[1.21403560e-04, 9.99878596e-01],
                   [1.00000000e+00, 6.28149646e-24]]))
        """
        T = o.shape[0]
        N = a.shape[0]

        prev_log_likelihood = 0.0
        for it in range(max_iters):
            alphas = alpha(o, a, b, start)
            betas = beta(o, a, b)
            gammas = gamma(alphas, betas)
            ksis = ksi(alphas, betas, a, b, o)

            for i in range(N):
                start[i] = gammas[0, i]

            for i in range(N):
                denom_gamma = np.sum(gammas[:-1, i])
                for j in range(N):
                    a[i, j] = np.sum(ksis[:-1, i, j]) / denom_gamma

            for i in range(N):
                for vk in o:
                    num_prob = np.float64(0)
                    denom_prob = np.float64(0)
                    for t in range(T):
                        if o[t] == vk:
                            num_prob += gammas[t, i]
                        denom_prob += gammas[t, i]
                    b[i, vk] = num_prob / denom_prob

            cur_log_likelihood = log_likelihood(alphas)
            if not ignore_eps and it != 1 and abs(cur_log_likelihood - prev_log_likelihood) < eps:
                break
            prev_log_likelihood = cur_log_likelihood

        return start, a, b


if __name__ == '__main__':
    a = np.array([
        [0.1, 0.3, 0.1, 0.2, 0.3],
        [0.3, 0.1, 0.1, 0.1, 0.4],
        [0.1, 0.1, 0.1, 0.4, 0.3],
        [0.3, 0.3, 0.2, 0.2, 0.0],
        [0.0, 0.1, 0.5, 0.2, 0.2]
    ])
    b = np.array([
        [0.3, 0.1, 0.2, 0.3, 0.1],
        [0.0, 0.1, 0.1, 0.3, 0.5],
        [0.1, 0.2, 0.3, 0.2, 0.2],
        [0.3, 0.3, 0.1, 0.2, 0.1],
        [0.1, 0.3, 0.2, 0.3, 0.1]
    ])
    start = np.array([0.2, 0.1, 0.1, 0.2, 0.4])
    o = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 2])

    new_start, new_a, new_b = baum_welch(o, a, b, start, max_iters=10000, ignore_eps=True)

    print(new_start)
    print(new_a)
    print(new_b)
