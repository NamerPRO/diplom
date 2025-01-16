from typing import List

import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn import datasets

from acoustic_model.gmm import GMM
from acoustic_model.hmm_state import HMMState
import utility.log_math as lmath


class HMM:

    def __init__(self, states: List[HMMState], transition_probabilities = None):
        self.__eps = 1e-7
        self.__states = states
        self.__states_n = len(states)

        if any(states[i].initial_probability > self.__eps for i in range(self.__states_n)):
            raise ValueError('Each initial logarithmic probability must be <= 0.')

        initial_probability_sum = states[0].initial_probability
        for i in range(1, self.__states_n):
            initial_probability_sum = lmath.log_sum(initial_probability_sum, states[i].initial_probability)

        if np.abs(initial_probability_sum) > self.__eps:
            raise ValueError("Hidden Markov Models hidden states initial probabilities must be logarithmic and sum to 0.")

        if transition_probabilities:
            if len(transition_probabilities) != self.__states_n:
                raise ValueError("transition_probabilities matrix must be N*N where N is number of states.")
            for line in transition_probabilities:
                line_sum = line[0]
                for i in range(1, len(line)):
                    line_sum = lmath.log_sum(line_sum, line[i])
                if np.abs(line_sum) > self.__eps:
                    raise ValueError("transition_probabilities matrix consist of incorrect logarithmic probabilities. Each row must sum up to 0.")
        initial_transition_probabilities = np.log(1. / self.__states_n)
        self.__transitions = transition_probabilities or [[initial_transition_probabilities for i in range(self.__states_n)] for i in range(self.__states_n)]

    @property
    def states_n(self):
        return self.__states_n

    @property
    def states(self):
        return self.__states

    @property
    def transitions(self):
        return self.__transitions

    def __alpha(self, observations):
        """
        Computes probabilities of being in hidden state 'j'
        at time 't' given all the observations that have come
        before for each time 't' and each hidden state 'j'.

        Formally:
            α_t(j)=P(o_1, o_2, ..., o_t, q_t=j|λ) ∀t, j

        Args:
            observations (): A 1D numpy array of shape '(N,)' where N is number of observations.
                                                               Each observation is represented by a numpy array of MFCC features.

        Returns:
            2D numpy array: Computed α_t(j) for each 't' and 'j'.
        """
        observations_n = observations.shape[0] # Number of observations

        alphas = np.zeros((observations_n, self.__states_n))

        for i in range(self.__states_n):
            alphas[0, i] = self.__states[i].initial_probability + self.__states[i].gmm[observations[0]]

        for t in range(1, observations_n):
            for j in range(self.__states_n): # j - current state
                for i in range(self.__states_n): # i - previous state
                    alphas[t, j] = lmath.log_sum(alphas[t, j], alphas[t - 1, i] + self.__transitions[i][j])
                alphas[t, j] += self.__states[j].gmm[observations[t]]

        return alphas

    def __beta(self, observations):
        """
        Computes probabilities of being in hidden state 'j'
        at time 't' knowing future observations, but not
        knowing observations in the past for each time 't'
        and each hidden state 'j'.

        Formally:
            β_t(j)=P(o_{t+1}, o_{t+2}, ..., o_T|q_t=j, λ) ∀t, j

        Args:
            observations (np.ndarray): A 1D numpy array of shape '(N,)' where N is number of observations.
                                                               Each observation is represented by a numpy array of MFCC features.

        Returns:
            2D numpy array: Computed β_t(j) for each 't' and 'j'.
        """
        observations_n = observations.shape[0]

        betas = np.zeros((observations_n, self.__states_n))

        # Initialization of betas[-1, i] is omitted due to np.log(1)=0

        for t in reversed(range(observations_n - 1)):
            for j in range(self.__states_n): # j - current state
                betas[t, j] = betas[t + 1, 0] + self.__transitions[j][0] + self.__states[0].gmm[observations[t + 1]]
                for i in range(1, self.__states_n): # i - previous state
                    betas[t, j] = lmath.log_sum(betas[t, j], betas[t + 1, i] + self.__transitions[j][i] + self.__states[i].gmm[observations[t + 1]])

        return betas

    def __gamma(self, alphas, betas):
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
        observations_n = alphas.shape[0]

        gammas = np.zeros((observations_n, self.__states_n))
        for t in range(observations_n):
            ab = alphas[t, 0] + betas[t, 0]
            for i in range(1, self.__states_n):
                ab = lmath.log_sum(ab, alphas[t, i] + betas[t, i])
            for i in range(self.__states_n):
                gammas[t, i] = alphas[t, i] + betas[t, i] - ab

        return gammas

    def __ksi(self, alphas, betas, observations):
        """
        For each 't', 'i', 'j' computes probability of
        being in hidden state 'i' at time 't' and in hidden
        state 'j' at time 't+1' given set of the observations.

        Formally:
            ξ_t(i,j)=P(q_t=i,q_{t+1}=j|o, λ) ∀t, i, j

        Args:
            alphas (2D numpy array): data produced by 'alpha' method.
            betas (2D numpy array): data produced by 'beta' method.
            observations (np.ndarray[np.ndarray[np.float64]]): A 1D numpy array of shape '(N,)' where N is number of observations.
                                                               Each observation is represented by a numpy array of MFCC features.

        Returns:
            3D numpy array: Computed ξ_t(i,j) for each 't', 'i' and 'j'.
        """
        observations_n = observations.shape[0]

        ksis = np.zeros((observations_n, self.__states_n, self.__states_n))

        for cur_observation_n in range(observations_n - 1):
            bottom = alphas[cur_observation_n, 0] + self.__transitions[0][0] + self.__states[0].gmm[observations[cur_observation_n + 1]] + betas[cur_observation_n + 1, 0]
            for k in range(self.__states_n):
                for w in range(k == 0, self.__states_n):
                    bottom = lmath.log_sum(bottom, alphas[cur_observation_n, k] + self.__transitions[k][w] + self.__states[w].gmm[observations[cur_observation_n + 1]] + betas[cur_observation_n + 1, w])
            for i in range(self.__states_n):
                for j in range(self.__states_n):
                    ksis[cur_observation_n, i, j] = alphas[cur_observation_n, i] + self.__transitions[i][j] + self.__states[j].gmm[observations[cur_observation_n + 1]] + betas[cur_observation_n + 1, j] - bottom

        return ksis

    def __gmm_gammas(self, observations, gammas):
        observations_n = observations.shape[0]
        n_components = self.__states[0].gmm.n_components
        gmm_gammas = np.zeros((self.__states_n, n_components, observations_n))
        for i in range(self.__states_n):
            gmm_i = self.__states[i].gmm
            for l in range(n_components):
                for t in range(observations_n):
                    gmm_gammas[i][l][t] = gammas[t][i] + gmm_i.log_lth_gaussian_prob(observations[t], l) - gmm_i[observations[t]]
        return gmm_gammas

    def baum_welch(self, observations, eps=1e-4, max_iters=1000, ignore_eps=False, train_gmm=False):
        observations_n = observations.shape[0]

        for it in range(max_iters):
            is_converged = True

            alphas = self.__alpha(observations)
            betas = self.__beta(observations)
            gammas = self.__gamma(alphas, betas)
            ksis = self.__ksi(alphas, betas, observations)

            new_transitions = np.ndarray((self.__states_n, self.__states_n))

            for i in range(self.__states_n):
                self.__states[i].initial_probability = gammas[0, i]

                gamma_sum = gammas[0][i]
                for j in range(1, observations_n - 1):
                    gamma_sum = lmath.log_sum(gamma_sum, gammas[j][i])

                for j in range(self.__states_n):
                    top = ksis[0, i, j]
                    for k in range(1, observations_n - 1):
                        top = lmath.log_sum(top, ksis[k, i, j])
                    new_transitions[i][j] = top - gamma_sum

                if train_gmm:
                    gmm_gammas = self.__gmm_gammas(observations, gammas)
                    gmm_i = self.__states[i].gmm
                    for l in range(gmm_i.n_components):
                        mmax1, mmax2 = max(gmm_gammas[i][l]), max(gammas[:, i])
                        lse1, lse2 = 0, 0
                        for t in range(0, observations_n):
                            lse1 += np.exp(gmm_gammas[i][l][t] - mmax1)
                            lse2 += np.exp(gammas[t][i] - mmax2)
                        lse1 = mmax1 + np.log(lse1)
                        lse2 = mmax2 + np.log(lse2)
                        gmm_i.c[l] = np.exp(lse1 - lse2)
                        updated_gmm_means = np.float64(0)
                        dim = observations.shape[1]
                        updated_gmm_covmatrices = np.zeros((dim, dim))
                        for t in range(observations_n):
                            updated_gmm_means += np.exp(gmm_gammas[i][l][t] - lse1) * observations[t]
                            updated_gmm_covmatrices += np.exp(gmm_gammas[i][l][t] - lse1) * (observations[t] - gmm_i.means[l])[:, np.newaxis] * (observations[t] - gmm_i.means[l])
                        is_converged &= np.linalg.norm(updated_gmm_means - gmm_i.means[l]) < eps and np.linalg.norm(updated_gmm_covmatrices - gmm_i.covmatrices[l]) < eps
                        gmm_i.means[l] = updated_gmm_means
                        gmm_i.covmatrices[l] = updated_gmm_covmatrices + 1e-5 * np.eye(dim)

            is_converged &= np.linalg.norm(new_transitions - self.__transitions) < eps
            self.__transitions = new_transitions
            if not ignore_eps and is_converged:
                return True # has converged

        return False # has not converged or ignore_eps=True

    def print_hmm_data(self, is_converged: bool = None):
        """
        Prints the entire information about the model.

        Args:
            is_converged (bool, optional): If passed, extra first line will
                be printed with information whether algorithm has converged.
                Obtained as a result of Baum-Welch algorithm. Defaults to None.
        """
        if is_converged is not None:
            print("Training has converged" if is_converged else "Training has not converged")
        print(f"States: {self.__states_n}")
        print(f"Initial probabilities: {[x.initial_probability for x in self.__states]}")
        print(f"Transitions:\n{self.__transitions}")
        print("GMMs:\n===")
        for i in range(self.__states_n):
            print(f"GMM-{i}: {self.__states[i].gmm.n_components} components")
            print(f"Weights:\n{self.__states[i].gmm.c}")
            print(f"Means:\n{self.__states[i].gmm.means}")
            print(f"Covs:\n{self.__states[i].gmm.covmatrices}\n===")
        print()

    def plot_gmm_for_1d_case(self, observations):
        """
        Visualizes each GMM by plotting them on different graphs.
        Each GMM graph consists of its components and observations.
        Note: Method does not check whether training has converged.

        Args:
            observations (np.array): An array of 1d observations.
        """
        n, dim = observations.shape
        if dim != 1:
            raise ValueError("Visualization only works for 1d case, but %dd observations found." % dim)
        for i in range(self.__states_n):
            gmm = self.__states[i].gmm
            plt.scatter(observations, np.zeros((n,)), c="black")
            arange = np.arange(-20, 20, 0.001)
            for j in range(gmm.n_components):
                plt.plot(arange, norm.pdf(arange, gmm.means[j], np.sqrt(gmm.covmatrices[j][0])))
            plt.show()

    @staticmethod
    def make_test_1d_model(states_n=3, gmm_n_components=2):
        hmm_states = []
        observations = []
        initial_probability = np.float64(1 / states_n)
        for i in range(states_n):
            obs = datasets.make_blobs(n_features=1, n_samples=75, centers=gmm_n_components)
            hmm_states.append(HMMState(
                str(i),
                GMM(
                    gmm_n_components,
                    obs,
                    means_init="random"
                ),
                initial_probability,
                log_probability=False
            ))
            observations += obs
        return HMM(hmm_states), np.array(observations)


#check kmeans
if __name__ == '__main__':
    hmm, observations = HMM.make_test_1d_model()
    has_converged = hmm.baum_welch(
        observations=observations,
        eps=0.0001,
        max_iters=3000,
        ignore_eps=False,
        train_gmm=True
    )
    hmm.print_hmm_data(has_converged)
    hmm.plot_gmm_for_1d_case(observations)