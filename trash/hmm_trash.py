from abc import ABC, abstractmethod
from typing import Final, List

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from acoustic_model.gmm import GMM
from acoustic_model.hmm_state import HMMState

from scipy.stats import norm

class HMMGMM:
    """
    Hidden Markov Model with Gaussian mixture emission probabilities.

    Private Attributes:
        __states_n (int): Number of states in Hidden Markov Model.

        __states (List[HMMState]): List of '__states_n' elements. Each item represent a state of
                                   Hidden Markov Models.

        __transitions (np.ndarray): A 2D nupy array of shape '(N,N)' where N is the number of states
                                    in the Hidden Markov Model. Consists of transition probabilities
                                    for Hidden Markov Model.
    """


    def __init__(self, states: List[HMMState], transition_probabilities = None):
        if not np.isclose(sum(state.initial_probability for state in states), 1):
            raise ValueError("Hidden Markov Models hidden states initial probabilities must sum to 1.")

        self.__states = states
        self.__states_n = len(states)

        if transition_probabilities:
            if len(transition_probabilities) != self.__states_n:
                raise ValueError("transition_probabilities matrix must be N*N where N is number of states.")
            for line in transition_probabilities:
                if not np.isclose(sum(line), 1.0):
                    raise ValueError("transition_probabilities matrix consist of incorrect probabilities.")
        self.__transitions = transition_probabilities or [[1. / self.__states_n for i in range(self.__states_n)] for i in range(self.__states_n)]

    @staticmethod
    def create_untrained_test_hmm_gmm_model(number_of_states=2, gmm_n_components=3, means_init="kmeans", kmeans_max_iterations=1000, kmeans_attempts=5):
        """
        Creates untrained Hidden Markov Model with Gaussian mixture emission probabilities with synthetic data.

        Args:
            number_of_states (int): Number of states in Hidden Markov Model.
            gmm_n_components (int): Number of components in each GMM.
            means_init (str): Type of mean initialisation in GMM: kmeans or random.
            kmeans_max_iterations (int): if means_init set to 'kmeans' is a number of iterations
                for this method, else ignored.
            kmeans_attempts (int): if means_init set to 'kmeans' is a number of attempts for
                this method, else ignored.
        """
        hmm_states = []
        all_observations = None
        per_gmm_observations = None
        for i in range(number_of_states):
            new_observations, _ = datasets.make_blobs(n_features=1, n_samples=30, centers=gmm_n_components)
            state = HMMState(str(i), GMM(gmm_n_components, new_observations, means_init, kmeans_max_iterations, kmeans_attempts), np.float64(1) / number_of_states)
            hmm_states.append(state)
            if all_observations is not None:
                all_observations = np.vstack((all_observations, new_observations))
                per_gmm_observations = np.vstack((per_gmm_observations, [new_observations]))
            else:
                all_observations = new_observations
                per_gmm_observations = [new_observations]
        return HMMGMM(hmm_states), all_observations, per_gmm_observations

    @staticmethod
    def create_untrained_test_hmm_gmm_train_and_visualize(number_of_states=2, gmm_n_components=3, means_init="kmeans", kmeans_max_iterations=1000, kmeans_attempts=5, print_observations=False, baum_welch_eps=0.00001, baum_welch_max_iterations=1000, baum_welch_ignore_eps=False):
        hmmgmm, all_observations, per_gmm_observations = HMMGMM.create_untrained_test_hmm_gmm_model(number_of_states, gmm_n_components, means_init, kmeans_max_iterations, kmeans_attempts)

        y = np.zeros((len(per_gmm_observations[0]),))
        arange = np.arange(-20, 20, 0.001)
        for i in range(number_of_states):
            state_i = hmmgmm.__states[i]
            gmm = state_i.gmm
            plt.scatter(per_gmm_observations[i], y, c="black")
            for j in range(gmm_n_components):
                plt.plot(arange, norm.pdf(arange, gmm.means[j], np.sqrt(gmm.covmatrices[j][0])))
            plt.show()


        hmmgmm.baum_welch(all_observations, baum_welch_eps, baum_welch_max_iterations, baum_welch_ignore_eps)
        # y = np.zeros((len(per_gmm_observations[0]),))
        # arange = np.arange(-20, 20, 0.001)
        print(f"Each state has {gmm_n_components}-component GMM.\n")
        print(per_gmm_observations if print_observations else "Observations print disabled.")
        print(f"\nTransition probabilities:\n{hmmgmm.__transitions}")
        for i in range(number_of_states):
            state_i = hmmgmm.__states[i]
            gmm = state_i.gmm
            output = f"""
                STATE {i}
                Initial probability: {state_i.initial_probability}
                GMM information
                ====
                coefficients: {gmm.c}
                means: {gmm.means.T}
                variances: {gmm.covmatrices.T}
                ===
            """
            print("\n".join([line.strip() for line in output.splitlines()]))
            plt.scatter(per_gmm_observations[i], y, c="black")
            for j in range(gmm_n_components):
                plt.plot(arange, norm.pdf(arange, gmm.means[j], np.sqrt(gmm.covmatrices[j][0])))
            plt.show()

    def alpha(self, observations):
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

        alphas = np.zeros((observations_n, self.__states_n), dtype=np.float64)

        for init_state in range(self.__states_n):
            alphas[0, init_state] = self.__states[init_state].initial_probability * self.__states[init_state].gmm[observations[0]]

        for cur_observation_n in range(1, observations_n):
            for cur_state_n in range(self.__states_n):
                for prev_state_n in range(self.__states_n):
                    alphas[cur_observation_n, cur_state_n] += alphas[cur_observation_n - 1, prev_state_n] * self.__transitions[prev_state_n][cur_state_n] * self.__states[cur_state_n].gmm[observations[cur_observation_n]]

        return alphas


    def beta(self, observations):
        """
        Computes probabilities of being in hidden state 'j'
        at time 't' knowing future observations, but not
        knowing observations in the past for each time 't'
        and each hidden state 'j'.

        Formally:
            β_t(j)=P(o_{t+1}, o_{t+2}, ..., o_T|q_t=j, λ) ∀t, j

        Args:
            observations (np.ndarray[np.ndarray[np.float64]]): A 1D numpy array of shape '(N,)' where N is number of observations.
                                                               Each observation is represented by a numpy array of MFCC features.

        Returns:
            2D numpy array: Computed β_t(j) for each 't' and 'j'.
        """
        observations_n = observations.shape[0]

        betas = np.zeros((observations_n, self.__states_n), dtype=np.float64)

        for init_state in range(self.__states_n):
            betas[-1, init_state] = 1

        for cur_observation_n in reversed(range(observations_n - 1)):
            for cur_state_n in range(self.__states_n):
                for prev_state_n in range(self.__states_n):
                    betas[cur_observation_n, cur_state_n] += betas[cur_observation_n + 1, prev_state_n] * self.__transitions[cur_state_n][prev_state_n] * self.__states[prev_state_n].gmm[observations[cur_observation_n + 1]]

        return betas


    def gamma(self, alphas, betas):
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

        gammas = np.zeros((observations_n, self.__states_n), dtype=np.float64)
        for cur_observation_n in range(observations_n):
            denom_prob = np.float64(0)
            for cur_state_n in range(self.__states_n):
                denom_prob += alphas[cur_observation_n, cur_state_n] * betas[cur_observation_n, cur_state_n]
            for cur_state_n in range(self.__states_n):
                num_prob = alphas[cur_observation_n, cur_state_n] * betas[cur_observation_n, cur_state_n]
                gammas[cur_observation_n, cur_state_n] = num_prob / denom_prob

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

        ksis = np.zeros((observations_n, self.__states_n, self.__states_n), dtype=np.float64)

        for cur_observation_n in range(observations_n - 1):
            denom_prob = np.float64(0)
            for k in range(self.__states_n):
                for w in range(self.__states_n):
                    denom_prob += alphas[cur_observation_n, k] * self.__transitions[k][w] * self.__states[w].gmm[observations[cur_observation_n + 1]] * betas[cur_observation_n + 1, w]
            for i in range(self.__states_n):
                for j in range(self.__states_n):
                    num_prob = alphas[cur_observation_n, i] * self.__transitions[i][j] * self.__states[j].gmm[observations[cur_observation_n + 1]] * betas[cur_observation_n + 1, j]
                    ksis[cur_observation_n, i, j] = num_prob / denom_prob

        return ksis

    def __gmm_gammas(self, observations, gammas):
        T = observations.shape[0]
        n_components = self.__states[0].gmm.n_components
        gmm_gammas = np.zeros((self.__states_n, n_components, T))
        for i in range(self.__states_n):
            gmm_i = self.__states[i].gmm
            for l in range(n_components):
                for t in range(T):
                    gmm_gammas[i][l][t] = gammas[t][i] * gmm_i.lth_gaussian_prob(observations[t], l) / gmm_i[observations[t]]
        return gmm_gammas

    def forward(self, observations):
        """
        Implementation of forward algorithm.

        Args:
            observations (np.ndarray[np.ndarray[np.float64]]): A 1D numpy array of shape '(N,)' where N is number of observations.
                                                               Each observation is represented by a numpy array of MFCC features.

        Returns:
            np.float64: Computed probability.
        """
        probs = self.__alpha(observations)
        return np.sum(probs[-1, :])


    def backward(self, observations):
        """
        Implementation of backward algorithm.

        Args:
            observations (np.ndarray[np.ndarray[np.float64]]): A 1D numpy array of shape '(N,)' where N is number of observations.
                                                               Each observation is represented by a numpy array of MFCC features.

        Returns:
            np.float64: Computed probability.
        """
        probs = self.__beta(observations)

        total = np.float64(0)
        for i in range(self.__states_n):
            total += probs[0, i] * self.__states[i].initial_probability * self.__states[i].gmm[observations[0]]

        return total


    def viterbi(self, observations):
        """
        Implementation of Viterbi algorithm.

        Args:
            observations (np.ndarray[np.ndarray[np.float64]]): A 1D numpy array of shape '(N,)' where N is number of observations.
                                                               Each observation is represented by a numpy array of MFCC features.

        Returns:
            Tuple[list, numpy.float64]: A tuple containing:
                - Most probable sequence of hidden states.
                - Probability of this sequence.
        """
        observations_n = observations.shape[0]

        dp = np.zeros((observations_n, self.__states_n), dtype=np.float64)
        q = np.full((observations_n, self.__states_n), -1, dtype=int)

        for j in range(self.__states_n):
            dp[0, j] = self.__states[j].initial_probability * self.__states[j].gmm[observations[0]]

        for t in range(1, observations_n):
            for j in range(self.__states_n):
                for i in range(self.__states_n):
                    end_in_j_prob = dp[t - 1, i] * self.__transitions[i, j] * self.__states[j].gmm[observations[t]]
                    if dp[t, j] < end_in_j_prob:
                        dp[t, j] = end_in_j_prob
                        q[t, j] = i

        max_prob = -np.inf
        max_prob_i = 0
        for i in range(self.__states_n):
            if dp[-1, i] > max_prob:
                max_prob = dp[-1, i]
                max_prob_i = i

        answ, time = [max_prob_i], -1
        while q[time][max_prob_i] != -1:
            answ.append(q[time][max_prob_i])
            max_prob_i = q[time][max_prob_i]
            time -= 1

        return reversed(answ), max_prob


    def exact_n_best(self):
        pass

    def baum_welch(self, observations, eps=0.00001, max_iters=1000, ignore_eps=False):
        """
        Implementation of Baum-Welch algorithm.

        Args:
            observations (np.ndarray[np.ndarray[np.float64]]): A 1D numpy array of shape '(T,)' where T is the number
                of time steps. Consists of observations.
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
        observations_n = observations.shape[0]

        for it in range(max_iters):
            converged_gmm_means = True
            converged_gmm_covmatrices = True
            converged_transitions = True

            alphas = self.__alpha(observations)
            betas = self.__beta(observations)
            gammas = self.__gamma(alphas, betas)
            ksis = self.__ksi(alphas, betas, observations)
            gmm_gammas = self.__gmm_gammas(observations, gammas)

            new_transitions = np.ndarray((self.__states_n, self.__states_n), dtype=np.float64)

            for i in range(self.__states_n):
                # Updating initial probabilities
                self.__states[i].initial_probability = gammas[0, i]

                # Updating transition probabilities
                gamma_sum_besides_last = np.sum(gammas[:-1, i])
                for j in range(self.__states_n):
                    new_transitions[i][j] = np.sum(ksis[:-1, i, j]) / gamma_sum_besides_last

                # Updating GMM parameters
                gamma_sum = gamma_sum_besides_last + gammas[-1, i]
                gmm_i = self.__states[i].gmm
                for l in range(gmm_i.n_components):
                    lth_gmm_gammas_sum = 0
                    observation_dim = observations.shape[1]
                    lth_gaussian_gmm_means = np.zeros((observation_dim,))
                    for t in range(observations_n):
                        lth_gmm_gammas_sum += gmm_gammas[i, l, t]
                        lth_gaussian_gmm_means += gmm_gammas[i, l, t] * observations[t]
                    gmm_i.c[l] = lth_gmm_gammas_sum / gamma_sum
                    updated_gmm_means = lth_gaussian_gmm_means / lth_gmm_gammas_sum
                    converged_gmm_means &= np.linalg.norm(updated_gmm_means - gmm_i.means[l]) < eps
                    gmm_i.means[l] = updated_gmm_means
                    lth_gaussian_gmm_variances = np.zeros((observation_dim, observation_dim))
                    for t in range(observations_n):
                        lth_gaussian_gmm_variances += gmm_gammas[i, l, t] * (observations[t] - gmm_i.means[l]).T * (observations[t] - gmm_i.means[l])
                    updated_gmm_covmatrices = lth_gaussian_gmm_variances / lth_gmm_gammas_sum
                    converged_gmm_covmatrices &= np.linalg.norm(updated_gmm_covmatrices - gmm_i.covmatrices[l]) < eps
                    gmm_i.covmatrices[l] = updated_gmm_covmatrices

            converged_transitions = np.linalg.norm(new_transitions - self.__transitions) < eps
            self.__transitions = new_transitions
            if not ignore_eps and converged_gmm_means and converged_gmm_covmatrices and converged_transitions:
                break


if __name__ == '__main__':
    # 6.997853649404956e-06
    # 4.791084619683631e-10
    # 1.6283528058423908e-13
    # 5.676950767782312e-18
    # 1.737599075242477e-21
    # 1.7678376903199878e-26
    # 5.9103495786803334e-30
    # 4.1232969088103066e-34
    # 2.841046251430036e-39
    # 3.3468022589485697e-43
    observations = np.array([[-5.1264101, 4.54832182, -4.58751138, 5.25010411],
                    [-4.92796138, 4.1509452, -7.95003681, -8.49320315],
                    [5.06547746, 0.47018908, 5.40772812, 0.51091962],
                    [-7.13369485, 5.0330122, -3.60152971, 5.29487825],
                    [-6.08928355, 3.30373881, -8.15888813, -9.10275598],
                    [6.60077888, -1.89502569, 4.97859433, 0.34524229],
                    [8.02113197, 0.4590251, 6.3297532, -0.05357864],
                    [-4.92348852, 4.34663022, -7.60703999, -8.42360852],
                    [-5.91335909, 4.48925547, -4.28575091, 6.90227208],
                    [7.47970946, -0.21634921, 5.52826701, 1.2531821]])
    hmm = HMMGMM(
        [HMMState(
            "1",
            GMM(
                3,
                observations,
                means_init="kmeans"
            ),
            np.float64(1),
            use_log_probability=False)
        ])
    xxx = hmm.alpha(observations)
    yyy = hmm.beta(observations)
    zzz = hmm.gamma(xxx, yyy)
    for i in range(len(zzz)):
        print(zzz[i][0])


    exit(0)
    HMMGMM.create_untrained_test_hmm_gmm_train_and_visualize(
        number_of_states=4,
        kmeans_attempts=25,
        baum_welch_ignore_eps=False,
        baum_welch_max_iterations=1000,
        means_init="random"
    )
    exit(0)


    hmmgmm, obs = HMMGMM.create_untrained_test_hmm_gmm_model()
    y = np.zeros((len(obs),))


    gmm = hmmgmm.baum_welch(np.array(obs))

    print(gmm[0].n_components)
    print(gmm[0].c)
    print(gmm[0].covmatrices)
    print(gmm[0].means)

    print("===========")

    print(gmm[1].n_components)
    print(gmm[1].c)
    print(gmm[1].covmatrices)
    print(gmm[1].means)

    plt.scatter(obs, y, c="black")


    plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm[0].means[0], gmm[0].covmatrices[0][0]))
    plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm[0].means[1], gmm[0].covmatrices[1][0]))
    plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm[0].means[2], gmm[0].covmatrices[2][0]))
    plt.show()

    plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm[1].means[0], gmm[1].covmatrices[0][0]))
    plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm[1].means[1], gmm[1].covmatrices[1][0]))
    plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm[1].means[2], gmm[1].covmatrices[2][0]))
    plt.show()

    # plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm.means[0], gmm.covmatrices[0][1]))
    # plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm.means[1], gmm.covmatrices[1][1]))
    # plt.plot(np.arange(-20, 20, 0.001), norm.pdf(np.arange(-20, 20, 0.001), gmm.means[2], gmm.covmatrices[2][1]))
    # plt.plot()

    # a = np.array([
    #     [0.1, 0.3, 0.1, 0.2, 0.3],
    #     [0.3, 0.1, 0.1, 0.1, 0.4],
    #     [0.1, 0.1, 0.1, 0.4, 0.3],
    #     [0.3, 0.3, 0.2, 0.2, 0.0],
    #     [0.0, 0.1, 0.5, 0.2, 0.2]
    # ])
    # b = np.array([
    #     [0.3, 0.1, 0.2, 0.3, 0.1],
    #     [0.0, 0.1, 0.1, 0.3, 0.5],
    #     [0.1, 0.2, 0.3, 0.2, 0.2],
    #     [0.3, 0.3, 0.1, 0.2, 0.1],
    #     [0.1, 0.3, 0.2, 0.3, 0.1]
    # ])
    # start = np.array([0.2, 0.1, 0.1, 0.2, 0.4])
    # o = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 2])
    #
    # hmm = HMMGMM(None, None)
    # new_start, new_a, new_b = baum_welch(o, a, b, start, max_iters=10000, ignore_eps=True)
    #
    # print(new_start)
    # print(new_a)
    # print(new_b)
