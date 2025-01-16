import itertools
import math

import numpy as np

from trash.decoder_test import ViterbiDecoder


# Exhaustive search for verification
def brute_force_n_best(pi, A, O, observations, nbest):
    M = len(observations)
    S = pi.shape[0]

    scores = []

    # loop over the cartesian product of |states|^M
    for ss in itertools.product(range(S), repeat=M):
        # score the state sequence
        score = pi[ss[0]] * O[ss[0], observations[0]]
        for i in range(1, M):
            score *= A[ss[i - 1], ss[i]] * O[ss[i], observations[i]]

        scores.append((np.log(score), ss))

    scores.sort(key=lambda x: -x[0])
    return scores[:nbest]


if __name__ == '__main__':
    A = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ])
    B = np.array([
        [0.9, 0.05, 0.05],
        [0.2, 0.7, 0.1],
        [0.1, 0.3, 0.6]
    ])
    initial_probs = np.array([0.6, 0.3, 0.1])
    states_n = len(initial_probs)
    obs = np.array([0, 1, 2, 1, 1, 2, 0, 2, 2, 1])

    nbest=10
    res = brute_force_n_best(initial_probs, A, B, obs, nbest)

    # Call
    decoder = ViterbiDecoder(np.log(initial_probs), np.log(A), np.log(B), states_n)
    obs = np.array([[0], [1], [2], [1], [1], [2], [0], [2], [2], [1]])
    my = decoder.decode_k(obs, nbest)
    for i in range(nbest):
        print(res[i])
        # print(my[i], i)
        # print(math.isclose(res[i][0], my[i][0]), res[i][1] == my[i][1])