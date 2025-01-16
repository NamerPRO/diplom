import heapq

import numpy as np

from acoustic_model.hmm import HMM


class ViterbiDecoder:

    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def __viterbi(self, observations):
        observations_n = observations.shape[0]
        best_costs = np.full((observations_n, self.hmm.states_n), -np.inf, dtype=np.float64)
        backtrack = np.full((observations_n, self.hmm.states_n), -1, dtype=int)
        for i in range(self.hmm.states_n):
            best_costs[0, i] = self.hmm.states[i].initial_probability + self.hmm.states[i].gmm[observations[0]]
        for t in range(1, observations_n):
            for j in range(self.hmm.states_n): # cur state
                for i in range(self.hmm.states_n): # prev state
                    cost = best_costs[t - 1, i] + self.hmm.transitions[i][j] + self.hmm.states[j].gmm[observations[t]]
                    if best_costs[t, j] < cost:
                        best_costs[t, j] = cost
                        backtrack[t, j] = i
        best_cost = -np.inf
        best_backtrack = -1
        for i in range(self.hmm.states_n):
            if best_costs[-1, i] > best_cost:
                best_cost = best_costs[-1, i]
                best_backtrack = i
        best_path = self.__restore_path((-1, best_backtrack), backtrack)
        return best_path, best_cost, best_costs, backtrack

    def __pre_calc_sums(self, observations, best_costs, path):
        observations_n = observations.shape[0]
        sums = np.zeros((observations_n,))
        sums[0] = best_costs[0][path[0]]
        for i in range(observations_n - 1):
            sums[i] = sums[i + 1] + best_costs[i][path[i]]
        return sums

    def __restore_path(self, backtrack_ptr, path_matrix, path=None):
        time, state = backtrack_ptr
        restored_path = [state]
        while path_matrix[time][state] != -1:
            restored_path.append(path_matrix[time][state])
            state = path_matrix[time][state]
            time -= 1
        return list(reversed(restored_path)) + (path[time:] if path else [])

    def decode(self, observations):
        best_path, best_cost, _, _ = self.__viterbi(observations)
        return best_path, best_cost

    def decode_k(self, observations, best_costs, n_best):
        path, cost, best_costs, path_matrix = self.__viterbi(observations)
        paths = [(cost, path)]
        merge_data = []
        tau = observations.shape[0]
        prev_path = []
        for k in range(n_best):
            sums = self.__pre_calc_sums(observations, best_costs, path)
            for t in range(1, tau):
                if prev_path and prev_path[t] == path[t]:
                    continue
                prev_state, cur_state = path[t - 1], path[t]
                best_cost, best_ptr = -np.inf, -1
                for l in range(self.hmm.states_n):
                    if l == prev_state:
                        continue
                    cost = best_costs[t - 1][l] + self.hmm.transitions[l][cur_state] + self.hmm.states[cur_state].gmm[
                        observations[t]] + sums[-1] - sums[t]
                    if best_cost < cost:
                        best_cost = cost
                        best_ptr = l
                heapq.heappush(merge_data, (-best_cost, (t, best_ptr)))
            cost, backtrack_ptr = heapq.heappop(merge_data)
            prev_path = path
            path = self.__restore_path(backtrack_ptr, path_matrix, path)
            paths.append((-cost, path))
            tau = backtrack_ptr[0]
        return paths
