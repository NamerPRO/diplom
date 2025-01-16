import heapq

import numpy as np

from acoustic_model.hmm import HMM


class ViterbiDecoder:

    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def _viterbi(self, observations):
        observations_n = observations.shape[0]
        best_costs = np.full((observations_n, self.hmm.states_n), -np.inf, dtype=np.float64)
        backtrack = np.full((observations_n, self.hmm.states_n), -1, dtype=object)
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
        restored_path, time = [best_backtrack], -1
        while backtrack[time][best_backtrack] != -1:
            best_backtrack = backtrack[time][best_backtrack]
            restored_path.append(backtrack[time][best_backtrack])
            time -= 1
        return list(reversed(restored_path)), best_cost, best_costs, backtrack

    def decode(self, observations):
        best_path, best_cost, _, _ = self._viterbi(observations)
        return best_path, best_cost


class SerialViterbiDecoder(ViterbiDecoder):

    def __init__(self, hmm: HMM):
        super().__init__(hmm)

    def __restore_path_with_suffix_costs(self, backtrack, path_matrix, paths, observations):
        time, state, path_ind = backtrack
        observations_n = observations.shape[0]
        suffix_costs = np.ndarray((observations_n,))
        suffix_costs[-1] = 0
        path_end = paths[path_ind][1]
        restored_path = [path_end[-1]]
        i = len(path_end) - 2
        while i >= time + 1:
            suffix_costs[i] = suffix_costs[i + 1]  + self.hmm.transitions[path_end[i]][path_end[i + 1]] + self.hmm.states[path_end[i + 1]].gmm[observations[i + 1]]
            restored_path.append(path_end[i])
            i -= 1
        suffix_costs[time] = suffix_costs[time + 1] + self.hmm.transitions[state][path_end[time + 1]] + self.hmm.states[path_end[time + 1]].gmm[observations[time + 1]]
        restored_path.append(state)
        while path_matrix[time][state] != -1:
            restored_path.append(path_matrix[time][state])
            state = path_matrix[time][state]
            time -= 1
            suffix_costs[time] = suffix_costs[time + 1] + self.hmm.transitions[restored_path[-1]][restored_path[-2]] + self.hmm.states[restored_path[-2]].gmm[observations[time + 1]]
        return list(reversed(restored_path)), suffix_costs

    def decode_k(self, observations, n_best):
        path, cost, best_costs, path_matrix = super()._viterbi(observations)
        paths = [(cost, path)]
        merge_data = []
        tau = observations.shape[0]
        prev_states = [{ path[i] } for i in range(tau)]
        sufs = None
        for k in range(n_best - 1):
            for t in range(1, tau):
                prev_state, cur_state = path[t - 1], path[t]
                best_cost, best_ptr = -np.inf, -1
                for l in range(self.hmm.states_n):
                    if l == prev_state or l in prev_states[t - 1] and t == tau - 1:
                        continue
                    suffix_cost = sufs[t] if sufs is not None else best_costs[-1][path[-1]] - best_costs[t][path[t]]
                    cost = best_costs[t - 1][l] + self.hmm.transitions[l][cur_state] + self.hmm.states[cur_state].gmm[observations[t]] + suffix_cost
                    if best_cost < cost:
                        best_cost = cost
                        best_ptr = l
                prev_states[t - 1].add(best_ptr)
                heapq.heappush(merge_data, (-best_cost, (t - 1, best_ptr, k)))
            cost, backtrack = heapq.heappop(merge_data)
            path, sufs = self.__restore_path_with_suffix_costs(backtrack, path_matrix, paths, observations)
            paths.append((-cost, path))
            tau = backtrack[0] + 2 # we need to process till t + 1 since t is time before merge inclusive so (t - 1) + 2 = t + 1
        return paths


class ParallelViterbiDecoder(ViterbiDecoder):

    def __init__(self, hmm: HMM):
        super().__init__(hmm)

    def decode_k(self, observations, n_best):
        observations_n = observations.shape[0]
        best_costs = np.full((observations_n, self.hmm.states_n, n_best), -np.inf)
        backtrack = np.full((observations_n, self.hmm.states_n, n_best), -1, dtype=object)
        ranks = np.full((observations_n, self.hmm.states_n, n_best), -1, dtype=object)

        for i in range(self.hmm.states_n):
            best_costs[0][i][0] = self.hmm.states[i].initial_probability + self.hmm.states[i].gmm[observations[0]]

        for t in range(1, observations_n):
            for i in range(self.hmm.states_n): # cur state
                kmin = []
                for j in range(self.hmm.states_n): # prev state
                    for l in range(n_best):
                        cost = best_costs[t - 1][j][l] + self.hmm.transitions[j][i] + self.hmm.states[i].gmm[observations[t]]
                        heapq.heappush(kmin, (-cost, j, l))

                for l in range(n_best):
                    cost, backtrack_ptr, rank = heapq.heappop(kmin)
                    best_costs[t][i][l] = -cost
                    backtrack[t][i][l] = backtrack_ptr
                    ranks[t][i][l] = rank

        data = []
        for i in range(self.hmm.states_n):
            for l in range(n_best):
                cost = best_costs[-1][i][l]
                heapq.heappush(data, (-cost, i, l))

        paths = []
        for l in range(n_best):
            cost, back_ptr, rank = heapq.heappop(data)
            restored_path, time = [back_ptr], -1
            while backtrack[time][back_ptr][rank] != -1:
                prev_ptr = back_ptr
                back_ptr = backtrack[time][back_ptr][rank]
                rank = ranks[time][prev_ptr][rank]
                restored_path.append(back_ptr)
                time -= 1
            paths.append((-cost, list(reversed(restored_path))))

        return paths
