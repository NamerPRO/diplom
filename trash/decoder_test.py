import heapq

import numpy as np

from acoustic_model.hmm import HMM


class ViterbiDecoder:

    def __init__(self, ip, transitions_probs, emission_probs, states_n):
        self.transitions_probs = transitions_probs
        self.emission_probs = emission_probs
        self.states_n = states_n
        self.ip = ip

    def __viterbi(self, observations):
        observations_n = observations.shape[0]
        best_costs = np.full((observations_n, self.states_n), -np.inf, dtype=np.float64)
        backtrack = np.full((observations_n, self.states_n), -1, dtype=object)
        for i in range(self.states_n):
            best_costs[0, i] = self.ip[i] + self.emission_probs[i][observations[0]] # self.hmm.states[i].gmm[observations[0]]
        for t in range(1, observations_n):
            for j in range(self.states_n): # cur state
                for i in range(self.states_n): # prev state
                    cost = best_costs[t - 1, i] + self.transitions_probs[i][j] + self.emission_probs[j][observations[t]] # self.hmm.states[j].gmm[observations[t]]
                    if best_costs[t, j] < cost:
                        best_costs[t, j] = cost
                        backtrack[t, j] = i
        best_cost = -np.inf
        best_backtrack = -1
        for i in range(self.states_n):
            if best_costs[-1, i] > best_cost:
                best_cost = best_costs[-1, i]
                best_backtrack = i
        best_path = self.__restore_path((-1, best_backtrack, -1), backtrack)
        return best_path, best_cost, best_costs, backtrack

    def pre_calc_sums(self, observations, path):
        observations_n = observations.shape[0]
        sums = np.zeros((observations_n,))
        sums[0] = self.ip[path[0]] + self.emission_probs[path[0]][observations[0]]
        for i in range(0, observations_n - 1):
            sums[i + 1] = sums[i] + self.transitions_probs[path[i]][path[i + 1]] + self.emission_probs[path[i + 1]][observations[i + 1]]  #best_costs[i + 1][path[i + 1]]
        return sums

    # think whether pre_calc_sums method can be integrated here
    def __restore_path(self, backtrack, path_matrix, paths=None):
        time, state, path_ind = backtrack
        # time -= 1
        restored_path = [state]
        # restored_path = [] if path else [state]
        while path_matrix[time][state] != -1:
            restored_path.append(path_matrix[time][state])
            state = path_matrix[time][state]
            time -= 1
        return list(reversed(restored_path)) + (paths[path_ind][1][backtrack[0] + 1:] if paths else [])

    def decode(self, observations):
        best_path, best_cost, _, _ = self.__viterbi(observations)
        return best_path, best_cost

    def decode_k(self, observations, n_best):
        path, cost, best_costs, path_matrix = self.__viterbi(observations)
        paths = [(cost, path)]
        merge_data = []
        tau = observations.shape[0]
        prev_path = []
        prev_states = np.ndarray((tau,), dtype=set)
        for i in range(tau):
            prev_states[i] = set()
            prev_states[i].add(path[i])

        for k in range(n_best - 1):
            sums = self.pre_calc_sums(observations, path)
            flag = True
            for t in range(1, tau):
                # if flag and prev_path and prev_path[t] == path[t]:
                #     continue
                # flag = False
                prev_state, cur_state = path[t - 1], path[t]
                best_cost, best_ptr = -np.inf, -1
                for l in range(self.states_n):
                    if l in prev_states[t - 1] and t == tau - 1:
                        continue
                    if l == prev_state:
                        continue
                    cost = best_costs[t - 1][l] + self.transitions_probs[l][cur_state] + self.emission_probs[cur_state][observations[t]] + sums[-1] - sums[t]# self.hmm.states[cur_state].gmm[observations[t]] + sums[-1] - sums[t]
                    if best_cost < cost:
                        best_cost = cost
                        best_ptr = l
                prev_states[t - 1].add(best_ptr)
                heapq.heappush(merge_data, (-best_cost, (t - 1, best_ptr, k)))
            cost, backtrack = heapq.heappop(merge_data)

            # prev_states[backtrack[0]].add(backtrack[1])

            prev_path = path
            path = self.__restore_path(backtrack, path_matrix, paths)
            paths.append((-cost, path))
            tau = backtrack[0] + 2 # we need to process till t inclusive so (t - 1) + 2 = t + 1
        return paths


if __name__ == '__main__':
    transition_probs = np.log(np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5]
    ]))
    emission_probs = np.log(np.array([
        [0.9, 0.05, 0.05],
        [0.2, 0.7, 0.1],
        [0.1, 0.3, 0.6]
    ]))
    initial_probs = np.log(np.array([0.6, 0.3, 0.1]))
    states_n = len(initial_probs)

    decoder = ViterbiDecoder(initial_probs, transition_probs, emission_probs, states_n)
    obs = np.array([[0], [1], [2], [1], [1], [2], [0], [2], [2], [1]])
    x = decoder.decode_k(obs, 1000)
    for i in x:
        print(i)
    # print(decoder.pre_calc_sums(obs, np.array([0, 1, 2, 1, 1, 2, 1, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 1, 2, 1, 1, 2, 0, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 1, 2, 2, 2, 2, 2, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 1, 2, 2, 1, 2, 2, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 2, 2, 1, 1, 2, 2, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 1, 1, 1, 1, 2, 2, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([1, 1, 2, 1, 1, 2, 2, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 1, 2, 1, 1, 1, 2, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 1, 1, 2, 2, 2, 2, 2, 2])))
    # print(decoder.pre_calc_sums(obs, np.array([0, 2, 1, 1, 2, 2, 0, 2, 2])))
