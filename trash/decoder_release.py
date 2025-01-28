import heapq

import numpy as np


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
        restored_path, time = [best_backtrack], -1
        while backtrack[time][best_backtrack] != -1:
            best_backtrack = backtrack[time][best_backtrack]
            restored_path.append(backtrack[time][best_backtrack])
            time -= 1
        return list(reversed(restored_path)), best_cost, best_costs, backtrack

    def __restore_path_with_suffix_costs(self, backtrack, path_matrix, paths, observations):
        time, state, path_ind = backtrack
        observations_n = observations.shape[0]
        suffix_costs = np.ndarray((observations_n,))
        suffix_costs[-1] = 0
        path_end = paths[path_ind][1]
        restored_path = [path_end[-1]]
        i = len(path_end) - 2
        while i >= time + 1:
            suffix_costs[i] = suffix_costs[i + 1]  + self.transitions_probs[path_end[i]][path_end[i + 1]] + self.emission_probs[path_end[i + 1]][observations[i + 1]]
            restored_path.append(path_end[i])
            i -= 1
        suffix_costs[time] = suffix_costs[time + 1] + self.transitions_probs[state][path_end[time + 1]] + self.emission_probs[path_end[time + 1]][observations[time + 1]]
        restored_path.append(state)
        while path_matrix[time][state] != -1:
            restored_path.append(path_matrix[time][state])
            state = path_matrix[time][state]
            time -= 1
            suffix_costs[time] = suffix_costs[time + 1] + self.transitions_probs[restored_path[-1]][restored_path[-2]] + self.emission_probs[restored_path[-2]][observations[time + 1]]
        return list(reversed(restored_path)), suffix_costs

    def decode(self, observations):
        best_path, best_cost, _, _ = self.__viterbi(observations)
        return best_path, best_cost

    def decode_k(self, observations, n_best):
        path, cost, best_costs, path_matrix = self.__viterbi(observations)
        paths = [(cost, path)]
        merge_data = []
        tau = observations.shape[0]
        prev_states = [{path[i]} for i in range(tau)]
        # for i in range(tau):
        #     prev_states[i].add(path[i])
        sufs = None
        for k in range(n_best - 1):
            for t in range(1, tau):
                prev_state, cur_state = path[t - 1], path[t]
                best_cost, best_ptr = -np.inf, -1
                for l in range(self.states_n):
                    if l == prev_state or l in prev_states[t - 1] and t == tau - 1:
                        continue
                    suffix_cost = sufs[t] if sufs is not None else best_costs[-1][path[-1]] - best_costs[t][path[t]]
                    cost = best_costs[t - 1][l] + self.transitions_probs[l][cur_state] + self.emission_probs[cur_state][observations[t]] + suffix_cost #sums[-1] - sums[t]# self.hmm.states[cur_state].gmm[observations[t]] + sums[-1] - sums[t]
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


class ParallelViterbiDecoder:

    def __init__(self, ip, transitions_probs, emission_probs, states_n):
        self.transitions_probs = transitions_probs
        self.emission_probs = emission_probs
        self.states_n = states_n
        self.ip = ip

    def pre_calc_sums(self, observations, path):
        observations_n = observations.shape[0]
        sums = np.zeros((observations_n,))
        sums[0] = self.ip[path[0]] + self.emission_probs[path[0]][observations[0]]
        for i in range(0, observations_n - 1):
            sums[i + 1] = sums[i] + self.transitions_probs[path[i]][path[i + 1]] + self.emission_probs[path[i + 1]][observations[i + 1]]  #best_costs[i + 1][path[i + 1]]
        return sums

    def decode_k(self, observations, n_best):
        observations_n = observations.shape[0]
        best_costs = np.full((observations_n, self.states_n, n_best), -np.inf)
        backtrack = np.full((observations_n, self.states_n, n_best), -1, dtype=object)
        ranks = np.full((observations_n, self.states_n, n_best), -1, dtype=object)

        for i in range(self.states_n):
            best_costs[0][i][0] = self.ip[i] + self.emission_probs[i][observations[0]]

        for t in range(1, observations_n):
            for i in range(self.states_n): # cur state
                kmin = []
                for j in range(self.states_n): # prev state
                    for l in range(n_best):
                        cost = best_costs[t - 1][j][l] + self.transitions_probs[j][i] + self.emission_probs[i][observations[t]]# self.hmm.states[i].gmm[observations[t]]
                        heapq.heappush(kmin, (-cost, j, l))

                for k in range(n_best):
                    cost, backtrack_ptr, rank = heapq.heappop(kmin)
                    best_costs[t][i][k] = -cost
                    backtrack[t][i][k] = backtrack_ptr
                    ranks[t][i][k] = rank

        data = []
        for i in range(self.states_n):
            for k in range(n_best):
                cost = best_costs[-1][i][k]
                heapq.heappush(data, (-cost, i, k))

        paths = []
        for k in range(n_best):
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

    # decoder = ViterbiDecoder(initial_probs, transition_probs, emission_probs, states_n)
    decoder2 = ParallelViterbiDecoder(initial_probs, transition_probs, emission_probs, states_n)
    obs = np.array([0, 1, 2, 1, 1, 2, 0, 2, 2])
    x = decoder2.decode_k(obs, 100)
    # for i in range(len(x)):
    #     www = decoder2.pre_calc_sums(obs, np.float64(x[i][1]))
    #     if not math.isclose(www, x[i][0]):
    #         print("HERE!")
        # print(x[i][1], i)
    # print(decoder2.pre_calc_sums(obs, [0, 1, 2, 1, 2, 2, 1, 1, 2]))
    # [0 1 2 1 2 1 1 2 2] -17.21674855485122
