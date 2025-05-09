import heapq

from recognition.models.language.language import KatzSmoothingLM
from recognition.models.acoustic.hmm import *
from utils.ngram import NGram


LMSF = 15
WIP = 0.5


class ViterbiDecoder:
    """Реализация декодера согласно алгоритму Витерби"""

    def __init__(self, hmms, lm):
        """
        Инициализация параметров декодера.

        Аргументы:
            hmm: Предварительно натренированная HMM-GMM модель.
            lm: Предварительно натренированная языковая модель.
        """
        self.hmms = hmms
        self.lm = lm

    def __viterbi(self, observations):
        """
        Дана на входе последовательность наблюдений observations = o₁, o₂, ..., oₜ.
        Находит наиболее вероятную последовательность скрытых состояний Q = q₁q₂q₃...qₜ.

        Аргументы:
            observations: Последовательность наблюдений, для которой требуется найти
                наиболее вероятную последовательность состояний.

        Возвращаемое значение:
            Кортеж, содержащий:
                - Найденная последовательность скрытых состояний.
                - Вероятность этой последовательности.
        """
        observations_n = observations.shape[0]
        best_costs = [np.full((observations_n, hmm.states_n), np.inf, dtype=np.float64) for hmm in self.hmms]
        backtrack = [np.full((observations_n, hmm.states_n), None, dtype=object) for hmm in self.hmms]
        for hmm_id, hmm in enumerate(self.hmms):
            for i in range(hmm.states_n):
                best_costs[hmm_id][0, i] = hmm.states[i].initial_probability + hmm.states[i].gmm[observations[0]]
        active_end_word_states = set()
        decoded_words = 1
        for t in range(1, observations_n):
            # intra-word transitions
            for hmm_id, hmm in enumerate(self.hmms):
                for j in range(hmm.states_n):  # cur state
                    for i in range(hmm.states_n):  # prev state
                        cost = best_costs[hmm_id][t - 1, i] + hmm.transitions[i][j] + hmm.states[j].gmm[observations[t]]
                        if best_costs[hmm_id][t, j] > cost:
                            best_costs[hmm_id][t, j] = cost
                            backtrack[hmm_id][t, j] = (hmm_id, i, False)
                    if best_costs[hmm_id][t, j] != np.inf and j == hmm.end_state:
                        active_end_word_states.add((hmm_id, hmm.association, j))
            # inter-word transitions
            if active_end_word_states:
                is_word_decoded = False
                for hmm_id, hmm in enumerate(self.hmms):
                    begin_state = hmm.begin_state
                    for hmm_end_state_id, prev_word, end_state in active_end_word_states:
                        ngram = NGram.from_words_list((prev_word, hmm.association))
                        cost = best_costs[hmm_end_state_id][t, end_state] + self.lm.get_probability(
                            ngram) * LMSF - decoded_words * np.log(WIP)
                        if best_costs[hmm_id][t, begin_state] > cost:
                            is_word_decoded = True
                            best_costs[hmm_id][t, begin_state] = cost
                            backtrack[hmm_id][t, begin_state] = (hmm_end_state_id, end_state, True)
                decoded_words += is_word_decoded
        best_cost = np.inf
        best_backtrack = (-1, -1)
        for hmm_id, hmm in enumerate(self.hmms):
            for i in range(hmm.states_n):
                if best_costs[hmm_id][-1, i] < best_cost:
                    best_cost = best_costs[hmm_id][-1, i]
                    best_backtrack = (hmm_id, i, True)
        restored_path, time = [best_backtrack], -1
        hmm_id, prev_state, _ = best_backtrack
        while backtrack[hmm_id][time, prev_state] != None:
            best_backtrack = backtrack[hmm_id][time, prev_state]
            restored_path.append(best_backtrack)
            hmm_id, prev_state, _ = best_backtrack
            time -= 1
        return list(reversed(restored_path)), best_cost

    def decode(self, observations):
        """
        Для последовательности скрытых состояний находит
        соответствующие ей последовательности звуков и слов.

        Аргументы:
            observations: Последовательность наблюдений.

        Возвращаемое значение:
            Кортеж, содержащий:
                - Вероятность декодированной последовательности.
                - Последовательность декодированных слов.
                - Последовательность декодированных звуков.
        """
        best_path, best_cost = self.__viterbi(observations)
        n = len(best_path)
        sounds_path = []
        words_path = []
        for i in range(n):
            hmm_id, state_id, insert_word = best_path[i]
            sounds_path.append(self.hmms[hmm_id].states[state_id].name)
            if insert_word:
                words_path.append(self.hmms[hmm_id].association)
        return best_cost, words_path, sounds_path


class NBestViterbiDecoder:
    """
    Реализация декодера согласно модифицированному алгоритму Витерби,
    ищущему N лучших последовательностей результатов.
    """

    def __init__(self, hmms, lm):
        """
        Инициализация параметров декодера.

        Аргументы:
            hmm: Предварительно натренированная HMM-GMM модель.
            lm: Предварительно натренированная языковая модель.
        """
        self.hmms = hmms
        self.lm = lm

    def __nbest_viterbi(self, observations, n_best):
        """
        Даны наблюдения. Находит 'n_best' последовательностей скрытых состояний
        для заданных наблюдений 'observations'.

        Аргументы:
            observations: Последовательность наблюдений.
            n_best: Количество последовательностей скрытых сотояний.

        Возвращаемое значение:
            Список кортежей, каждый из которых содержит:
                - Вероятность последовательности скрытых состояний.
                - Последовательность скрытых состояний.
        """
        observations_n = observations.shape[0]
        state_data = [np.full((observations_n, hmm.states_n, n_best), None, dtype=object) for hmm in self.hmms]
        for hmm_id, hmm in enumerate(self.hmms):
            for i in range(hmm.states_n):
                cost = hmm.states[i].initial_probability + hmm.states[i].gmm[observations[0]]
                state_data[hmm_id][0, i, 0] = (cost, hmm_id, None, False, -1)
        active_end_word_states = set()
        decoded_words = np.full(n_best, 1)
        for t in range(1, observations_n):
            for hmm_id, hmm in enumerate(self.hmms):
                for i in range(hmm.states_n):  # cur state
                    kmin = []
                    for j in range(hmm.states_n):  # prev state
                        for l in range(n_best):
                            c = state_data[hmm_id][t - 1, j, l][0] if state_data[hmm_id][
                                                                          t - 1, j, l] is not None else np.inf
                            cost = c + hmm.transitions[j][i] + hmm.states[i].gmm[observations[t]]
                            heapq.heappush(kmin, (cost, j, l))
                    for l in range(n_best):
                        cost, backtrack_ptr, rank = heapq.heappop(kmin)
                        state_data[hmm_id][t, i, l] = (cost, hmm_id, backtrack_ptr, False, rank)
                    if state_data[hmm_id][t, i, 0][0] != np.inf and i == hmm.end_state:
                        active_end_word_states.add((hmm_id, hmm.association, i))
            if active_end_word_states:
                are_words_decoded = np.full(n_best, False)
                for hmm_id, hmm in enumerate(self.hmms):
                    begin_state = hmm.begin_state
                    kmin = []
                    for hmm_end_state_id, prev_word, end_state in active_end_word_states:
                        ngram = NGram.from_words_list((prev_word, hmm.association))
                        for l in range(n_best):
                            cost = state_data[hmm_end_state_id][t, end_state, l][0] + self.lm.get_probability(
                                ngram) * LMSF - decoded_words[l] * np.log(WIP)
                            heapq.heappush(kmin, (cost, hmm_end_state_id, end_state, True, l))
                    data = heapq.heappop(kmin)
                    for l in range(n_best):
                        if data[0] < state_data[hmm_id][t, begin_state, l][0]:
                            are_words_decoded[l] = True
                            heapq.heappush(kmin, state_data[hmm_id][t, begin_state, l])
                            state_data[hmm_id][t, begin_state, l] = data
                            data = heapq.heappop(kmin)
                for l, is_word_decoded in enumerate(are_words_decoded):
                    decoded_words[l] += is_word_decoded
        kmin = []
        for hmm_id, hmm in enumerate(self.hmms):
            for i in range(hmm.states_n):
                for l in range(n_best):
                    cost = state_data[hmm_id][-1, i, l][0]
                    heapq.heappush(kmin, (cost, hmm_id, i, True, l))
        paths = []
        for l in range(n_best):
            cost, hmm_id, back_ptr, _, rank = heapq.heappop(kmin)
            restored_path, time = [(hmm_id, back_ptr, True)], -1
            while state_data[hmm_id][time, back_ptr, rank][2] != None:
                _, hmm_id, back_ptr, hmm_toggle, rank = state_data[hmm_id][time, back_ptr, rank]
                restored_path.append((hmm_id, back_ptr, hmm_toggle))
                time -= 1
            paths.append((cost, list(reversed(restored_path))))

        return paths

    def decode_k(self, observations, n_best):
        """
        Для каждой последовательности скрытых состояний находит соостветствующую ей последовательность слов.

        Аргументы:
            observations: Последовательность наблюдений.
            n_best: Количество последовательностей скрытых сотояний.

        Возвращаемое значение:
            Список кортежей, каждый из которых содержит:
                - Вероятность декодированной последовательности.
                - Последовательность декодированных слов.
                - Последовательность декодированных звуков.
        """
        costs_and_paths = self.__nbest_viterbi(observations, n_best)
        cost_words_sounds = []
        for i, (cost, path) in enumerate(costs_and_paths):
            cost_words_sounds.append((cost, [], []))
            for hmm_id, state_id, insert_word in path:
                cost_words_sounds[i][2].append(self.hmms[hmm_id].states[state_id].name)
                if insert_word:
                    cost_words_sounds[i][1].append(self.hmms[hmm_id].association)
        return cost_words_sounds


class MultipassDecoding:
    """Реализация подхода multipass decoding"""

    def __init__(self, lm, nbest_decoder):
        """
        Инициализация параметров.

        Аргументы:
            lm: Языковая модель, используемая для переоценки вероятностей декодированных результатов.
            nbest_decoder: Декодер, возвращающий 'n_best' последовательностей
                скрытых состояний.
        """
        self.__lm = lm
        self.__nbest_decoder = nbest_decoder

    def decode(self, observations, n_best):
        """
        Метод переоценивает вероятности декодированных результатов с использованием языковой
        модели более высокой размерности (например, 3-грамной или 4-грамной) и получает
        индекс наиболее вероятной последовательности декодированных слов по итогам переоценки.

        Аргументы
            observations: Последовательность наблюдений.
            n_best: Количество последовательностей слов, которые должны быть получены в результате декодирования
                и среди которых после переоценки вероятностей будет выбран наиболее вероятный результат.

        Возвращаемое значение:
            Кортеж, содержащий:
                - Индекс, соответствующий наиболее вероятному результату.
                - 'n_best' последовательностей декодированных результатов.
                - Список переоцененных вероятностей.
        """
        decodings = self.__nbest_decoder.decode_k(observations, n_best)
        probabilities = np.full((len(decodings),), np.inf, dtype=float)
        for i, decoding in enumerate(decodings):
            _, line, _ = decoding
            ngram = NGram(self.__lm.n)
            for word in line:
                ngram.update(word)
                probabilities[i] += self.__lm.get_probability(ngram)
            ngram.update(NGram.get_sys_token('end'))
            ngram_probability = self.__lm.get_probability(ngram)
            probabilities[i] = lmath.log_sum(probabilities[i], ngram_probability)
        best_index = probabilities.argmin()
        return best_index, decodings, probabilities


if __name__ == '__main__':

    # HMMManager.create_trained_three_state_per_phone_model(
    #     hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt15/hmm_training_dataset",
    #     gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt15/gmm_training_dataset",
    #     lexicon_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt15/lexicon.txt",
    # )
    # for hmm in HMMManager.HMMS_CONTAINER.values():
    #     HMMManager.print_hmm_gmm_data(hmm)
    # exit(0)

    np.set_printoptions(linewidth=np.inf)

    hmm0, is_converged0 = HMMManager.create_trained_monophone_model(
        word="включить",
        lexicon=["ф", "к", "ль", "ю", "ч", "и", "ть"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/switch_on"
    )
    # hmm0.set_transitions(np.array([
    #     [-np.log(0.9), -np.log(0.1), np.inf, np.inf, np.inf, np.inf],
    #     [np.inf, -np.log(0.8), -np.log(0.2), np.inf, np.inf, np.inf],
    #     [np.inf, np.inf, -np.log(0.9), -np.log(0.1), np.inf, np.inf],
    #     [np.inf, np.inf, np.inf, -np.log(0.7), -np.log(0.3), np.inf],
    #     [np.inf, np.inf, np.inf, np.inf, -np.log(0.8), -np.log(0.2)],
    #     [np.inf, np.inf, np.inf, np.inf, np.inf, -np.log(1)]
    # ]))
    HMMManager.print_hmm_gmm_data(hmm0, is_converged0)
    hmm1, is_converged1 = HMMManager.create_trained_monophone_model(
        word="лампу",
        lexicon=["л", "а", "м", "п", "у"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/lamp"
    )
    # hmm1.set_transitions(np.array([
    #     [-np.log(0.9), -np.log(0.1), np.inf, np.inf, np.inf],
    #     [np.inf, -np.log(0.8), -np.log(0.2), np.inf, np.inf],
    #     [np.inf, np.inf, -np.log(0.9), -np.log(0.1), np.inf],
    #     [np.inf, np.inf, np.inf, -np.log(0.8), -np.log(0.2)],
    #     [np.inf, np.inf, np.inf, np.inf, -np.log(1)]
    # ]))
    HMMManager.print_hmm_gmm_data(hmm1, is_converged1)

    hmm2, is_converged2 = HMMManager.create_trained_monophone_model(
        word="открыть",
        lexicon=["а", "т", "к", "р", "ы", "ть"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/open"
    )
    HMMManager.print_hmm_gmm_data(hmm2, is_converged2)

    hmm3, is_converged3 = HMMManager.create_trained_monophone_model(
        word="кран",
        lexicon=["к", "р", "а", "н"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/tap"
    )
    HMMManager.print_hmm_gmm_data(hmm3, is_converged3)

    hmm4, is_converged4 = HMMManager.create_trained_monophone_model(
        word="выключить",
        lexicon=["в", "ы", "к", "ль", "ю", "ч", "и", "ть"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/switch_off"
    )
    HMMManager.print_hmm_gmm_data(hmm4, is_converged4)

    hmm5, is_converged5 = HMMManager.create_trained_monophone_model(
        word="шкаф",
        lexicon=["ш", "к", "а", "ф"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/closet"
    )
    HMMManager.print_hmm_gmm_data(hmm5, is_converged5)

    hmm6, is_converged6 = HMMManager.create_trained_monophone_model(
        word="закрыть",
        lexicon=["з", "а", "к", "р", "ы", "ть"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/close"
    )
    HMMManager.print_hmm_gmm_data(hmm6, is_converged6)

    hmm7, is_converged7 = HMMManager.create_trained_monophone_model(
        word="мыть",
        lexicon=["м", "ы", "ть"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/wash"
    )
    HMMManager.print_hmm_gmm_data(hmm7, is_converged7)

    hmm8, is_converged8 = HMMManager.create_trained_monophone_model(
        word="пол",
        lexicon=["п", "о", "л"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/floor"
    )
    HMMManager.print_hmm_gmm_data(hmm8, is_converged8)

    # hmm5, is_converged5 = HMMManager.create_trained_monophone_model(
    #     word="свет",
    #     lexicon=["с", "вь", "е", "т"],
    #     gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.7/sounds",
    #     hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.7/light"
    # )
    # HMMManager.print_hmm_gmm_data(hmm5, is_converged5)

    lm = KatzSmoothingLM.from_train_corpus(
        n=2,
        k=5,
        corpus_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/lm/corpus.txt",
        reserved_probability=0.001
    )
    print(lm.get_probability(NGram.from_words_list(('включить', 'лампу'))))
    print(lm.get_probability(NGram.from_words_list(('включить', 'включить'))))
    print(lm.get_probability(NGram.from_words_list(('лампу', 'лампу'))))
    print(lm.get_probability(NGram.from_words_list(('лампу', 'включить'))))

    decoder = NBestViterbiDecoder(
        hmms=[hmm0, hmm1, hmm2, hmm3, hmm4, hmm5, hmm6, hmm7, hmm8],
        lm=lm
    )
    # mfccs = mfcc("C:/Users/PeterA/Desktop/vkr/test/_____attempt12.6/switch_on_lamp.wav")
    # mfccs = mfcc("C:/Users/PeterA/Desktop/vkr/test/_____attempt12.6/open_tap.wav")
    mfccs = mfcc("C:/Users/PeterA/Desktop/vkr/test/_____attempt12.11/_phrases/close_closet.wav")
    cost_words_sounds = decoder.decode_k(mfccs, 1)
    for c, w, s in cost_words_sounds:
        print(c, w, s)

    multipass_decoder = MultipassDecoding(lm, decoder)
    index, decodings, _ = multipass_decoder.decode(mfccs, 10)
    print(decodings[index])

    exit(0)
    hmm1, is_converged1 = HMMManager.create_trained_monophone_model(
        word="включить",
        lexicon=["ф", "к", "лю", "ч", "и", "ть"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt9/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt9/switch_on"
    )
    HMMManager.print_hmm_data(hmm1, is_converged1)

    hmm2, is_converged2 = HMMManager.create_trained_monophone_model(
        word="свет",
        lexicon=["с", "ве", "т"],
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt9/sounds",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt9/light"
    )
    HMMManager.print_hmm_data(hmm2, is_converged2)

    lm = KatzSmoothingLM.from_train_corpus(
        n=2,
        k=5,
        corpus_path="C:/Users/PeterA/Desktop/vkr/test/_____attempt9/lm/corpus.txt",
        reserved_probability=0.001
    )
    # print(lm.get_probability(NGram.from_words_list(('включи', 'свет'))))
    # print(lm.get_probability(NGram.from_words_list(('включи', 'включи'))))
    # print(lm.get_probability(NGram.from_words_list(('свет', 'включи'))))
    # print(lm.get_probability(NGram.from_words_list(('свет', 'свет'))))

    decoder = ViterbiDecoder(
        hmms=[hmm1, hmm2],
        lm=lm
    )

    # decoder = NBestViterbiDecoder(
    #     hmms=[hmm1, hmm2],
    #     lm=lm
    # )

    mfccs1 = mfcc("C:/Users/PeterA/Desktop/vkr/test/_____attempt9/switch_on_light.wav")


    def printer(mfccs):
        for x in mfccs:
            # words, cost = decoder.decode_k(x, n_best=1)
            words, cost, sounds = decoder.decode(x)
            # words, sounds = decoder.decode_k(x, 10)
            print(words, cost, sounds)
            # for w, s in zip(words, sounds):
            #     print(w)
            #     print(s)
            #     print("===")


    printer([mfccs1])  # mfccs2, mfccs3, mfccs4
    # printer([mfccs1, mfccs2, mfccs3, mfccs4, mfccs5, mfccs6, mfccs7, mfccs8, mfccs9, mfccs10, mfccs11])
    print("text", text)
