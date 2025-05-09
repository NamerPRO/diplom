import logging
from pathlib import Path

from matplotlib import pyplot as plt
from scipy.stats import norm

import utils.log_math as lmath
from recognition.models.acoustic.gmm import GMM
from recognition.models.acoustic.hmm_state import HMMState
from recognition.preprocess.mfcc import *


class HMM:
    """
    Класс представляющий скрытую марковскую модель
    """

    def __init__(self, association, states, transition_probabilities=None):
        """
        Инициализирует HMM, выполняет валидацию параметров.

        Аргументы:
            states: Список состояний HMM.
            transition_probabilities: Матрица переходов между состояниями.
        """
        self.__eps = 1e-7
        self.__states = states
        self.__states_n = len(states)

        states[0].initial_probability = 0
        for i in range(1, self.__states_n):
            states[i].initial_probability = np.inf

        if transition_probabilities is not None:
            if len(transition_probabilities) != self.__states_n:
                raise ValueError("Матрица transition_probabilities должна быть размера N*N, где N есть число состояний.")
            for line in transition_probabilities:
                line_sum = line[0]
                for i in range(1, len(line)):
                    line_sum = lmath.log_sum(line_sum, line[i])
                if np.abs(line_sum) > self.__eps:
                    raise ValueError(
                        "Матрица transition_probabilities состоит из некорректных логарифмических вероятностей. Каждая строка должна суммироваться в 0.")
            self.__transitions = transition_probabilities
        else:
            self.__transitions = np.full((self.states_n, self.states_n), np.inf)
            for i in range(self.states_n - 1):
                self.__transitions[i, i] = -np.log(0.5)
                self.__transitions[i, i + 1] = -np.log(0.5)
            self.__transitions[self.states_n - 1, self.states_n - 1] = 0

        self.association = association
        self.begin_state = -1
        self.end_state = -1
        for id, state in enumerate(self.states):
            if state.is_start_state:
                self.begin_state = id
            if state.is_final_state:
                self.end_state = id
        if self.begin_state == -1 or self.end_state == -1:
            raise ValueError("Ожидалось, что у HMM будут начальное и конечное состояния.")

    @property
    def states_n(self):
        """
        Возвращает states_n.
        """
        return self.__states_n

    @property
    def states(self):
        """
        Возвращает states.
        """
        return self.__states

    @property
    def transitions(self):
        """
        Возвращает transitions.
        """
        return self.__transitions

    def set_transitions(self, transitions):
        """
        Установливает матрицу переходов между
        состояниями transitions.
        """
        self.__transitions = transitions

    def __alpha(self, observations):
        """
        Для каждого момента времени 't' и каждого скрытого состояния 'j'
        вычисляет вероятности нахождения в скрытом состоянии 'j' в момент
        времени 't' с учетом всех наблюдений, которые были до этого.

        Формально:
            α_t(j)=P(o_1, o_2, ..., o_t, q_t=j|λ) ∀t, j

        Аргументы:
            observations: Двумерный массив numpy формы '(N,M)', где N — количество наблюдений,
                а M размерность наблюдений. Каждое наблюдение представлено массивом признаков MFCC.

        Возвращаемое значение:
            Вычисленное α_t(j) для каждого 't' и 'j'.
        """
        T, N = observations.shape[0], self.states_n

        alphas = np.full((T, N), np.inf, dtype=np.float64)

        for j in range(N):
            alphas[0, j] = self.__states[j].initial_probability + self.__states[j].gmm[observations[0]]

        for t in range(1, T):
            for j in range(N):
                for i in range(N):
                    alphas[t, j] = lmath.log_sum(alphas[t, j],
                                                 alphas[t - 1, i] + self.transitions[i][j] + self.__states[j].gmm[
                                                     observations[t]])

        return alphas

    def __beta(self, observations):
        """
        Для каждого момента времени 't' и каждого скрытого состояния 'j'
        вычисляет вероятности нахождения в скрытом состоянии 'j' в момент
        времени 't', зная будущие наблюдения, но не зная наблюдений в прошлом.

        Формально:
            β_t(j)=P(o_{t+1}, o_{t+2}, ..., o_T|q_t=j, λ) ∀t, j

        Аргументы:
            observations: Двумерный массив numpy формы '(N,M)', где N — количество наблюдений,
                а M размерность наблюдений. Каждое наблюдение представлено массивом признаков MFCC.

        Возвращаемое значение:
            Вычисленное β_t(j) для каждого 't' и 'j'.
        """
        T, N = observations.shape[0], self.states_n

        betas = np.full((T, N), np.inf, dtype=np.float64)

        for j in range(N):
            betas[-1, j] = 0

        for t in reversed(range(T - 1)):
            for j in range(N):
                for i in range(N):
                    betas[t, j] = lmath.log_sum(betas[t, j],
                                                betas[t + 1, i] + self.transitions[j][i] + self.__states[i].gmm[
                                                    observations[t + 1]])

        return betas

    def __gamma(self, alphas, betas):
        """
        Для каждого момента времени 't' и каждого скрытого состояния 'j'
        вычисляет вероятности нахождения в скрытом состоянии 'j' в момент
        времени 't', зная наблюдения, которые были до, и наблюдения, которые
        будут после.

        Формально:
            γ_t(j)=P(q_t=j|o, λ) ∀t, j

        Аргументы:
            alphas: Данные, полученные в результате вызова метода 'alpha'.
            betas: Данные, полученные в результате вызова метода 'beta'.

        Возвращаемое значение:
            Вычисленное γ_t(j) для каждого 't' и 'j'.
        """
        T, N = alphas.shape[0], self.states_n

        gammas = np.full((T, N), np.inf, dtype=np.float64)
        for t in range(T):
            denom_prob = np.inf
            for j in range(N):
                denom_prob = lmath.log_sum(denom_prob, alphas[t, j] + betas[t, j])
            for j in range(N):
                gammas[t, j] = alphas[t, j] + betas[t, j] - denom_prob

        return gammas

    def __ksi(self, alphas, betas, observations):
        """
        Для каждого 't', 'i', 'j' вычисляет вероятность
        нахождения в скрытом состоянии 'i' в момент времени 't' и в скрытом
        состоянии 'j' в момент времени 't+1' для заданного набора наблюдений.

        Формально:
            ξ_t(i,j)=P(q_t=i,q_{t+1}=j|o, λ) ∀t, i, j

        Аргументы:
            alphas: Данные, полученные в результате вызова метода 'alpha'.
            betas: Данные, полученные в результате вызова метода 'beta'.
            observations: Двумерный массив numpy формы '(N,M)', где N — количество наблюдений,
                а M размерность наблюдений. Каждое наблюдение представлено массивом признаков MFCC.

        Возвращаемое значение:
            Вычисленное ξ_t(i,j) для каждого 't', 'i' и 'j'.
        """
        T, N = observations.shape[0], self.states_n

        ksis = np.full((T, N, N), np.inf, dtype=np.float64)

        for t in range(T - 1):
            denom_prob = np.inf
            for k in range(N):
                for w in range(N):
                    denom_prob = lmath.log_sum(denom_prob, alphas[t, k] + self.transitions[k][w] + self.__states[w].gmm[
                        observations[t + 1]] + betas[t + 1, w])
            for i in range(N):
                for j in range(N):
                    ksis[t, i, j] = alphas[t, i] + self.transitions[i][j] + self.__states[j].gmm[observations[t + 1]] + \
                                    betas[t + 1, j] - denom_prob

        return ksis

    def __gmm_gammas(self, observations, gammas):
        """
        Для каждого 'i', 'l', 't' вычисляет вероятность того,
        что наблюдение в момент времени 't' относится к 'l'-ому
        компоненту 'i'-ой модели гауссовских смесей.

        Формально:
            γ_{il}(t)=P(q_t=i,X_{it}=l|o, λ) ∀i, l, t,
                где X_{it} случайная величина, указывающая на компонент смеси в момент времени 't' для состояния 'i'.

        Возвращаемое значение:
            Вычисленное γ_{il}(t) для каждого 'i', 'l' и 't'.
        """
        observations_n = observations.shape[0]
        n_components = self.__states[0].gmm.n_components
        gmm_gammas = np.zeros((self.__states_n, n_components, observations_n))
        for i in range(self.__states_n):
            gmm_i = self.__states[i].gmm
            for l in range(n_components):
                for t in range(observations_n):
                    gmm_gammas[i][l][t] = gammas[t][i] + gmm_i.log_lth_gaussian_prob(observations[t], l) - gmm_i[
                        observations[t]]
        return gmm_gammas

    def __converge_criteria(self, alphas):
        """
        Метод вычисляет число являющееся критерием сходимости
        алгоритма Баума-Велша. Если изменение такого числа на
        текущем шаге незначительно отличается от числа на
        предыдущем, то считается, что алгоритм сошелся.

        Аргументы:
            alphas: Данные, полученные в результате вызова
                метода 'alpha'.

        Возвращаемое значение:
            Число, используемое в дальнейшем как критерий сходимости.
        """
        return np.sum(lmath.log_sum_arr(alphas[-1, :]))

    def baum_welch(self, observations, eps=1e-4, max_iters=1000, ignore_eps=False, train_gmm=False):
        """
        Обучение скрытой марковской модели согласно алгоритму Баума-Велша.

        Аргументы:
            observations: Двумерный массив numpy формы '(N,M)', где N — количество наблюдений,
                а M размерность наблюдений. Каждое наблюдение представлено массивом признаков MFCC.
            eps: Величина, на которую должны отличаться друг относительно друга параментры HMM модели,
                чтобы считалось, что алгоритм сошелся. По-умолчанию: 1e-4.
            max_iters: Предельное число итераций, которым ограничен алгоритм. По-умолчанию: 1000.
            ignore_eps: Принудительно выполнить предельное число итераций max_iterations, игнорируя
                параметр eps. По-умолчанию: False,
            train_gmm: Обучать GMM внутри алгоритма Баума-Велша. По-умолчанию: False

        Возвращаемое значение:
            False, если было выполнено предельное число итераций. Иначе True.
        """
        observations_n = observations.shape[0]
        prev_likelihood = np.inf
        cur_likelihood = np.inf

        for it in range(max_iters):
            # is_converged = True

            alphas = self.__alpha(observations)
            betas = self.__beta(observations)
            gammas = self.__gamma(alphas, betas)
            ksis = self.__ksi(alphas, betas, observations)

            new_transitions = np.ndarray((self.__states_n, self.__states_n))

            for i in range(self.__states_n):
                self.__states[i].initial_probability = gammas[0, i]

                denom_gamma = lmath.log_sum_arr(gammas[:-1, i])
                for j in range(self.__states_n):
                    new_transitions[i, j] = lmath.log_sum_arr(ksis[:-1, i, j]) - denom_gamma

                if train_gmm:
                    gmm_gammas = self.__gmm_gammas(observations, gammas)
                    gmm_i = self.__states[i].gmm
                    for l in range(gmm_i.n_components):
                        mmax1, mmax2 = max(gmm_gammas[i][l]), max(gammas[:, i])
                        lse1, lse2 = self.__eps, self.__eps
                        for t in range(0, observations_n):
                            if gmm_gammas[i][l][t] != np.inf and mmax1 != np.inf:
                                lse1 += np.exp(gmm_gammas[i][l][t] - mmax1)
                            if gammas[t][i] != np.inf and mmax2 != np.inf:
                                lse2 += np.exp(gammas[t][i] - mmax2)
                        lse1 = mmax1 + np.log(lse1)
                        lse2 = mmax2 + np.log(lse2)
                        gmm_i.c[l] = 1 if lse1 == lse2 == np.inf else np.exp(lse1 - lse2)
                        updated_gmm_means = np.float64(0)
                        dim = observations.shape[1]
                        updated_gmm_covmatrices = np.zeros((dim, dim))
                        for t in range(observations_n):
                            m1 = 1 if gmm_gammas[i][l][t] == lse1 == np.inf else np.exp(gmm_gammas[i][l][t] - lse1)
                            updated_gmm_means += m1 * observations[t]
                            updated_gmm_covmatrices += m1 * (observations[t] - gmm_i.means[l])[:, np.newaxis] * (
                                        observations[t] - gmm_i.means[l])
                        # is_converged &= np.linalg.norm(updated_gmm_means - gmm_i.means[l]) < eps and np.linalg.norm(
                        #     updated_gmm_covmatrices - gmm_i.covmatrices[l]) < eps
                        gmm_i.means[l] = updated_gmm_means
                        gmm_i.covmatrices[l] = updated_gmm_covmatrices + 1e-5 * np.eye(dim)

            cur_likelihood = self.__converge_criteria(alphas)
            self.__transitions = new_transitions
            if not ignore_eps and np.abs(cur_likelihood - prev_likelihood) < eps:
                return True
            prev_likelihood = cur_likelihood

        logging.warning(
            f"Алгоритм не сходится! Тек/Пред вероятность: {cur_likelihood}/{prev_likelihood}. Дельта: {cur_likelihood - prev_likelihood}.")
        return False


class HMMManager:
    """
    Класс, предоставляющий методы создания и визуализации
    HMM-GMM моделей (объекты класса HMM).
    """

    GMMS_CONTAINER = {}

    def __init__(self):
        """
        Класс утилитарный. Создавать его объект нельзя.
        """
        raise TypeError("Объект этого класса не должен быть создан.")

    @staticmethod
    def create_trained_monophone_model(word, lexicon, gmm_dataset_path, hmm_dataset_path, states_per_phone=1):
        """
        Создает натренированную HMM-GMM модель, где данные для для тренировки
        GMM расположены по пути gmm_dataset_path, а данные для тренировки
        HMM расположены по пути hmm_dataset_path. Слово, с которым должна
        ассоциироваться созданная модель, указано в 'word', а поседовательность
        звуков, из которых состоит 'word' есть в 'lexicon'.

        Аргументы:
            word: Слово, с которым ассоциируется созданная модель.
            lexicon: Последовательность звуков, из которых состоит 'word'.
            gmm_dataset_path: Путь до данных, на которых должна обучаться GMM.
            hmm_dataset_path: Путь до данных, на которых должна обучаться HMM.
            phones_count: Количество состояний, на которых разбит один звук.
                По-умолчанию: 1.

        Возвращаемое значение:
            Кортеж, содержащий:
                - Натренированную HMM-GMM модель.
                - True, если алгоритм тренировки сошелся. Иначе False.
        """
        hmm_states = []
        for phone in lexicon:
            for i in range(states_per_phone):
                full_phone = f"{phone}_{i}"
                if full_phone not in HMMManager.GMMS_CONTAINER:
                    gmm_observations = np.empty((0, 13))
                    folder_path = f"{gmm_dataset_path}/{full_phone}"
                    for samples in Path(folder_path).iterdir():
                        observations = mfcc(samples.resolve())
                        gmm_observations = np.vstack((gmm_observations, observations))
                    HMMManager.GMMS_CONTAINER[full_phone] = GMM(n_components=1, observations=gmm_observations,
                                                                means_init="kmeans")
                    HMMManager.GMMS_CONTAINER[full_phone].train()
                gmm = HMMManager.GMMS_CONTAINER[full_phone]
                hmm_states.append(HMMState(name=full_phone, gmm=gmm, initial_probability=-np.log(1 / len(lexicon)),
                                           is_start_state=phone == lexicon[0], is_final_state=phone == lexicon[-1]))
        hmm_observations = np.empty((0, 13))
        for data in Path(hmm_dataset_path).iterdir():
            data_path = str(data)
            observations = mfcc(data_path)
            hmm_observations = np.vstack((hmm_observations, observations))
        hmm = HMM(word, hmm_states)
        has_converged = hmm.baum_welch(hmm_observations, max_iters=100, ignore_eps=False, train_gmm=False)
        return hmm, has_converged

    @staticmethod
    def get_hmm_gmm_data(hmm, is_converged=None, print_gmm_info=False, are_likelihoods_log=False):
        """
        Получает и возвращает информацию об HMM модели и связанных с ней GMM.

        Аргументы:
            hmm: HMM, о которой нужно вывести информацию.
            is_converged: Если передано, будет выведена дополнительная первая строка с
                информацией о том, сошелся ли алгоритм. Инициализируется результатом работы
                алгоритма Баума-Велча. По-умолчанию: None.
            print_gmm_info: Если True, выводит информацию о GMM. Иначе не выводит.
                По-умолчанию: False.
            are_likelihoods_log: Если True, все вероятности в выводе логарифмические.
                По-умолчанию: False.

        Возвращаемое значение:
            Информация об HMM модели и связанных с ней GMM.
        """
        hmm_data = f"HMM-{hmm.association}:\n"
        if is_converged is not None:
            hmm_data += "Обучение сошлось\n" if is_converged else "Обучение не сошлось\n"
        hmm_data += f"Состояния: {hmm.states_n}\n"
        initial_probabilities = [x.initial_probability for x in hmm.states] if are_likelihoods_log else [
            np.exp(-x.initial_probability) for x in hmm.states]
        hmm_data += f"Начальные вероятности: {initial_probabilities}\n"
        hmm_data += f"Матрица переходов:\n{hmm.transitions if are_likelihoods_log else np.exp(-hmm.transitions)}\n"
        if print_gmm_info:
            hmm_data += "GMM:\n===\n"
            for i in range(hmm.states_n):
                hmm_data += f"GMM номер {i}: {hmm.states[i].gmm.n_components} компонентов\n"
                hmm_data += f"Весовые коэффициенты:\n{hmm.states[i].gmm.c}\n"
                hmm_data += f"Математические ожидания:\n{hmm.states[i].gmm.means}\n"
                hmm_data += f"Ковариационные матрицы:\n{hmm.states[i].gmm.covmatrices}\n===\n"
        return f"{hmm_data}\n"

    @staticmethod
    def print_hmm_gmm_data(hmm, is_converged=None, print_gmm_info=False, are_likelihoods_log=False):
        """
        Выводит информацию об HMM модели и связанных с ней GMM.

        Аргументы:
            hmm: HMM, о которой нужно вывести информацию.
            is_converged: Если передано, будет выведена дополнительная первая строка с
                информацией о том, сошелся ли алгоритм. Инициализируется результатом работы
                алгоритма Баума-Велча. По-умолчанию: None.
            print_gmm_info: Если True, выводит информацию о GMM. Иначе не выводит.
                По-умолчанию: False.
            are_likelihoods_log: Если True, все вероятности в выводе логарифмические.
                По-умолчанию: False.
        """
        print(HMMManager.get_hmm_gmm_data(hmm, is_converged, print_gmm_info, are_likelihoods_log))

    @staticmethod
    def plot_gmm_for_1d_case(hmm, observations):
        """
        Визуализирует каждую GMM, отображая их на разных графиках.
        Каждый график GMM состоит из своих компонентов и наблюдений.
        Примечание: метод не проверяет, сошлось ли обучение.

        Аргументы:
            hmm: HMM, GMM которой нужно визуализировать.
            observations: Массив одномерных наблюдений.
        """
        n, dim = observations.shape
        if dim != 1:
            raise ValueError("Visualization only works for 1d case, but %dd observations found." % dim)
        for i in range(hmm.states_n):
            gmm = hmm.states[i].gmm
            plt.scatter(observations, np.zeros((n,)), c="black")
            arange = np.arange(-20, 20, 0.001)
            for j in range(gmm.n_components):
                plt.plot(arange, norm.pdf(arange, gmm.means[j], np.sqrt(gmm.covmatrices[j][0])))
            plt.show()


if __name__ == '__main__':
    hmm, has_converged = HMM.create_trained_monophone_model(
        gmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/ss/",
        hmm_dataset_path="C:/Users/PeterA/Desktop/vkr/test/ssw/"
    )
    hmm.print_hmm_data(has_converged)

    # hmm_states = [
    #     HMMState(
    #         name="а",
    #         gmm=GMM(
    #             n_components=1,
    #             observations=np.array([]),
    #             means_init='kmeans'
    #         ),
    #         initial_probability=-np.log(1 / 42),
    #     )
    # ]

    # hmm, observations = HMM.make_test_1d_model(
    #     states_n=1,
    #     gmm_means_init="random"
    # )
    # has_converged = hmm.baum_welch(
    #     observations=observations,
    #     eps=0.0001,
    #     max_iters=1000,
    #     ignore_eps=False,
    #     train_gmm=True
    # )
    # hmm.print_hmm_data(has_converged)
    # hmm.plot_gmm_for_1d_case(observations)
