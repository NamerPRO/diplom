import logging
import os
import pickle
import time
from enum import Enum
from pathlib import Path

from recognition.decoders.viterbi import ViterbiDecoder, NBestViterbiDecoder, MultipassDecoding
from recognition.models.acoustic.hmm import HMMManager
from recognition.models.language.language import KatzSmoothingLM, NoSmoothingLM, LaplaceSmoothingLM, \
    AdditiveSmoothingLM, SimpleGoodTuringSmoothingLM
from recognition.preprocess.mfcc import mfcc


class Decoder(Enum):
    """
    Перечисление возможных алгоритмов декодирования.
    """
    VITERBI = 0
    N_BEST_VITERBI = 1


class LanguageModel(Enum):
    """
    Перечисление возможных типов языковых моделей.
    """
    NO_SMOOTHING = 0
    LAPLACE_SMOOTHING = 1
    ADDITIVE_SMOOTHING = 2
    GOOD_TURING_SMOOTHING = 3
    KATZ_SMOOTHING_FROM_TRAINING_CORPUS = 4
    KATZ_SMOOTHING_FROM_ARPA_FILE = 5


class AcousticModelInitialization(Enum):
    """
    Перечисление возможных типов инициализации акустической модели.
    """
    FROM_TRAINING_DATA = 0
    FROM_SAVED_FILE = 1


class LogicLayer:
    """
    Логический слой приложения, обрабатывающий бизнес-логику.
    """

    def __init__(self):
        """
        Инициализация экземпляра LogicLayer.
        """
        self.hmms = []
        self.decoder_lm = None
        self.multipass_lm = None
        self.n_best = -1

    def restore_hmms(self, path_to_saved_model):
        """
        Метод получает HMM-GMM модель из сохраненной копии,
        путь до которой содержится в переменной path_to_saved_model.

        Аргументы:
            path_to_saved_model: Путь до сохраненной копии, из которой
                нужно получить HMM-GMM модель.
        """
        with open(path_to_saved_model, "rb") as file:
            self.hmms = pickle.load(file)
        return True

    def load_hmms(self, hmm_dataset_path, gmm_dataset_path):
        """
        Метод получает HMM-GMM модель путем ее тренировки.
        Полученная HMM-GMM модель сохраняется в файл и
        может быть потом из него восстановлена.

        Аргументы:
            hmm_dataset_path: Путь до тренировочных данных, которые
                используются для обучения HMM.
            gmm_dataset_path: Путь до тренировочных данных, которые
                используются для обучения GMM.
        """
        try:
            self.hmms = []
            for hmm_folder in Path(hmm_dataset_path).iterdir():
                if hmm_folder.is_dir():
                    dataset_dir = f"{hmm_folder.resolve()}/dataset"
                    topology = f"{hmm_folder.resolve()}/topology.txt"
                    with open(topology, "r", encoding="utf-8") as topology_file:
                        associated_word, lexicon = topology_file.read().split(" ", 1)
                        hmm, is_training_converged = HMMManager.create_trained_monophone_model(
                            word=associated_word,
                            lexicon=lexicon.split(" "),
                            gmm_dataset_path=gmm_dataset_path,
                            hmm_dataset_path=dataset_dir
                        )
                        if not is_training_converged:
                            logging.warning("HMM training not converged!")
                        self.hmms.append(hmm)
            hmms_save_name = f"./hmm-gmm-save/hmm-gmm_{int(time.time() * 1000)}.pkl"
            os.makedirs("./hmm-gmm-save", exist_ok=True)
            with open(hmms_save_name, "wb") as file:
                pickle.dump(self.hmms, file)
            return True
        except Exception as e:
            logging.error(e)
            return False

    def get_language_model(self, language_model_type, n, k=5, reserved_probability=0.01, addition=5, training_corpus=None, arpa_file_path=None):
        """
        Метод получает объект языковой модели по переданным ему параметрам,
        которые используются для инициализации этой языковой модели. Если
        какой-то из переданных на вход методу параметров не требуется для
        инициализации языковой модели, он игнорируется.

        Аргументы:
            language_model_type: Тип языковой модели, объект которой нужно
                получить.
            n: Размерность языковой модели.
            k: Порог отсечения. Используется в зглаживании по Кацу. Определяет,
                какие n-граммы встречаются достаточно часто, чтобы использовать
                их несглаженные вероятности, а какие подвергнуть сглаживанию.
                По-умолчанию: 5.
            reserved_probability: Вероятностная масса, зарезервированная под
                неизвестные n-граммы. По-умолчанию: 0.01.
            addition: Параметр аддитивного сглаживания. По-умолчанию: 5.
            training_corpus: Путь до тренировочного корпуса текста, на
                котором следует обучить создаваемую языковую модель.
                Может быть None, если модель создается на основе
                ARPA-файла. По-умолчанию: None.
            arpa_file_path: Путь до ARPA-файла, но основе которого нужно
                создать языковую модель. Может быть None, если модель
                создается путем тренировки на тренировочном наборе данных.
        """
        if language_model_type == LanguageModel.NO_SMOOTHING:
            return NoSmoothingLM(n, training_corpus)
        elif language_model_type == LanguageModel.LAPLACE_SMOOTHING:
            return LaplaceSmoothingLM(n, training_corpus)
        elif language_model_type == LanguageModel.ADDITIVE_SMOOTHING:
            return AdditiveSmoothingLM(n, training_corpus, addition)
        elif language_model_type == LanguageModel.GOOD_TURING_SMOOTHING:
            return SimpleGoodTuringSmoothingLM(n, training_corpus)
        elif language_model_type == LanguageModel.KATZ_SMOOTHING_FROM_TRAINING_CORPUS:
            return KatzSmoothingLM.from_train_corpus(n, training_corpus, k, reserved_probability)
        elif language_model_type == LanguageModel.KATZ_SMOOTHING_FROM_ARPA_FILE:
            return KatzSmoothingLM.from_arpa_file(arpa_file_path)

    def decode(self, path_to_wav):
        """
        Метод, запускающий процесс декодирования содержимого
        аудио-файла, расположенного по пути path_to_wav, и
        возвращающий полученный результат.

        Аргументы:
            path_to_wav: Путь до аудио-файла, который следует
                декодировать.
        """
        observations = mfcc(path_to_wav)
        if self.multipass_lm is None:
            decoder = ViterbiDecoder(self.hmms, self.decoder_lm)
            return decoder.decode(observations)
        else:
            decoder = MultipassDecoding(self.multipass_lm, NBestViterbiDecoder(self.hmms, self.decoder_lm))
            return decoder.decode(observations, self.n_best)

    def get_perplexity(self, corpus_path):
        """
        Метод, запускающий вычисление метрик Perplexity для
        языковой модели в декодере и, если используется подход
        multipass decoding, для языковой модели, используемой при
        переоценке вероятностей. Если подход multipass decoding не
        используется, perplexity2=None.

        Аргументы:
            corpus_path: Путь до корпуса текста, по которому вычисляется
                значение метрики Perplexity.
        """
        perplexity1 = self.decoder_lm.perplexity(corpus_path)
        if self.multipass_lm is not None:
            perplexity2 = self.multipass_lm.perplexity(corpus_path)
        else:
            perplexity2 = None
        return perplexity1, perplexity2

    def get_oov(self, corpus_path):
        """
        Метод, запускающий вычисление метрик Out-Of-Vocabulary для
        языковой модели в декодере и, если используется подход
        multipass decoding, для языковой модели, используемой при
        переоценке вероятностей. Если подход multipass decoding не
        используется, oov2=None.

        Аргументы:
            corpus_path: Путь до корпуса текста, по которому вычисляется
                значение метрики Out-Of-Vocabulary.
        """
        oov1 = self.decoder_lm.oov_rate(corpus_path)
        if self.multipass_lm is not None:
            oov2 = self.multipass_lm.oov_rate(corpus_path)
        else:
            oov2 = None
        return oov1, oov2

