import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

import recognition.metrix.wer as wer
import utils.substring_searcher as searcher
from app.logic import Decoder, LanguageModel, AcousticModelInitialization
from recognition.decoders import viterbi
from recognition.models.acoustic.hmm import HMMManager


class Controller:
    """
    Класс контроллера для управления главным окном приложения.
    """

    def __init__(self, ui, logic):
        """
        Инициализация контроллера.

        Аргмуенты:
            ui: Объект пользовательского интерфейса.
            logic: Объект, реализующий бизнес-логику приложения.
        """
        self.ui = ui
        self.logic = logic

        self.ui.acoustic_model_box.currentIndexChanged.connect(self.on_load_acoustic_model_option_change)
        self.ui.decoding_box.currentIndexChanged.connect(self.on_decoding_option_change)
        self.ui.decoder_language_model_box.currentIndexChanged.connect(self.on_decoder_lm_option_change)
        self.ui.multipass_language_model_box.currentIndexChanged.connect(self.on_multipass_lm_option_change)

        self.ui.save_settings_button.clicked.connect(self.on_save_settings_button_clicked)
        self.ui.recognize_audio_button.clicked.connect(self.on_recognize_audio_button_clicked)
        self.ui.compute_perplexity_button.clicked.connect(self.on_perplexity_button_clicked)
        self.ui.compute_wer_button.clicked.connect(self.on_wer_button_clicked)
        self.ui.compute_oov_button.clicked.connect(self.on_oov_button_clicked)

        self.ui.hmm_gmm_list_widget.currentRowChanged.connect(self.hmm_gmm_list_widget_click)
        self.ui.decoding_list_widget.currentRowChanged.connect(self.decoding_list_widget_click)

        self.recognition_in_progress = False
        self.perplexity_computation_in_progress = False
        self.wer_computation_in_progress = False
        self.oov_computation_in_progress = False

    class RecognizeAudioWorker(QThread):
        """
        Фоновый поток для выполнения операции распознования речи
        и поиска заданной последовательности данных в распознанном
        результате.

        Сигналы:
            decoding_done: Сигнал, испускаемый при завершении работы потока.
        """
        decoding_done = pyqtSignal(tuple, str, str, bool)

        def __init__(self, logic, path_to_wav, search_string):
            """
            Инициализация потока.

            Аргументы:
                logic: Объект, реализующий бизнес-логику приложения.
                path_to_wav: Путь до аудио-файла формата wav с содержимым,
                    которое необходимо распознать.
                search_string: Последовательность данных, наличие которой
                    требуется проверить в распознанном результате.
            """
            super().__init__()
            self.logic = logic
            self.path_to_wav = path_to_wav
            self.search_string = search_string

        def run(self):
            """
            Метод, выполняющий задачу распознования содержимого аудио-файла
            и проверку наличия заданной последовательности данных в распознанном
            результате в отдельном потоке. Он автоматически вызывается при запуске
            потока методом 'start()'.
            """
            result = self.logic.decode(self.path_to_wav)
            if self.logic.multipass_lm is None:
                word_sequence = " ".join(result[1])
            else:
                best_index, decodings, _ = result
                word_sequence = " ".join(decodings[best_index][1])
            is_substring_found = searcher.substring_search(word_sequence, self.search_string)
            self.decoding_done.emit(result, word_sequence, self.search_string, is_substring_found)

    def validation_before_recognition(self):
        """
        Метод, выполняющий предварительную валидацию перед запуском операции
        проверки наличия заданной последовательности данных в аудиопотоке.
        """
        if self.recognition_in_progress:
            self.ui.show_alert(
                "В настоящий момент уже запущена процедура распознования. Пожалуйста, дождитесь ее завершения. Затем повторите попытку.")
            return False
        if not self.ui.is_ready_to_recognize:
            self.ui.show_alert(
                "Модель не настроена. Перейдите во вкладку \"Настройки\", чтобы сделать это. Затем повторите попытку.")
            return False
        path_to_wav = self.ui.get_text(self.ui.main_selected_decode_file)
        if not path_to_wav:
            self.ui.show_alert("Путь до WAV-файла не указан. Выберите WAV-файл и повторите попытку.")
            return False
        search_string = self.ui.get_text(self.ui.main_search_substring_input).strip().lower()
        if not search_string:
            self.ui.show_alert(
                "Искомая последовательность данных не задана. Пожалуйста, задайте ее. Затем повторите попытку.")
            return False
        return True, path_to_wav, search_string

    def on_recognize_audio_button_clicked(self):
        """
        Метод вызывется при нажатии кнопки "Проверить наличие фразы" из главной
        вкладки приложения. В методе запускается операция распознования содержимого
        аудио файла и проверка наличия данных в распознанном результате в отдельном
        потоке.
        """
        data = self.validation_before_recognition()
        if data is False:
            return
        self.worker = Controller.RecognizeAudioWorker(self.logic, data[1], data[2])
        self.worker.decoding_done.connect(self.on_recognition_finished)
        self.ui.main_decoding_result_area.setText(
            "Выполняется распознование. Когда оно завершится, в этом поле отобразится результат.")
        self.recognition_in_progress = True
        self.worker.start()

    def on_recognition_finished(self, decoding_response, decoded_word_sequence, search_string, is_search_string_found):
        """
        Метод вызывается в главном потоке по завершении операций распознования речи и проверки наличия
        заданной последовательности данных в распознанном резульате. В нем происходит обновление
        пользовательского интерфейса с выводом распознанной последовательности слов, искомой
        последовательности данных, а также вердикта: содержится ли искомая последовательность
        данных в распознанной последовательности слов или нет.

        Аргументы:
            decoding_response: Результат работы декодера. Содержит последовательности декодированных звуков, слов,
                вероятности этих последовательностей и, если используется подход multipass decoding, вероятности,
                переоцененные языковой моделью более высокого порядка.
            decoded_word_sequence: Строка, представляющая декодированную последовательность слов.
            search_string: Искомая последовательность данных.
            is_search_string_found: True, если искомая последовательность данных содержится в распознанном результате.
                False, в противном случае.
        """
        self.ui.main_decoding_result_area.setText(
            f"Декодированная последовательность слов:\n{decoded_word_sequence}\n\nИскомая подследовательность данных:\n{search_string}\n\nИскомая последовательность {"" if is_search_string_found else "не "}содержится в декодированном результате.")
        self.fill_decoding_tab(decoding_response)
        self.recognition_in_progress = False

    class SettingsSaveWorker(QThread):
        """
        Фоновый поток для выполнения операций обучения акустической
        и языковой (или языковых при подходе multipass decoding) моделей.

        Сигналы:
            settings_saved: Сигнал, испускаемый при завершении работы потока.
        """
        settings_saved = pyqtSignal(tuple)

        def __init__(self, logic, hmm_gmm_initialization_type, hmm_dataset_path=None, gmm_dataset_path=None,
                     path_to_saved_model=None, decoder_lm_extracted_data=None, multipass_lm_extracted_data=None,
                     progress=None):
            """
            Инициализация потока.

            Аргументы:
                logic: Объект, реализующий бизнес-логику приложения.
                hmm_gmm_initialization_type: Тип инициализации HMM-GMM модели. HMM-GMM модель
                    можно либо получить из сохраненной копии (AcousticModelInitialization.FromSavedFile),
                    либо обучить на тренировочных данных (AcousticModelInitialization.FromTrainingData).
                hmm_dataset_path: Путь до набора данных для обучения HMM. Может быть None, если модель
                    требуется получить из сохраненной копии.
                gmm_dataset_path: Путь до набора данных для обучения GMM. Может быть None, если модель
                    требуется получить из сохраненной копии.
                path_to_saved_model: Путь до сохраненной копии, из которой следует восстановить HMM-GMM
                    модель. Может быть None, если модель требуется обучить на тренировочных данных.
                decoder_lm_extracted_data: Данные о языковой модели, которая должна использоваться
                    в декодере.
                multipass_lm_extracted_data: Данные о языковой модели, которая должна использоваться
                    при переоценке вероятностей в подходе multipass decoding.
                progress: Объект окна ожидания, которое появляется при сохранении настроек модели из
                    вкладки "Настройки".
            """
            super().__init__()

            self.logic = logic
            self.hmm_gmm_initialization_type = hmm_gmm_initialization_type
            self.hmm_dataset_path = hmm_dataset_path
            self.gmm_dataset_path = gmm_dataset_path
            self.path_to_saved_model = path_to_saved_model
            self.decoder_lm_extracted_data = decoder_lm_extracted_data
            self.multipass_lm_extracted_data = multipass_lm_extracted_data
            self.progress = progress

        def run(self):
            """
            Метод, выполняющий задачу обучения HMM-GMM модели, а также языковой
            модели (или языковых моделей при подходе multipass decoding) в отдельном
            потоке. Он автоматически вызывается при запуске потока методом 'start()'.
            """
            if self.hmm_gmm_initialization_type == AcousticModelInitialization.FROM_TRAINING_DATA:
                response = self.logic.load_hmms(self.hmm_dataset_path, self.gmm_dataset_path)
            else:
                response = self.logic.restore_hmms(self.path_to_saved_model)
            if response:
                self.logic.decoder_lm = self.logic.get_language_model(*self.decoder_lm_extracted_data)
                if self.logic.decoder_lm.n != 2:
                    self.settings_saved.emit((False, self.progress))
                    return
                if self.multipass_lm_extracted_data is None:
                    self.logic.multipass_lm = None
                else:
                    self.logic.multipass_lm = self.logic.get_language_model(*self.multipass_lm_extracted_data)
            self.settings_saved.emit((response, self.progress))

    def validate_hmm_gmm_settings(self, progress):
        """
        Метод выполняющий валидацию настроек, относящихся к HMM-GMM модели.

        Аргументы:
            progress: Объект окна ожидания, которое появляется при сохранении настроек модели из
                вкладки "Настройки".
        """
        load_hmm_gmm_type = AcousticModelInitialization(self.ui.acoustic_model_box.currentIndex())
        if load_hmm_gmm_type == AcousticModelInitialization.FROM_TRAINING_DATA:
            hmm_dataset_path = self.ui.get_text(self.ui.hmm_dataset_path)
            gmm_dataset_path = self.ui.get_text(self.ui.gmm_dataset_path)
            if not hmm_dataset_path or not gmm_dataset_path:
                progress.close()
                self.ui.show_alert("Пожалуйста, укажите пути до наборов данных для обучения HMM-GMM модели.")
                return False
            return True, load_hmm_gmm_type, hmm_dataset_path, gmm_dataset_path, None
        else:
            path_to_saved_model = self.ui.get_text(self.ui.saved_hmm_gmm_model)
            if not path_to_saved_model:
                progress.close()
                self.ui.show_alert("Пожалуйста, укажите путь до сохраненной HMM-GMM модели.")
                return False
            return True, load_hmm_gmm_type, None, None, path_to_saved_model

    def validate_lm_settings(self, progress):
        """
        Метод выполняющий валидацию настроек, относящихся к языковым моделям.

        Аргументы:
            progress: Объект окна ожидания, которое появляется при сохранении настроек модели из
                вкладки "Настройки".
        """
        decoder_language_model_type = LanguageModel(self.ui.decoder_language_model_box.currentIndex())
        decoder_lm_extracted_data = self.extract_language_model_init_data(decoder_language_model_type,
                                                                          self.ui.decoder_lm_widgets, progress)
        if not decoder_lm_extracted_data:
            return False
        if Decoder(self.ui.decoding_box.currentIndex()) == Decoder.N_BEST_VITERBI:
            multipass_language_model_type = LanguageModel(self.ui.multipass_language_model_box.currentIndex())
            multipass_lm_extracted_data = self.extract_language_model_init_data(multipass_language_model_type,
                                                                                self.ui.multipass_lm_widgets,
                                                                                progress)
            if not multipass_lm_extracted_data:
                return False
            return True, decoder_lm_extracted_data, multipass_lm_extracted_data
        return True, decoder_lm_extracted_data, None

    def validate_other_services_running(self):
        """
        Метод, проверяющий, что никакие задачи не используют текущую модель.
        Модель нельзя изменить (то есть сохранить новые настройки), если она
        в настоящий момент задействована.
        """
        if self.recognition_in_progress:
            self.ui.show_alert(
                "В настоящий момент невозможно изменить настройки, так как модель сейчас распознает содержимое аудио-файла.")
            return False
        if self.perplexity_computation_in_progress:
            self.ui.show_alert(
                "В настоящий момент невозможно изменить настройки, так как сейчас вычисляется метрика Perplexity.")
            return False
        if self.wer_computation_in_progress:
            self.ui.show_alert(
                "В настоящий момент невозможно изменить настройки, так как сейчас вычисляется метрика Word Error Rate.")
            return False
        if self.oov_computation_in_progress:
            self.ui.show_alert(
                "В настоящий момент невозможно изменить настройки, так как сейчас вычисляется метрика Out-Of-Vocabulary.")
            return False
        return True

    def on_save_settings_button_clicked(self):
        """
        Метод, вызываемый при нажатии кнопки "Сохранить настройки" из вкладки
        "Настройки" приложения. В методе выполняется применение заданных пользователем
        настроек. В отдельном потоке запускается операция обучения акустической
        и языковой (или языковых при подходе multipass decoding) моделей.
        """
        if not self.validate_other_services_running():
            return
        progress = self.ui.show_progress_dialog(text="Настройки применяются...", title=" ")
        try:
            self.ui.is_ready_to_recognize = False
            hmm_gmm_response = self.validate_hmm_gmm_settings(progress)
            if hmm_gmm_response is False:
                return
            lm_response = self.validate_lm_settings(progress)
            if lm_response is False:
                return

            self.logic.n_best = int(self.ui.get_text(self.ui.n_best))
            viterbi.LMSF = int(self.ui.get_text(self.ui.lmsf))
            viterbi.WIP = float(self.ui.get_text(self.ui.wip))

            self.settings_worker = Controller.SettingsSaveWorker(self.logic, *hmm_gmm_response[1:], *lm_response[1:],
                                                                 progress)
            self.settings_worker.settings_saved.connect(self.on_save_settings_finish)
            self.settings_worker.start()
        except Exception as e:
            logging.error(e)
            progress.close()
            self.ui.show_alert(
                "Произошла ошибка при сохранении настроек. Проверьте, что все указанные значения действительны.")

    def on_save_settings_finish(self, data):
        """
        Метод, вызываемый по завершении операции обучения моделей. Метод информирует
        пользователя об успешном сохранении настроек, если сохранение настроек прошло
        успешно. В противном случае, метод информирует об ошибке обучения HMM-GMM модели,
        если в процессе ее обучения алгоритм Баума-Велша не сошелся.

        Аргументы:
            data: Кортеж, состоящий из двух элементов. Первый элемент кортежа есть True, если
                алгоритм Баума-Велша сошелся и False в противном случае. Второй элемент кортежа
                есть объект окна ожидания, которое появляется при сохранении настроек модели из
                вкладки "Настройки".
        """
        is_success, progress = data
        if not is_success:
            progress.close()
            self.ui.show_alert(
                "Произошла ошибка при обучении модели. Причины: либо указаны недействительные пути до набора данных для обучения HMM-GMM модели, либо указан ARPA-файл для декодера, содержащий информацию не о биграммной языковой модели.")
            return
        self.fill_hmm_gmm_tab()
        self.ui.is_ready_to_recognize = True
        progress.close()
        self.ui.show_alert(
            "Настройки были сохранены. Акустическая и языковая модели успешно натренированы. Это окно можно закрывать и приступать к распознованию речи.")

    def fill_hmm_gmm_tab(self):
        """
        Метод заполняет список 'QListWidget' созданными HMM-GMM моделями, а также список
        'hmm_gmm_information_list', каждый элемент которого содержит детальную информацию о
        соответствующей HMM-GMM модели.
        """
        self.ui.hmm_gmm_list_widget.clear()
        self.ui.hmm_gmm_information_list.clear()
        self.ui.hmm_gmm_text_area.clear()
        for hmm in self.logic.hmms:
            self.ui.hmm_gmm_list_widget.addItem(f"HMM-{hmm.association}")
            self.ui.hmm_gmm_information_list.append(
                HMMManager.get_hmm_gmm_data(hmm, is_converged=True, print_gmm_info=True, are_likelihoods_log=False))

    def hmm_gmm_list_widget_click(self, index):
        """
        Метод, выводящий в текстовое поле информацию о выбранной пользователем
        из списка 'QListWidget' HMM-GMM модели. Информация о модели берется из
        массива 'hmm_gmm_information_list', заполняемого в методе
        'fill_hmm_gmm_tab'.

        Аргументы:
            index: Индекс выбранной пользователем из списка 'QListWidget' HMM-GMM
                модели. Элементы списка 'QListWIdget' и 'hmm_gmm_information_list'
                связаны по индексу. Следовательно, по 'index' можно извлечь
                информацию о выбранной пользователем HMM-GMM модели.
        """
        self.ui.hmm_gmm_text_area.setText(self.ui.hmm_gmm_information_list[index])

    def extract_language_model_init_data(self, decoder_language_model_type, lm_widgets, progress):
        """
        Метод, извлекающий введеные пользователем на вкладке "Настройки" параметры языковой модели.
        Параментр 'training_corpus' или 'arpa_file_path' валидируется в зависимости от того, какой
        тип языковой модели выбрал пользователь. Если валидация не пройдена, то метод возвращает
        False. Иначе, метод возвращает кортеж извлеченных параметров. Метод может генерировать
        исключение, если параметры 'n', 'k', 'reserved_probability', 'addition' не соответсвуют
        типам 'int', 'int', 'float', 'int' соответственно. Это исключение обрабатывается в
        методе 'on_save_settings_button_clicked'.

        Аргументы:
            decoder_language_model_type: Тип языковой модели.
            lm_widgets: Кортеж, состоящий из виджетов, соответствующих настройкам языковой модели.
            progress: Объект окна ожидания, которое появляется при сохранении настроек модели из
                вкладки "Настройки".
        """
        n = int(self.ui.get_text(lm_widgets[1]))
        k = int(self.ui.get_text(lm_widgets[4]))
        reserved_probability = float(self.ui.get_text(lm_widgets[5]))
        addition = int(self.ui.get_text(lm_widgets[3]))
        training_corpus = self.ui.get_text(lm_widgets[6])
        arpa_file_path = self.ui.get_text(lm_widgets[0])

        if decoder_language_model_type == LanguageModel.KATZ_SMOOTHING_FROM_ARPA_FILE and not arpa_file_path and not Path(
                arpa_file_path).exists():
            progress.close()
            self.ui.show_alert("Путь до ARPA-файла не указан или некорректен.")
            return False

        if decoder_language_model_type != LanguageModel.KATZ_SMOOTHING_FROM_ARPA_FILE and not training_corpus and not Path(
                training_corpus).exists():
            progress.close()
            self.ui.show_alert("Путь до набора данных для обучения языковой модели не указан или некорректен.")
            return False

        return decoder_language_model_type, n, k, reserved_probability, addition, training_corpus, arpa_file_path

    def fill_decoding_tab(self, decoding_response):
        """
        Метод заполняет список 'QListWidget' декодированными результатами, а также список
        'decoding_information_list', каждый элемент которого содержит детальную информацию о
        соответствующем декодированном результате.

        Аргументы:
            decoding_response: Результат работы декодера. Содержит последовательности декодированных звуков, слов,
                вероятности этих последовательностей и, если используется подход multipass decoding, вероятности,
                переоцененные языковой моделью более высокого порядка.
        """
        self.ui.decoding_list_widget.clear()
        self.ui.decoding_information_list.clear()
        self.ui.decoding_text_area.clear()
        if self.logic.multipass_lm is None:
            self.ui.decoding_list_widget.addItem("Вариант №1")
            self.ui.decoding_information_list.append(
                f"Вариант №1\n\nПоследовательность слов:\n{" ".join(decoding_response[1])}\n\nПоследовательность звуков:\n{", ".join(decoding_response[2])}\n\nИсходная логарифмическая вероятность декодированной последовательности: {decoding_response[0]}")
        else:
            best_index, decodings, p_reestemated = decoding_response
            for i, decoding in enumerate(decodings, start=1):
                self.ui.decoding_list_widget.addItem(f"Вариант №{i}")
                self.ui.decoding_information_list.append(
                    f"Вариант №{i}\n\nПоследовательность слов:\n{" ".join(decoding[1])}\n\nПоследовательность звуков:\n{", ".join(decoding[2])}\n\nИсходная логарифмическая вероятность декодированной последовательности: {decoding[0]}\n\nПереоцененная логарифмическая вероятность: {p_reestemated[i - 1]}{"\n\nПросматриваемый вариант выбран среди всех как наиболее вероятный по результатам переоценки вероятностей." if best_index == i - 1 else ""}")

    def decoding_list_widget_click(self, index):
        """
        Метод, выводящий в текстовое поле информацию о выбранном пользователем
        из списка 'QListWidget' декодированном результате. Информация о
        декодированном результате берется из массива 'decoding_information_list',
        заполняемого в методе 'fill_decoding_tab'.

        Аргументы:
            index: Индекс выбранного пользователем из списка 'QListWidget' декодированного
                результата. Элементы списка 'QListWIdget' и 'decoding_information_list'
                связаны по индексу. Следовательно, по 'index' можно извлечь информацию
                о выбранном пользователем декодированном результате.
        """
        self.ui.decoding_text_area.setText(self.ui.decoding_information_list[index])

    class PerplexityComputationWorker(QThread):
        """
        Фоновый поток для выполнения операции вычисления метрики Perplexity.

        Сигналы:
            perplexity_computed: Сигнал, испускаемый при завершении работы потока.
        """
        perplexity_computed = pyqtSignal(tuple)

        def __init__(self, logic, corpus_path):
            """
            Инициализация потока.

            Аргументы:
                logic: Объект, реализующий бизнес-логику приложения.
                corpus_path: Путь до корпуса текста, по которому следует
                    вычислить метрику Perplexity.
            """
            super().__init__()
            self.logic = logic
            self.corpus_path = corpus_path

        def run(self):
            """
            Метод, выполняющий задачу вычисления метрики Perplexity в отдельном
            потоке. Он автоматически вызывается при запуске потока методом 'start()'.
            """
            perplexity1, perplexity2 = self.logic.get_perplexity(self.corpus_path)
            self.perplexity_computed.emit((perplexity1, perplexity2))

    def perplexity_validation(self):
        """
        Метод, выполняющий предварительную валидацию перед запуском процедуры
        вычисления метрики Perplexity.
        """
        if self.perplexity_computation_in_progress:
            self.ui.show_alert(
                "В настоящий момент метрика Perplexity уже вычисляется. Дождитесь окончания вычислений. Затем повторите попытку.")
            return False
        if not self.ui.is_ready_to_recognize:
            self.ui.show_alert(
                "Модель не настроена. Перейдите во вкладку \"Настройки\", чтобы сделать это. Затем повторите попытку.")
            return False
        corpus_path = self.ui.get_text(self.ui.perplexity_test_corpus_path)
        if not corpus_path:
            self.ui.show_alert(
                "Путь до корпуса текста, по которому следует посчитать метрику Perplexity не указан. Укажите его и повторите попытку.")
            return False
        return True, corpus_path

    def on_perplexity_button_clicked(self):
        """
        Метод вызывется при нажатии кнопки "Посчитать Perplexity" из вкладки "Метрика Perplexity"
        приложения. В методе запускается операция вычисления метрики Perplexity в отдельном
        потоке.
        """
        value = self.perplexity_validation()
        if value is False:
            return
        self.ui.set_text(self.ui.perplexity_decoder, "Вычисляется...")
        self.ui.set_text(self.ui.perplexity_multipass, "Вычисляется...")
        self.perplexity_worker = Controller.PerplexityComputationWorker(self.logic, value[1])
        self.perplexity_worker.perplexity_computed.connect(self.on_perplexity_computed)
        self.perplexity_computation_in_progress = True
        self.perplexity_worker.start()

    def on_perplexity_computed(self, data):
        """
        Метод, вызываемый по завершении вычисления метрики Perplexity.
        В методе происходит обновление пользовательского интерфейса с
        выводом в соответствующие текстовые поля посчитанные значения
        метрики Perplexity.

        Аргументы:
            data: Кортеж, первый элемент которого есть посчитанная
                Perplexity для языковой модели, используемой в декодере,
                а второй элемент есть посчитанная Perplexity для языковой
                модели, используемой при переоценке вероятностей декодированных
                результатов в подходе multipass decoding. Если подход multipass
                decoding не используется, в соответствующем поле будет выведена
                информация "Модель не используется".
        """
        perplexity1, perplexity2 = data
        self.ui.set_text(self.ui.perplexity_decoder, str(perplexity1))
        self.ui.set_text(self.ui.perplexity_multipass,
                         str(perplexity2) if perplexity2 is not None else "Модель не используется")
        self.perplexity_computation_in_progress = False

    class WerComputationWorker(QThread):
        """
        Фоновый поток для выполнения операции вычисления метрики Word Error Rate.

        Сигналы:
            wer_computed: Сигнал, испускаемый при завершении работы потока.
        """
        wer_computed = pyqtSignal(float)

        def __init__(self, logic, audio_file_path, transcription_path):
            """
            Инициализация потока.

            Аргументы:
                logic: Объект, реализующий бизнес-логику приложения.
                audio_file_path: Путь до аудио-файла, содержимое которого бедет
                    распознано, чтобы потом сравнить его с содержимым транскрипции.
                transcription_path: Путь до транскрипции, содержащей расшифровку
                    аудио-файла, расположенного по пути audio_file_path.
            """
            super().__init__()
            self.logic = logic
            self.audio_file_path = audio_file_path
            with open(transcription_path, "r", encoding="utf-8") as transcription_file:
                self.transcription_text = transcription_file.read().strip()

        def run(self):
            """
            Метод, выполняющий задачу вычисления метрики Word Error Rate в отдельном
            потоке. Он автоматически вызывается при запуске потока методом 'start()'.
            """
            result = self.logic.decode(self.audio_file_path)
            if self.logic.multipass_lm is None:
                decoded_words = " ".join(result[1])
            else:
                best_index, decodings, _ = result
                decoded_words = " ".join(decodings[best_index][1])
            wer_value = wer.wer(decoded_words, self.transcription_text)
            self.wer_computed.emit(wer_value)

    def wer_validation(self):
        """
        Метод, выполняющий предварительную валидацию перед запуском процедуры
        вычисления метрики Word Error Rate.
        """
        if self.wer_computation_in_progress:
            self.ui.show_alert(
                "В настоящий момент метрика Word Error Rate уже вычисляется. Дождитесь окончания вычислений. Затем повторите попытку.")
            return False
        if not self.ui.is_ready_to_recognize:
            self.ui.show_alert(
                "Модель не настроена. Перейдите во вкладку \"Настройки\", чтобы сделать это. Затем повторите попытку.")
            return False
        audio_file_path = self.ui.get_text(self.ui.wer_audio_path)
        transcription_path = self.ui.get_text(self.ui.wer_transcription_path)
        if not audio_file_path or not transcription_path:
            self.ui.show_alert(
                "Требуется указать пути до аудио-файла и его транскрипции. Пожалуйста, проверьте, что все пути указаны. Затем повторите попытку.")
            return False
        return audio_file_path, transcription_path

    def on_wer_button_clicked(self):
        """
        Метод вызывется при нажатии кнопки "Посчитать Word Error Rate" из вкладки
        "Метрика Word Error Rate" приложения. В методе запускается операция вычисления
        метрики Word Error Rate в отдельном потоке.
        """
        value = self.wer_validation()
        if value is False:
            return
        self.ui.set_text(self.ui.wer_value, "Вычисляется...")
        self.wer_worker = Controller.WerComputationWorker(self.logic, *value)
        self.wer_worker.wer_computed.connect(self.on_wer_computation_finish)
        self.wer_computation_in_progress = True
        self.wer_worker.start()

    def on_wer_computation_finish(self, wer_value):
        """
        Метод, вызываемый по завершении вычисления метрики Word Error Rate.
        В методе происходит обновление пользовательского интерфейса с
        выводом в соответствующeе текстовое поле посчитанного значения
        метрики Word Error Rate.

        Аргументы:
            wer_value: Вычисленное значение метрики Word Error Rate, которое
                выводится в соответствующем текстовом поле.
        """
        self.ui.set_text(self.ui.wer_value, str(wer_value))
        self.wer_computation_in_progress = False

    def on_decoding_option_change(self, index):
        """
        Метод, вызываемый при переключении пользователем алгоритма декодирования
        из выпадающего списка во вкладке "Настройки". Если выбран алгоритм
        "Витерби", то количество декодированных последовательностей есть 1,
        подход multipass decoding не используется и нужно скрыть соответствующие
        элементы интерфейса. Если выбран алгоритм "Витерби с поиском N лучших
        последовательностей", то такие элементы следует отобразить.

        Аргументы:
            index: Индекс выбранного элемента из выпадающего списка.
        """
        self.decoder = Decoder(index)
        if self.decoder == Decoder.VITERBI:
            self.ui.n_best.hide()
            self.ui.multipass_language_model_label.hide()
            self.ui.multipass_language_model_box.hide()
            self.ui.multipass_lm_holder.hide()
        else:
            self.ui.n_best.show()
            self.ui.multipass_language_model_label.show()
            self.ui.multipass_language_model_box.show()
            self.ui.multipass_lm_holder.show()

    class OOVComputationWorker(QThread):
        """
        Фоновый поток для выполнения операции вычисления метрики Out-of-Vocabulary.

        Сигналы:
            oov_computed: Сигнал, испускаемый при завершении работы потока.
        """
        oov_computed = pyqtSignal(tuple)

        def __init__(self, logic, corpus_path):
            """
            Инициализация потока.

            Аргуметнты:
                logic: Объект, реализующий бизнес-логику приложения.
                corpus_path: Путь до корпуса текста, по которому следует вычислить
                    значение метрики Out-of-Vocabulary.
            """
            super().__init__()
            self.logic = logic
            self.corpus_path = corpus_path

        def run(self):
            """
            Метод, выполняющий задачу вычисления метрики Out-of-Vocabulary в отдельном
            потоке. Он автоматически вызывается при запуске потока методом 'start()'.
            """
            oov1, oov2 = self.logic.get_oov(self.corpus_path)
            self.oov_computed.emit((oov1, oov2))

    def oov_validation(self):
        """
        Метод, выполняющий предварительную валидацию перед запуском процедуры
        вычисления метрики Out-of-Vocabulary.
        """
        if self.oov_computation_in_progress:
            self.ui.show_alert(
                "В настоящий момент метрика Out-Of-Vocabulary уже вычисляется. Дождитесь окончания вычислений. Затем повторите попытку.")
            return False
        if not self.ui.is_ready_to_recognize:
            self.ui.show_alert(
                "Модель не настроена. Перейдите во вкладку \"Настройки\", чтобы сделать это. Затем повторите попытку.")
            return False
        corpus_path = self.ui.get_text(self.ui.oov_test_corpus_path)
        if not corpus_path:
            self.ui.show_alert(
                "Путь до корпуса текста, по которому следует посчитать метрику Out-Of-Vocabulary не указан. Укажите его и повторите попытку.")
            return False
        return corpus_path

    def on_oov_button_clicked(self):
        """
        Метод вызывется при нажатии кнопки "Посчитать Out-Of-Vocabulary"
        из вкладки "Метрика Out-of-Vocabulary" приложения. В методе запускается
        операция вычисления метрики Out-of-Vocabulary в отдельном потоке.
        """
        value = self.oov_validation()
        if value is False:
            return
        self.ui.set_text(self.ui.oov_decoder, "Вычисляется...")
        self.ui.set_text(self.ui.oov_multipass, "Вычисляется...")
        self.oov_worker = Controller.OOVComputationWorker(self.logic, value)
        self.oov_worker.oov_computed.connect(self.on_oov_computation_finish)
        self.oov_computation_in_progress = True
        self.oov_worker.start()

    def on_oov_computation_finish(self, data):
        """
        Метод, вызываемый по завершении вычисления метрики Out-of-Vocabulary.
        В методе происходит обновление пользовательского интерфейса с
        выводом в соответствующие текстовые поля посчитанные значения
        метрики Out-of-Vocabulary.

        Аргументы:
            data: Кортеж, первый элемент которого есть посчитанная Out-of-Vocabulary для языковой
                модели, используемой в декодере, а второй элемент которого есть посчитанная
                Out-of-Vocabulary для языковой модели, используемой при переоценке вероятностей
                декодированных результатов в подходе multipass decoding. Если подход
                multipass decoding не используется, в соответствующем поле будет выведена
                информация "Модель не используется".
        """
        oov1, oov2 = data
        self.ui.set_text(self.ui.oov_decoder, str(oov1))
        self.ui.set_text(self.ui.oov_multipass, str(oov2) if oov2 is not None else "Модель не используется")
        self.oov_computation_in_progress = False

    def on_decoder_lm_option_change(self, index):
        """
        Метод, вызываемый при переключении пользователем языковой модели, используемой в декодере,
        из выпадающего списка во вкладке "Настройки".

        Аргументы:
            index: Индекс выбранного элемента из выпадающего списка.
        """
        self.ui.switch_language_model_settings(self.ui.decoder_lm_widgets, LanguageModel(index))

    def on_multipass_lm_option_change(self, index):
        """
        Метод, вызываемый при переключении пользователем языковой модели, используемой при переоценке
        вероятностей в подходе multipass decoding, из выпадающего списка во вкладке "Настройки".

        Аргументы:
            index: Индекс выбранного элемента из выпадающего списка.
        """
        self.ui.switch_language_model_settings(self.ui.multipass_lm_widgets, LanguageModel(index))

    def on_load_acoustic_model_option_change(self, index):
        """
        Метод, вызываемый при переключении пользователем способа инициализации акустической модели
        из выпадающего списка во вкладке "Настройки".

        Аргументы:
            index: Индекс выбранного элемента из выпадающего списка.
        """
        self.ui.switch_acoustic_model_initialization(AcousticModelInitialization(index))
