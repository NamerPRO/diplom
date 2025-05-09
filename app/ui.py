import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox, \
    QFileDialog, QComboBox, QLineEdit, QTabWidget, QMainWindow, QScrollArea, QListWidget, QTextEdit

from app.logic import LanguageModel, AcousticModelInitialization


class WaitWindow(QWidget):
    def __init__(self, title="", text="", parent=None):
        """
        Инициализация окна ожидания, отображаемого при сохранении
        настроек модели из вкладки "Настройки".

        Аргументы:
            title: Заголовок окна. По-умолчанию: "".
            text: Текст, отображаемый в окне. По-умолчанию: "", что
                означает отсутствие текста.
            parent: Родительское окно. По-умолчанию: None, что
                означает отсутствие ссылки.
        """
        super().__init__()
        np.set_printoptions(linewidth=int(1e9))
        
        self.setWindowTitle(title)
        self.setFixedSize(200, 60)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.CustomizeWindowHint)
        self.center_on_parent(parent)

        vbox_layout = QVBoxLayout()

        font = QFont()
        font.setPointSize(10)

        label = QLabel(text)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox_layout.addWidget(label)

        self.setLayout(vbox_layout)

    def center_on_parent(self, parent):
        """
        Метод, располагающий окно в середине родительского окна.

        Аргументы:
            parent: Родительское окно.
        """
        if parent:
            parent_rect = parent.geometry()
            self_rect = self.frameGeometry()
            x = parent_rect.x() + (parent_rect.width() - self_rect.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self_rect.height()) // 2
            self.move(x, y)


class UILayer(QMainWindow):
    """
    Класс UILayer представляет собой главное окно приложения.
    Это окно используется для обеспечения взаимодействия пользователя
    с разработанной системой. Оно определяет интерфейс, через который
    возможно следующее:
    1. Инициализация задачи поиска последовательности данных в
        аудиопотоке.
    2. Просмотр декодированных результатов: последовательности декодированных,
        слов и звуков, - а также вероятностей декодированных результатов.
    3. Настройка системы, включающая тренировку акустической и языковой моделей.
    4. Просмотр параметров натренированных моделей.
    5. Вычисление метрик, таких как Perplexity, Out-Of-Vocabulary, Word Error Rate.
    """

    def __init__(self):
        """
        Конструктор класса UILayer.
        """
        super().__init__()
        self.is_ready_to_recognize = False
        self.initUI()

    def initUI(self):
        """
        Инициализация пользовательского интерфейса, который в самом общем смысле
        состоит из семи вкладок:
        1. Главная - вкладка, через которую инициализируется процесс поиска заданной
            последовательности данных в аудиопотоке.
        2. Декодирование - вкладка, на которой можно посмотреть результат работы декодера:
            последовательности декодированных слов и звуков, - а также вероятности
            декодированных результатов.
        3. HMM-GMM - вкладка, на которой можно посмотреть параметры натренированной
            HMM-GMM модели.
        4. Метрика Perplexity - вкладка, которая содержит функционал вычисления
            метрики Perplexity.
        5. Метрика Word Error Rate - вкладка, которая содержит функционал вычисления
            метрики Word Error Rate.
        6. Метрика Out-of-Vocabulary - вкладка, которая содержит функионал вычисления
            метрики Out-Of-Vocabulary.
        7. Настройки - вкладка, на которой можно настроить систему распознования речи
            и запустить процесс тренировки модели.
        """
        self.setWindowTitle("Поиск заданной последовательности данных в аудиопотоке")
        self.setFixedSize(900, 500)
        self.setStyleSheet("background-color: #FFFFFF; color: #121212;")
        self.setWindowFlags(Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowCloseButtonHint)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.tabs = QTabWidget()

        self.tabs.addTab(self.create_main_tab(), "Главная")
        self.tabs.addTab(self.create_decoding_result_tab(), "Декодирование")
        self.tabs.addTab(self.create_hmm_gmm_info_tab(), "HMM-GMM")
        self.tabs.addTab(self.create_perplexity_tab(), "Метрика Perplexity")
        self.tabs.addTab(self.create_wer_tab(), "Метрика Word Error Rate")
        self.tabs.addTab(self.create_oov_tab(), "Метрика Out-of-Vocabulary")
        self.tabs.addTab(self.create_settings_tab(), "Настройки")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.central_widget.setLayout(layout)

    def create_main_tab(self):
        """
        Инициализация интерфейса для вкладки "Главное".
        """
        tab = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.init_title("Главная", self.main_layout)

        label_font = QFont()
        label_font.setPointSize(12)
        main_select_decode_file_label = QLabel("Выберите файл для распознования")
        main_select_decode_file_label.setFont(label_font)
        self.main_layout.addWidget(main_select_decode_file_label)
        self.main_selected_decode_file = self.select_file_layout("", self.main_layout, is_folder=False, file_type="wav")

        self.main_search_substring = QLabel("Искомая последовательность данных")
        self.main_search_substring.setFont(label_font)
        self.main_layout.addWidget(self.main_search_substring)
        self.main_search_substring_input = self.set_input_layout(None,
                                                                 "Введите искомую последовательность через пробел без знаков препинания...",
                                                                 self.main_layout)

        main_decoding_result = QLabel("Результат распознования")
        main_decoding_result.setFont(label_font)
        self.main_layout.addWidget(main_decoding_result)

        self.main_decoding_result_area = QTextEdit()
        self.main_decoding_result_area.setPlaceholderText("В этом поле отобразится результат распознования.")
        self.main_decoding_result_area.setReadOnly(True)
        self.main_layout.addWidget(self.main_decoding_result_area)

        self.recognize_audio_button = self.set_button("Проверить наличие фразы", self.main_layout)

        tab.setLayout(self.main_layout)
        return tab

    def create_decoding_result_tab(self):
        """
        Инициализация интерфейса для вкладки "Декодирование".
        """
        tab = QWidget()
        self.decoding_tab_layout = QVBoxLayout()

        self.init_title("Декодирование", self.decoding_tab_layout)

        font = QFont()
        font.setPointSize(10)

        self.decoding_list_widget = QListWidget()
        self.decoding_list_widget.setFixedHeight(150)
        self.decoding_list_widget.setFont(font)
        self.decoding_list_widget.addItems(["Результаты последнего успешного декодирования будут отображены здесь."])
        self.decoding_tab_layout.addWidget(self.decoding_list_widget)

        self.decoding_information_list = ["Выберите элемент из списка выше и информация отобразится в этом поле."]

        self.decoding_text_area = QTextEdit()
        self.decoding_text_area.setPlaceholderText(
            "Выберите элемент из списка выше и информация отобразится в этом поле.")
        self.decoding_text_area.setMinimumHeight(300)
        self.decoding_text_area.setReadOnly(True)
        self.decoding_tab_layout.addWidget(self.decoding_text_area)

        tab.setLayout(self.decoding_tab_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(tab)
        return scroll_area

    def create_hmm_gmm_info_tab(self):
        """
        Инициализация интерфейса для вкладки "HMM-GMM".
        """
        tab = QWidget()
        self.hmm_gmm_layout = QVBoxLayout()

        self.init_title("HMM-GMM модель", self.hmm_gmm_layout)

        font = QFont()
        font.setPointSize(10)

        self.hmm_gmm_list_widget = QListWidget()
        self.hmm_gmm_list_widget.setFixedHeight(150)
        self.hmm_gmm_list_widget.setFont(font)
        self.hmm_gmm_list_widget.addItems(["Модель HMM-GMM не натренирована! Посетите вкладку настройки."])
        self.hmm_gmm_layout.addWidget(self.hmm_gmm_list_widget)

        self.hmm_gmm_information_list = ["Выберите элемент из списка выше и информация отобразится в этом поле."]

        self.hmm_gmm_text_area = QTextEdit()
        self.hmm_gmm_text_area.setPlaceholderText(
            "Выберите элемент из списка выше и информация отобразится в этом поле.")
        self.hmm_gmm_text_area.setMinimumHeight(500)
        self.hmm_gmm_text_area.setReadOnly(True)
        self.hmm_gmm_layout.addWidget(self.hmm_gmm_text_area)

        tab.setLayout(self.hmm_gmm_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(tab)
        return scroll_area

    def create_perplexity_tab(self):
        """
        Инициализация интерфейса для вкладки "Метрика Perplexity".
        """
        tab = QWidget()
        self.perplexity_layout = QVBoxLayout()
        self.perplexity_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.init_title("Perplexity", self.perplexity_layout)

        font = QFont()
        font.setPointSize(12)

        description = QLabel()
        description.setFont(font)
        description.setText(
            "Perplexity (PP, перплексия) — это метрика, часто используемая для оценки качества языковых моделей.\nОна измеряет, насколько хорошо модель предсказывает тестовые данные.\nВ контексте языковых моделей, чем ниже перплексия,\nтем лучше модель предсказывает последовательности слов.\n")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.perplexity_layout.addWidget(description)

        self.perplexity_test_corpus_path = self.select_file_layout("Путь до корпуса текста", self.perplexity_layout,
                                                                   is_folder=False, file_type="txt")

        self.perplexity_decoder = self.set_input_layout("Perplexity (для модели в декодере)",
                                                        "В поле отобразится вычисленное значение метрики.",
                                                        self.perplexity_layout, is_editable=False)
        self.perplexity_multipass = self.set_input_layout("Perplexity (для модели на втором этапе декодирования)",
                                                          "В поле отобразится вычисленное значение метрики.",
                                                          self.perplexity_layout, is_editable=False)

        self.perplexity_layout.addStretch(1)
        self.compute_perplexity_button = self.set_button("Посчитать Perplexity", self.perplexity_layout)

        tab.setLayout(self.perplexity_layout)
        return tab

    def create_wer_tab(self):
        """
        Инициализация интерфейса для вкладки "Метрика Word Error Rate".
        """
        tab = QWidget()
        self.wer_layout = QVBoxLayout()
        self.wer_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.init_title("Word Error Rate", self.wer_layout)

        font = QFont()
        font.setPointSize(12)

        description = QLabel()
        description.setFont(font)
        description.setText(
            "Word Error Rate (WER) — это метрика, используемая для оценки точности\nсистем автоматического распознавания речи. Она показывает,\nнасколько сильно расшифрованный текст отличается от\nэталонного (правильного) текста.\n")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wer_layout.addWidget(description)

        self.wer_audio_path = self.select_file_layout("Путь до аудио-файла", self.wer_layout,
                                                      is_folder=False, file_type="wav")
        self.wer_transcription_path = self.select_file_layout("Путь до транскрипции", self.wer_layout,
                                                              is_folder=False, file_type="txt")
        self.wer_value = self.set_input_layout("Word Error Rate", "В поле отобразится вычисленное значение метрики.",
                                               self.wer_layout, is_editable=False)

        self.wer_layout.addStretch(1)
        self.compute_wer_button = self.set_button("Посчитать Word Error Rate", self.wer_layout)

        tab.setLayout(self.wer_layout)
        return tab

    def create_oov_tab(self):
        """
        Инициализация интерфейса для вкладки "Метрика Out-of-Vocabulary".
        """
        tab = QWidget()
        self.oov_layout = QVBoxLayout()
        self.oov_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.init_title("Out-Of-Vocabulary", self.oov_layout)

        font = QFont()
        font.setPointSize(12)

        description = QLabel()
        description.setFont(font)
        description.setText(
            "Out-Of-Vocabulary (OOV) - это метрика, которая используется для оценки доли слов,\nотсутствующих в словаре системы. Чем выше этот показатель,\nтем сложнее языковой модели правильно оценивать поданные ей данные,\nтак как она не обучена на неизвестных данных.\n")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.oov_layout.addWidget(description)

        self.oov_test_corpus_path = self.select_file_layout("Путь до корпуса текста", self.oov_layout,
                                                            is_folder=False, file_type="txt")

        self.oov_decoder = self.set_input_layout("Out-Of-Vocabulary (для модели в декодере)",
                                                 "В поле отобразится вычисленное значение метрики.",
                                                 self.oov_layout, is_editable=False)
        self.oov_multipass = self.set_input_layout("Out-Of-Vocabulary (для модели на втором этапе декодирования)",
                                                   "В поле отобразится вычисленное значение метрики.",
                                                   self.oov_layout, is_editable=False)

        self.oov_layout.addStretch(1)
        self.compute_oov_button = self.set_button("Посчитать Out-Of-Vocabulary", self.oov_layout)

        tab.setLayout(self.oov_layout)
        return tab

    def create_settings_tab(self):
        """
        Инициализация интерфейса для вкладки "Настройки".
        """
        tab = QWidget()
        self.settings_tab_layout = QVBoxLayout()

        self.init_title("Настройки", self.settings_tab_layout)

        self.set_input_layout("Акустическая модель", "", self.settings_tab_layout, False, "HMM-GMM модель")

        _, self.acoustic_model_box = self.init_acoustic_model_initialization_combobox(self.settings_tab_layout)

        self.acoustic_model_widget = self.get_acoustic_model_widget(self.settings_tab_layout)
        self.switch_acoustic_model_initialization(AcousticModelInitialization.FROM_TRAINING_DATA)

        _, self.decoding_box = self.init_decoding_algorithm_combobox(self.settings_tab_layout)

        self.n_best = self.set_input_layout("Количество лучших последовательностей", "Введите значение параметра...",
                                            self.settings_tab_layout, value="10")

        self.lmsf = self.set_input_layout("Параметр LMSF", "Введите значение параметра...", self.settings_tab_layout,
                                          value="15")
        self.wip = self.set_input_layout("Параметр WIP", "Введите значение параметра...", self.settings_tab_layout,
                                         value="0.5")

        _, self.decoder_language_model_box = self.init_language_model_combobox("Языковая модель в декодере",
                                                                               self.settings_tab_layout)

        _, *self.decoder_lm_widgets = self.get_language_model_settings(self.settings_tab_layout, False)
        self.switch_language_model_settings(self.decoder_lm_widgets, LanguageModel.KATZ_SMOOTHING_FROM_TRAINING_CORPUS)

        self.multipass_language_model_label, self.multipass_language_model_box = self.init_language_model_combobox(
            "Языковая модель на втором этапе декодирования", self.settings_tab_layout)

        self.multipass_lm_holder, *self.multipass_lm_widgets = self.get_language_model_settings(
            self.settings_tab_layout, True)
        self.switch_language_model_settings(self.multipass_lm_widgets, LanguageModel.KATZ_SMOOTHING_FROM_TRAINING_CORPUS)

        self.save_settings_button = self.set_button("Сохранить настройки", self.settings_tab_layout)

        tab.setLayout(self.settings_tab_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(tab)
        return scroll_area

    def init_title(self, text, layout):
        """
        Метод отрисовывающий текст черного цвета шрифта Verdana размером 25pt,
        использующийся в качестве заголовка на каждой из вкладок. Сверху и снизу
        текста заданы отступы размером 30px каждый.

        Аргументы:
            text: Текст, содержащийся в заголовке.
            layout: Макет, в который добавляется описанный выше заголовок.
        """
        self.settings_title = QLabel(text)
        self.settings_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.settings_title.setFont(QFont("Verdana", 25, QFont.Weight.Bold))
        self.settings_title.setStyleSheet("padding-top: 30px; padding-bottom: 30px;")
        layout.addWidget(self.settings_title)

    def init_language_model_combobox(self, text, layout):
        """
        Определяет выпадающий список с заданным текстом 'text',
        расположенным слева от него. В выпадающем списке предлагается
        выбрать тип языковой модели.

        Аргументы:
            text: Текст, расположенный слева от выпадающего списка.
            layout: Макет, в который добавляется выпадающий список с
                текстом, идущим слева от него.

        Возвращаемое значение:
            Виджет текста, расположенный слева от выпадающего списка, и
            выпадающий список, рядом с которым расположен виджет текста.
        """
        return self.init_combobox(
            text=text,
            options=[
                "Модель без сглаживания",
                "Модель со сглаживанием по Лапласу",
                "Модель с аддитивным сглаживанием",
                "Модель со сглаживанием по Гуду-Тьюрингу",
                "Модель со сглаживанием по Кацу (по тренировочному корпусу)",
                "Модель со сглаживанием по Кацу (из ARPA файла)"
            ],
            default_option_index=4,
            layout=layout
        )

    def init_acoustic_model_initialization_combobox(self, layout):
        """
        Определяет выпадающий список с заданным текстом 'text',
        расположенным слева от него. В выпадающем списке предлагается
        выбрать, обучить HMM-GMM модель на тренировочных данных или
        получить из сохраненной копии.

        Аргументы:
            layout: Макет, в который добавляется выпадающий список с
                текстом, идущим слева от него.

        Возвращаемое значение:
            Виджет текста, расположенный слева от выпадающего списка, и
            выпадающий список, рядом с которым расположен виджет текста.
        """
        return self.init_combobox(
            text="Инициализация акустической модели",
            options=[
                "Обучить на тренировочных данных",
                "Получить из сохраненной копии"
            ],
            default_option_index=0,
            layout=layout
        )

    def get_acoustic_model_widget(self, layout):
        """
        Метод, возвращающий виджет, содержащий дочерние виджеты,
        образующие две категории:
        1. Интерфейс для выбора папок с данными для обучения
            HMM-GMM модели.
        2. Интерфейс для выбора файла (сохраненной копии),
            из которого будет получена модель.
        Логика из метода 'switch_acoustic_model_initialization'
        позволяет переключаться между пунктами 1 и 2, путем
        скрытия элементов из противоположного пункта, то есть
        выбирать способ инициализации акустической модели.

        Аргументы:
            layout: Макет, в который добавляется описанный
                выше виджет.

        Возвращаемое значение:
            Описанный выше виджет.
        """
        background_widget = QWidget()
        background_widget.setStyleSheet("background-color: #F0F0F0;")

        vbox = QVBoxLayout()

        self.hmm_dataset_path = self.select_file_layout("Путь к набору данных для обучения HMM", vbox, is_folder=True)
        self.gmm_dataset_path = self.select_file_layout("Путь к набору данных для обучения GMM", vbox, is_folder=True)
        self.saved_hmm_gmm_model = self.select_file_layout("Путь к сохраненной HMM-GMM модели", vbox, file_type="pkl")

        background_widget.setLayout(vbox)
        layout.addWidget(background_widget)
        return background_widget

    def switch_acoustic_model_initialization(self, initialization_type):
        """
        Метод позволяет переключаться между пунктами 1 и 2, описанными
        в описании метода 'get_acoustic_model_widget' путем скрытия элементов
        из противоположного пункта, то есть выбирать способ инициализации
        акустической модели.

        Аргументы:
            initialization_type: Тип инициализации HMM-GMM модели. Если указан
                AcousticModelInitialization.FromTrainingData, то отобразятся элементы
                из пункта 1 (и скроются из пункта 2). Иначе, наоборот.
        """
        self.hmm_dataset_path.hide()
        self.gmm_dataset_path.hide()
        self.saved_hmm_gmm_model.hide()

        if initialization_type == AcousticModelInitialization.FROM_TRAINING_DATA:
            self.hmm_dataset_path.show()
            self.gmm_dataset_path.show()
        else:
            self.saved_hmm_gmm_model.show()

    def init_decoding_algorithm_combobox(self, layout):
        """
        Определяет выпадающий список с заданным текстом,
        расположенным слева от него. В выпадающем списке предлагается
        выбрать варинат алгоритма декодирования.

        Аргументы:
            layout: Макет, в который добавляется описанный
                выше виджет.

        Возвращаемое значение:
            Виджеты текста, расположенного слева от выпадающего списка, и
            выпадающего списка, рядом с которым расположен виджет текста.
        """
        return self.init_combobox(
            text="Алгоритм декодирования",
            options=[
                "Витерби",
                "Витерби с поиском N лучших последовательностей"
            ],
            default_option_index=1,
            layout=layout
        )

    def get_text(self, widget):
        """
        Возвращает текст из найденного QLineEdit в виджете.
        Все подаваемые на вход методу виджеты должны иметь единственный
        QLineEdit по договоренности.

        Аргументы:
            widget: Виджет, в котором по договоренности содержится ровно
                один QLineEdit.

        Возвращаемое значение:
            Текст из найденного QLineEdit в виджете.
        """
        return widget.findChild(QLineEdit).text()

    def set_text(self, widget, text):
        """
        Устанавливает текст для найденного QLineEdit в виджете.
        Все подаваемые на вход методу виджеты должны иметь единственный
        QLineEdit по договоренности.

        Аргументы:
            widget: Виджет, в котором по договоренности содержится ровно
                один QLineEdit.
        """
        widget.findChild(QLineEdit).setText(text)

    def init_combobox(self, text, options, layout, default_option_index=0):
        """
        Метод, задающий интерфейс для выпадающего списка с текстом 'text',
        расположенным слева от него. Вариаты выпадающего списка перечислены
        в 'options'. Выбранный по-умолчанию вариант задается default_option_index
        и по-умолчанию есть 0 (самый первый).

        Аргументы:
            text: Текст, расположенный слева от выпадающего списка.
            options: Варианты в выпадающем списке.
            layout: Макет, в который добавляется описанный выше выпадающий
                список с текстом слева от него.
            default_option_index: Выбранный по-умолчанию вариант выпадающего
                списка. По-умолчанию: 0 (самый первый). Не модет быть больше,
                чем максимальный индекс массива 'options'.

        Возвращаемое значение:
            Виджеты текста, расположенного слева от выпадающего списка, и
            выпадающего списка, рядом с которым расположен виджет текста.
        """
        hbox = QHBoxLayout()
        label = QLabel(text)
        font = QFont()
        font.setPointSize(10)
        label.setFont(font)
        combo_box = QComboBox(self)
        combo_box.setFont(font)
        for option in options:
            combo_box.addItem(option)
        combo_box.setCurrentIndex(default_option_index)
        hbox.addWidget(label)
        hbox.addWidget(combo_box)
        layout.addLayout(hbox)
        return label, combo_box

    def get_language_model_settings(self, layout, allow_enter_n=True):
        """
        Метод, устаналивающий виджет, содержащий набор дочерних виджетов,
        комбинации которых можно использовать для инициализации языковой
        модели каждого типа:
        1. Модель без сглаживания.
        2. Модель со сглаживанием по Лапласу.
        3. Модель с аддитивным сглаживанием.
        4. Модель со сглаживанием по Гуду-Тьюрингу.
        5. Модель со сглаживанием по Кацу (обучение на тренировочном корпсе)
        6. Модель со сглаживанием по Кацу (восстановление из сохраненного ARPA-файла)
        Логика из метода 'switch_language_model_settings' позволяет переключаться между пунктами 1-6, путем
        скрытия элементов из всех пунктов, кроме выбранного, то есть выбрать таким образом способ инициализации
        языковой модели.

        Аргументы:
            layout: Макет, в который добавляется описанный
                выше виджет.
            allow_enter_n: Если True, то пользователю разрешается устанавливать размерность языковой модели.
                В противном случае, размерность языковой модели устанавливать запрещается.

        Возвращаемое значение:
            Кортеж, состоящий из описанного выше виджета и его составляющих (всех дочерних виджетов).
        """
        background_widget = QWidget()
        background_widget.setStyleSheet("background-color: #F0F0F0;")

        vbox = QVBoxLayout()

        lm_arpa_file_path = self.select_file_layout("Выберите ARPA-файл", vbox, file_type="arp")
        lm_n = self.set_input_layout("Параметр n", "Введите значение параметра...", vbox, allow_enter_n,
                                     "3" if allow_enter_n else "2")
        lm_laplace_alpha = self.set_input_layout("Параметр alpha", "Введите значение параметра...", vbox,
                                                 is_editable=False, value="1")
        lm_additive_alpha = self.set_input_layout("Параметр alpha", "Введите значение параметра...", vbox, value="10")
        lm_k = self.set_input_layout("Параметр k", "Введите значение параметра...", vbox, value="5")
        lm_reserved_probability = self.set_input_layout("Зарезервированная вероятность",
                                                        "Вероятность выделенная под неизвестные слова...", vbox,
                                                        value="0.01")
        lm_training_dataset_path = self.select_file_layout("Выберите файл - тренировочный корпус текста", vbox)

        background_widget.setLayout(vbox)
        layout.addWidget(background_widget)

        return (background_widget, lm_arpa_file_path, lm_n, lm_laplace_alpha, lm_additive_alpha, lm_k,
                lm_reserved_probability, lm_training_dataset_path)

    def switch_language_model_settings(self, language_model_widgets, type):
        """
        Метод позволяет переключаться между пунктами 1-6, описанными выше в методе
        'get_language_model_settings' путем скрытия элементов из всех пунктов, кроме
        выбранного, то есть выбрать таким образом способ инициализации языковой модели.

        Аргуметны:
            language_model_widgets: Кортеж, состоящий из дочерних виджетов, получаемых
                из метода 'get_language_model_settings'.
            type: Тип языковой модели. Определяет, какие из виджетов следует отрисовать, а
                какие скрыть.
        """
        for item in language_model_widgets:
            item.hide()

        if type == LanguageModel.KATZ_SMOOTHING_FROM_ARPA_FILE:
            language_model_widgets[0].show()
        else:
            language_model_widgets[1].show()

            if type == LanguageModel.LAPLACE_SMOOTHING:
                language_model_widgets[2].show()

            if type == LanguageModel.ADDITIVE_SMOOTHING:
                language_model_widgets[3].show()

            if type == LanguageModel.KATZ_SMOOTHING_FROM_TRAINING_CORPUS:
                language_model_widgets[4].show()
                language_model_widgets[5].show()

            language_model_widgets[6].show()

    def select_file_layout(self, text, layout, is_folder=False, file_type="txt"):
        """
        Метод отрисовывает интерфейс выбора файла. Интерфейс состоит из
        текста, определяемого параметром 'text', текстового поля, в котором
        отображается путь выбранного файла/папки, и, идущей следом, кнопки,
        нажатие на которую открывает системное окно выбора файла/папки.

        Аргументы:
            text: Текст, отображаемый слева от текстового поля, в котором
                отображается путь выбранного файла.
            layout: Макет, в который будет добавлен виджет, содержащий
                описанные выше элементы интерфейса.
            is_folder: Если True, то нажатие на кнопку откроет системное
                окно выбора папки. Иначе нажатие на кнопку откроет системное
                окно выбора файла.
            file_type: Имеет эффект только есть is_folder=False. Определяет,
                файлы какого типа следует предлагать выбирать пользователю в
                системном окне выбора файла. Возможные значения: txt, pkl, wav.

        Возвращаемое значение:
            Виджет, содержащий описанные выше элементы интерфейса.
        """
        background_widget = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)

        font = QFont()
        font.setPointSize(10)

        if text:
            label = QLabel(text)
            label.setFont(font)
            input_layout.addWidget(label)

        line_edit = QLineEdit()
        line_edit.setEnabled(False)
        input_layout.addWidget(line_edit)

        select_button = QPushButton("...")
        select_button.setFont(font)
        select_button.setFixedWidth(50)
        select_button.clicked.connect(lambda: self.select_path(line_edit, is_folder, file_type))
        input_layout.addWidget(select_button)

        background_widget.setLayout(input_layout)
        layout.addWidget(background_widget)
        return background_widget

    def select_path(self, line_edit, is_folder, file_type):
        """
        Метод, инициирующий открытие системного окна выбора файла
        заданного типа либо папки.

        Аргументы:
            line_edit: Объект QLineEdit, в который выводится текст - путь
                до выбранного файла/папки.
            is_folder: True, если следует открыть системное окно выбора папки.
                False, если следует открыть системное окно выбора файла.
            file_type: Имеет эффект только есть is_folder=False. Определяет,
                файлы какого типа следует предлагать выбирать пользователю в
                системном окне выбора файла. Возможные значения: txt, pkl, wav.
        """
        if is_folder:
            selected_item = QFileDialog.getExistingDirectory(self, "Выберите папку")
        else:
            if file_type == "txt":
                selected_item, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Text Files (*.txt)")
            elif file_type == "pkl":
                selected_item, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "Pickle Files (*.pkl)")
            elif file_type == "arp":
                selected_item, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "ARP Files (*.ARP)")
            else:
                selected_item, _ = QFileDialog.getOpenFileName(self, "Выберите файл", "", "WAV Files (*.wav)")

        if selected_item:
            line_edit.setText(selected_item)

    def set_input_layout(self, text, placeholder, layout, is_editable=True, value=""):
        """
        Метод, отрисовывающий поле ввода значения параметра. Слева от такого
        поля отрисовывается текст (описание параметра). Если поле можно
        изменять, то значение 'is_editable' есть True. Если значение параметра
        фиксировано и не подлежит изменению со стороны пользователя, то
        'is_editable=False'. Если поле пустое (в него не введен никакой текст),
        то в нем отображается подсказка 'placeholder'.

        Аргументы:
            text: Текст, расположенный слева от текстового поля, описывающий
                ввод значения какого параметра ожидается.
            placeholder: Подсказка, выводимая в текстовом поле, если оно пустое.
            layout: Макет, в который будет добавлен виджет, содержащий описанные
                выше элементы интерфейса.
            is_editable: True, если текстовое поле можно изменять. False иначе.
            value: Значение, которое будет записано в текстовое поле.
                По-умолчанию: "", то есть текстовое поле пустое.

        Возвращаемое значение:
            Виджет, содержащий описанные выше элементы интерфейса.
        """
        background_widget = QWidget()
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        font = QFont()
        font.setPointSize(10)

        if text is not None and text != "":
            label = QLabel(text)
            label.setFont(font)
            hbox.addWidget(label)

        line_edit = QLineEdit()
        line_edit.setFont(font)
        line_edit.setPlaceholderText(placeholder)
        if value:
            line_edit.setText(value)
        if not is_editable:
            line_edit.setStyleSheet("background-color: #C7C7C7;")
        line_edit.setEnabled(is_editable)
        hbox.addWidget(line_edit)

        background_widget.setLayout(hbox)
        layout.addWidget(background_widget)
        return background_widget

    def set_button(self, text, layout):
        """
        Метод, отрисовывающий стилизированную кнопку
        синего цвета с заданным текстом, отрисованным
        на ней.

        Аргументы:
            text: Текст, который отрисовывается на кнопке.
            layout: Макет, в который будет добавлена кнопка.

        Возвращаемое значение:
            Виджет кнопки.
        """
        button = QPushButton(text, self)
        button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        font-size: 16px;
                        font-weight: bold;
                        border-radius: 10px;
                        padding: 10px;
                        border: 2px solid #2980b9;
                        margin-top: 20px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:pressed {
                        background-color: #1c5985;
                    }
                """)
        layout.addWidget(button)
        return button

    def show_alert(self, message):
        """
        Метод, выводящий диалоговое окно сообщения с
        текстом 'message' в этом окне.

        Аргументы:
            message: Текст в диалоговом окне сообщения.
        """
        QMessageBox.information(self, "Внимание!", message)

    def show_progress_dialog(self, title, text):
        """
        Метод создает и отображает окно ожидания, которое появляется
        при сохранении настроек модели из вкладки "Настройки".

        Аргументы:
            title: Заголовок окна.
            text: Текст, отображаемый в окне.

        Возвращаемое значение:
            Объект созданного окна, через который с этим окном можно
            взаимодействовать, например, закрывать его.
        """
        progress_window = WaitWindow(title, text, self)
        progress_window.show()
        QApplication.processEvents()
        return progress_window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = UILayer()
    gui.show()
    sys.exit(app.exec())
