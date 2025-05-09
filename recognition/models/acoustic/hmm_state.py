class HMMState:
    """
    Предствляет одно скрытое состояние скрытой марковской модели.
    """

    def __init__(self, name, gmm, initial_probability, is_start_state=False, is_final_state=False):
        """
        Инициализация состояния скрытой марковской модели.

        Аргументы:
            name: Уникальное имя состояния скрытой марковской модели.
            gmm: Модель гауссовой смеси, связанная с этим состоянием.
            initial_probability: Вероятность начала в этом состоянии.
                Сумма таких вероятностей для всех состояний HMM должна быть равна 1.
            is_start_state: True, если состояние является начальным. False в противном
                случае. В начальное состояние возможен переход из любого финального состояния.
            is_final_state: True, если состояние является финальным. False в противном случае.
                Из финального состояния возможен переход в любое начальное состояние.
        """
        self.name = name
        self.gmm = gmm
        self.initial_probability = initial_probability
        self.is_start_state = is_start_state
        self.is_final_state = is_final_state
