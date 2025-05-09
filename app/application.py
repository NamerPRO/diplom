from PyQt6.QtWidgets import QApplication

from app.controller import Controller
from app.logic import LogicLayer
from app.ui import UILayer

if __name__ == "__main__":
    """
    Точка входа в приложение.
    
    1. Создаётся экземпляр QApplication для управления главным циклом событий GUI.
    2. Создаются экземпляры логического слоя (LogicLayer) и слоя пользовательского интерфейса (UILayer).
    3. Создаётся контроллер (Controller), который связывает UI и логику.
    4. Отображается UI.
    5. Запускается главный цикл событий приложения.
    """
    app = QApplication([])

    logic = LogicLayer()
    ui = UILayer()

    controller = Controller(ui, logic)

    ui.show()
    app.exec()

