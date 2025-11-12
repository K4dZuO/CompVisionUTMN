"""
Главная точка входа приложения для анализа оптического потока.

Запуск приложения:
------------------
python main.py

Зависимости:
------------
- PyQt5 для GUI
- OpenCV для обработки видео и изображений
- NumPy для численных вычислений
- Matplotlib для визуализации (опционально)

Оптимизации:
------------
- Асинхронная обработка для избежания блокировки UI
- Эффективное управление памятью
- Оптимизированные файловые операции
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import OpticalFlowMainWindow


def main():
    """Главная функция приложения."""
    # Создание приложения Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Анализ оптического потока")
    
    # Создание главного окна
    window = OpticalFlowMainWindow()
    window.show()
    
    # Запуск event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

