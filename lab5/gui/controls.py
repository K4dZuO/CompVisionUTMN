"""
Модуль с элементами управления для GUI.

Содержит специализированные виджеты для:
- Управления параметрами алгоритмов
- Навигации по видео
- Выбора режимов визуализации
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QSpinBox, QDoubleSpinBox, QPushButton,
                             QComboBox, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class ParameterSlider(QWidget):
    """Виджет слайдера с меткой для параметра."""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, label: str, min_val: float, max_val: float,
                 default_val: float, step: float = 0.1, decimals: int = 2):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.decimals = decimals
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Метка
        self.label = QLabel(label)
        layout.addWidget(self.label)
        
        # Горизонтальный layout для слайдера и значения
        h_layout = QHBoxLayout()
        
        # Слайдер
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_val / step))
        self.slider.setMaximum(int(max_val / step))
        self.slider.setValue(int(default_val / step))
        self.slider.valueChanged.connect(self._on_slider_changed)
        h_layout.addWidget(self.slider)
        
        # Отображение значения
        self.value_label = QLabel(f"{default_val:.{decimals}f}")
        self.value_label.setMinimumWidth(60)
        h_layout.addWidget(self.value_label)
        
        layout.addLayout(h_layout)
        
        self._value = default_val
    
    def _on_slider_changed(self, value):
        """Обработка изменения слайдера."""
        self._value = value * self.step
        self.value_label.setText(f"{self._value:.{self.decimals}f}")
        self.valueChanged.emit(self._value)
    
    def get_value(self) -> float:
        """Получение текущего значения."""
        return self._value
    
    def set_value(self, value: float):
        """Установка значения."""
        value = max(self.min_val, min(self.max_val, value))
        self._value = value
        self.slider.setValue(int(value / self.step))
        self.value_label.setText(f"{value:.{self.decimals}f}")


class AlgorithmParametersWidget(QWidget):
    """Виджет для управления параметрами алгоритмов."""
    
    parametersChanged = pyqtSignal(str, dict)  # algorithm_name, parameters
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Группа для параметров Хорна-Шанка
        self.hs_group = QGroupBox("Алгоритм Хорна-Шанка")
        hs_layout = QVBoxLayout()
        self.hs_group.setLayout(hs_layout)
        
        self.hs_lambda = ParameterSlider("Lambda (λ)", 0.1, 10.0, 1.0, 0.1, 2)
        self.hs_lambda.valueChanged.connect(lambda v: self._on_param_changed('horn_schunck', 'lambda', v))
        hs_layout.addWidget(self.hs_lambda)
        
        self.hs_iterations = ParameterSlider("Итерации", 10, 200, 100, 10, 0)
        self.hs_iterations.valueChanged.connect(lambda v: self._on_param_changed('horn_schunck', 'iterations', int(v)))
        hs_layout.addWidget(self.hs_iterations)
        
        layout.addWidget(self.hs_group)
        
        # Группа для параметров Лукаса-Канаде
        self.lk_group = QGroupBox("Алгоритм Лукаса-Канаде")
        lk_layout = QVBoxLayout()
        self.lk_group.setLayout(lk_layout)
        
        self.lk_window_size = ParameterSlider("Размер окна", 3, 31, 15, 2, 0)
        self.lk_window_size.valueChanged.connect(lambda v: self._on_param_changed('lucas_kanade', 'window_size', int(v)))
        lk_layout.addWidget(self.lk_window_size)
        
        self.lk_max_level = ParameterSlider("Уровни пирамиды", 0, 4, 2, 1, 0)
        self.lk_max_level.valueChanged.connect(lambda v: self._on_param_changed('lucas_kanade', 'max_level', int(v)))
        lk_layout.addWidget(self.lk_max_level)
        
        self.lk_max_corners = ParameterSlider("Макс. точек", 100, 2000, 500, 100, 0)
        self.lk_max_corners.valueChanged.connect(lambda v: self._on_param_changed('lucas_kanade', 'max_corners', int(v)))
        lk_layout.addWidget(self.lk_max_corners)
        
        layout.addWidget(self.lk_group)
        
        # Растягивание вниз
        layout.addStretch()
    
    def _on_param_changed(self, algorithm: str, param_name: str, value):
        """Обработка изменения параметра."""
        # Эмиссия сигнала будет реализована в главном окне
        pass
    
    def get_horn_schunck_params(self) -> dict:
        """Получение параметров Хорна-Шанка."""
        return {
            'lambda': self.hs_lambda.get_value(),
            'iterations': int(self.hs_iterations.get_value())
        }
    
    def get_lucas_kanade_params(self) -> dict:
        """Получение параметров Лукаса-Канаде."""
        return {
            'window_size': int(self.lk_window_size.get_value()),
            'max_level': int(self.lk_max_level.get_value()),
            'max_corners': int(self.lk_max_corners.get_value())
        }


class VideoControlsWidget(QWidget):
    """Виджет для управления воспроизведением видео."""
    
    playClicked = pyqtSignal()
    pauseClicked = pyqtSignal()
    stopClicked = pyqtSignal()
    frameChanged = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.max_frames = 0
        self.current_frame = 0
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Верхний ряд: кнопки управления
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.playClicked)
        controls_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("⏸")
        self.pause_btn.clicked.connect(self.pauseClicked)
        controls_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.clicked.connect(self.stopClicked)
        controls_layout.addWidget(self.stop_btn)

        main_layout.addLayout(controls_layout)

        # Нижний ряд: слайдер и метка кадра
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(QLabel("Кадр:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimumWidth(150)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        bottom_layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(60)
        bottom_layout.addWidget(self.frame_label)

        main_layout.addLayout(bottom_layout)
    
    def set_max_frames(self, max_frames: int):
        """Установка максимального количества кадров."""
        self.max_frames = max_frames
        self.frame_slider.setMaximum(max_frames - 1)
        self.update_frame_label()
    
    def set_current_frame(self, frame_idx: int):
        """Установка текущего кадра."""
        self.current_frame = frame_idx
        self.frame_slider.setValue(frame_idx)
        self.update_frame_label()
    
    def _on_frame_slider_changed(self, value):
        """Обработка изменения слайдера кадра."""
        self.current_frame = value
        self.update_frame_label()
        self.frameChanged.emit(value)
    
    def update_frame_label(self):
        """Обновление метки кадра."""
        self.frame_label.setText(f"{self.current_frame} / {self.max_frames - 1}")


class VisualizationControlsWidget(QWidget):
    """Виджет для управления визуализацией."""
    
    visualizationChanged = pyqtSignal(str, dict)  # mode, parameters
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Выбор режима визуализации
        layout.addWidget(QLabel("Режим визуализации:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "HSV плотный поток",
            "Стрелки на сетке",
            "Heat map",
            "Разреженный поток"
        ])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)
        
        # Параметры визуализации
        self.scale_slider = ParameterSlider("Масштаб", 0.1, 5.0, 1.0, 0.1, 2)
        self.scale_slider.valueChanged.connect(self._on_params_changed)
        layout.addWidget(self.scale_slider)
        
        self.alpha_slider = ParameterSlider("Прозрачность", 0.0, 1.0, 0.7, 0.1, 2)
        self.alpha_slider.valueChanged.connect(self._on_params_changed)
        layout.addWidget(self.alpha_slider)
        
        layout.addStretch()
    
    def _on_mode_changed(self, mode: str):
        """Обработка изменения режима."""
        self._on_params_changed()
    
    def _on_params_changed(self):
        """Обработка изменения параметров."""
        mode = self.mode_combo.currentText()
        params = {
            'scale': self.scale_slider.get_value(),
            'alpha': self.alpha_slider.get_value()
        }
        self.visualizationChanged.emit(mode, params)
    
    def get_visualization_params(self) -> tuple:
        """Получение параметров визуализации."""
        return self.mode_combo.currentText(), {
            'scale': self.scale_slider.get_value(),
            'alpha': self.alpha_slider.get_value()
        }

