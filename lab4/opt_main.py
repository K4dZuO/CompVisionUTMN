"""
Главное приложение с GUI для сегментации изображений (оптимизированная версия).
Использует оптимизированные функции из segmentation.core.optimized.
"""
import sys
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog,
                               QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
                               QGroupBox, QTabWidget, QScrollArea, QCheckBox, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Импортируем оптимизированные функции
from segmentation.core.optimized import (edge_segmentation_optimized, 
                                         threshold_iterative_vectorized,
                                         kmeans_1d_vectorized,
                                         adaptive_threshold_vectorized,
                                         smooth_histogram_vectorized)
from segmentation.core.edges import edge_segmentation  # Для prewitt (если не оптимизирован)
from segmentation.core.threshold_global import threshold_ptile, threshold_multilevel
from segmentation.core.threshold_adaptive import adaptive_threshold  # Fallback
from segmentation.core.hist_utils import (compute_histogram, smooth_histogram, 
                                          find_peaks, find_thresholds_between_peaks)


class ImageDisplayWidget(QWidget):
    """Виджет для отображения изображения."""
    
    def __init__(self, title="Image", parent_panel=None):
        super().__init__()
        self.parent_panel = parent_panel  # Ссылка на родительскую панель для синхронизации
        self.sibling_display = None  # Ссылка на другое изображение для синхронизации
        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(500, 200)  # Уменьшаем минимальный размер
        self.image_label.setScaledContents(False)  # Отключаем автоматическое масштабирование
        # Разрешаем виджету расширяться
        self.image_label.setSizePolicy(
            QSizePolicy.Expanding, 
            QSizePolicy.Expanding
        )
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.image_label, 1)  # Добавляем stretch factor
        self.setLayout(self.layout)
        
        # Сохраняем исходное изображение для пересчета при изменении размера
        self.original_pixmap = None
    
    def set_image(self, image: np.ndarray):
        """Устанавливает изображение для отображения."""
        if image is None:
            self.image_label.clear()
            self.original_pixmap = None
            return
        
        # Конвертируем в uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Конвертируем в QImage
        h, w = image.shape
        if len(image.shape) == 2:
            qimage = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimage = QImage(image.data, w, h, w * 3, QImage.Format_RGB888)
        
        # Сохраняем исходный pixmap
        self.original_pixmap = QPixmap.fromImage(qimage)
        
        # Масштабируем изображение
        if self.parent_panel is not None:
            self._scale_image_synchronized()
        else:
            self._scale_image()
    
    def _scale_image(self):
        """Масштабирует изображение под доступное пространство."""
        if self.original_pixmap is None:
            return
        
        # Получаем доступный размер с учетом отступов
        available_size = self.image_label.size()
        if available_size.width() <= 0 or available_size.height() <= 0:
            # Если размер еще не определен, устанавливаем минимальный размер
            return
        
        # Масштабируем с сохранением пропорций, чтобы изображение полностью помещалось
        scaled_pixmap = self.original_pixmap.scaled(
            available_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def _scale_image_synchronized(self):
        """Масштабирует изображение с учетом синхронизации с другими виджетами."""
        if self.original_pixmap is None:
            return
        
        # Если есть родительская панель, вычисляем общий доступный размер
        if self.parent_panel is not None:
            # Получаем размер панели
            panel_size = self.parent_panel.size()
            # Вычисляем доступное пространство для изображений (исключая гистограмму)
            # Приблизительно: общая высота минус отступы и гистограмма
            available_height = panel_size.height() - 380 - 120  # 380 для гистограммы, 120 для отступов и заголовков
            available_width = panel_size.width() - 30  # 30 для отступов
            
            # Делим высоту поровну между двумя изображениями
            single_image_height = max(200, available_height // 2)
            
            # Получаем реальный размер label'а
            available_size = self.image_label.size()
            if available_size.width() <= 0 or available_size.height() <= 0:
                return
            
            # Используем ширину label'а и вычисляем высоту с учетом пропорций
            final_width = min(available_size.width(), available_width)
            
            # Вычисляем максимальную высоту с учетом пропорций изображения
            pixmap_aspect = self.original_pixmap.width() / self.original_pixmap.height()
            max_height_by_width = final_width / pixmap_aspect
            
            # Выбираем минимальную из доступных высот
            final_height = min(available_size.height(), single_image_height, max_height_by_width)
            
            # Масштабируем с учетом ограничений
            scaled_pixmap = self.original_pixmap.scaled(
                int(final_width), int(final_height),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # Если есть "братский" виджет, пересчитываем и его
            if self.sibling_display is not None and self.sibling_display.original_pixmap is not None:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(5, self.sibling_display._scale_image_synchronized)
        else:
            # Стандартное масштабирование
            self._scale_image()
    
    def showEvent(self, event):
        """Переопределяем для правильного масштабирования при первом показе."""
        super().showEvent(event)
        if self.original_pixmap is not None:
            # Пересчитываем масштаб после того, как виджет стал видимым
            from PySide6.QtCore import QTimer
            if self.parent_panel is not None:
                QTimer.singleShot(10, self._scale_image_synchronized)
            else:
                QTimer.singleShot(10, self._scale_image)
    
    def resizeEvent(self, event):
        """Переопределяем для автоматического пересчета масштаба при изменении размера."""
        super().resizeEvent(event)
        # Пересчитываем масштаб после изменения размера
        if self.original_pixmap is not None:
            # Используем QTimer.singleShot для отложенного пересчета после завершения изменения размера
            from PySide6.QtCore import QTimer
            if self.parent_panel is not None:
                QTimer.singleShot(10, self._scale_image_synchronized)
            else:
                QTimer.singleShot(10, self._scale_image)


class HistogramWidget(QWidget):
    """Виджет для отображения гистограммы."""
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(10, 10, 10, 10)
        # Увеличиваем размер фигуры для лучшей видимости
        self.figure = plt.Figure(figsize=(8, 4))
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(600, 350)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
    
    def plot_histogram(self, hist: np.ndarray, title="Гистограмма"):
        """Отображает гистограмму."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(range(len(hist)), hist)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Интенсивность', fontsize=10)
        ax.set_ylabel('Частота', fontsize=10)
        # Убеждаемся, что подписи не обрезаются
        self.figure.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    """Главное окно приложения (оптимизированная версия)."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сегментация изображений (Оптимизированная версия)")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.original_image = None
        self.gray_image = None
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализирует интерфейс."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Левая панель - управление
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Правая панель - отображение
        display_panel = self.create_display_panel()
        main_layout.addWidget(display_panel, 2)
        
        central_widget.setLayout(main_layout)
    
    def create_control_panel(self) -> QWidget:
        """Создаёт панель управления."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Загрузка изображения
        load_btn = QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        load_btn.setMinimumHeight(35)
        layout.addWidget(load_btn)
        
        # Кнопка сброса
        reset_btn = QPushButton("Сброс настроек")
        reset_btn.clicked.connect(self.reset_settings)
        reset_btn.setMinimumHeight(35)
        layout.addWidget(reset_btn)
        
        # Вкладки для разных методов
        tabs = QTabWidget()
        
        # Вкладка 1: Сегментация по краям
        edges_tab = self.create_edges_tab()
        tabs.addTab(edges_tab, "Края")
        
        # Вкладка 2: Глобальные пороги
        global_tab = self.create_global_threshold_tab()
        tabs.addTab(global_tab, "Глобальные пороги")
        
        # Вкладка 3: Адаптивные пороги
        adaptive_tab = self.create_adaptive_threshold_tab()
        tabs.addTab(adaptive_tab, "Адаптивные пороги")
        
        layout.addWidget(tabs)
        panel.setLayout(layout)
        return panel
    
    def create_edges_tab(self) -> QWidget:
        """Создаёт вкладку для сегментации по краям."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Оператор
        operator_group = QGroupBox("Оператор")
        operator_layout = QVBoxLayout()
        operator_layout.setSpacing(8)
        operator_layout.setContentsMargins(10, 10, 10, 10)
        self.operator_combo = QComboBox()
        self.operator_combo.addItems(["sobel", "prewitt", "roberts"])
        operator_layout.addWidget(QLabel("Оператор:"))
        operator_layout.addWidget(self.operator_combo)
        operator_group.setLayout(operator_layout)
        layout.addWidget(operator_group)
        
        # Порог
        threshold_group = QGroupBox("Порог")
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(8)
        threshold_layout.setContentsMargins(10, 10, 10, 10)
        self.edge_threshold_slider = QSlider(Qt.Horizontal)
        self.edge_threshold_slider.setMinimum(0)
        self.edge_threshold_slider.setMaximum(200)
        self.edge_threshold_slider.setValue(50)
        self.edge_threshold_spinbox = QDoubleSpinBox()
        self.edge_threshold_spinbox.setMinimum(0)
        self.edge_threshold_spinbox.setMaximum(200)
        self.edge_threshold_spinbox.setValue(50)
        self.edge_threshold_slider.valueChanged.connect(
            lambda v: self.edge_threshold_spinbox.setValue(v))
        self.edge_threshold_spinbox.valueChanged.connect(
            lambda v: self.edge_threshold_slider.setValue(int(v)))
        threshold_layout.addWidget(QLabel("Порог градиента:"))
        threshold_layout.addWidget(self.edge_threshold_slider)
        threshold_layout.addWidget(self.edge_threshold_spinbox)
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Кнопка применения
        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.apply_edge_segmentation)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_global_threshold_tab(self) -> QWidget:
        """Создаёт вкладку для глобальной пороговой сегментации."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Метод
        method_group = QGroupBox("Метод")
        method_layout = QVBoxLayout()
        method_layout.setSpacing(8)
        method_layout.setContentsMargins(10, 10, 10, 10)
        self.global_method_combo = QComboBox()
        self.global_method_combo.addItems(["P-tile", "Последовательные приближения", "K-средних", "K-средних (многоуровневая)"])
        method_layout.addWidget(QLabel("Метод:"))
        method_layout.addWidget(self.global_method_combo)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        # P-tile параметры
        self.ptile_group = QGroupBox("P-tile параметры")
        ptile_layout = QVBoxLayout()
        ptile_layout.setSpacing(8)
        ptile_layout.setContentsMargins(10, 10, 10, 10)
        self.ptile_slider = QSlider(Qt.Horizontal)
        self.ptile_slider.setMinimum(0)
        self.ptile_slider.setMaximum(100)
        self.ptile_slider.setValue(30)
        self.ptile_spinbox = QDoubleSpinBox()
        self.ptile_spinbox.setMinimum(0)
        self.ptile_spinbox.setMaximum(100)
        self.ptile_spinbox.setValue(30)
        self.ptile_slider.valueChanged.connect(
            lambda v: self.ptile_spinbox.setValue(v))
        self.ptile_spinbox.valueChanged.connect(
            lambda v: self.ptile_slider.setValue(int(v)))
        ptile_layout.addWidget(QLabel("Процент P:"))
        ptile_layout.addWidget(self.ptile_slider)
        ptile_layout.addWidget(self.ptile_spinbox)
        self.ptile_group.setLayout(ptile_layout)
        layout.addWidget(self.ptile_group)
        
        # Итеративный метод параметры
        self.iterative_group = QGroupBox("Параметры итеративного метода")
        iterative_layout = QVBoxLayout()
        iterative_layout.setSpacing(8)
        iterative_layout.setContentsMargins(10, 10, 10, 10)
        self.iterative_eps_spinbox = QDoubleSpinBox()
        self.iterative_eps_spinbox.setMinimum(0.1)
        self.iterative_eps_spinbox.setMaximum(10.0)
        self.iterative_eps_spinbox.setValue(0.5)
        iterative_layout.addWidget(QLabel("Точность (eps):"))
        iterative_layout.addWidget(self.iterative_eps_spinbox)
        self.iterative_group.setLayout(iterative_layout)
        layout.addWidget(self.iterative_group)
        
        # K-means параметры
        self.kmeans_group = QGroupBox("Параметры K-средних")
        kmeans_layout = QVBoxLayout()
        kmeans_layout.setSpacing(8)
        kmeans_layout.setContentsMargins(10, 10, 10, 10)
        self.kmeans_k_spinbox = QSpinBox()
        self.kmeans_k_spinbox.setMinimum(2)
        self.kmeans_k_spinbox.setMaximum(10)
        self.kmeans_k_spinbox.setValue(2)
        kmeans_layout.addWidget(QLabel("Количество кластеров (k):"))
        kmeans_layout.addWidget(self.kmeans_k_spinbox)
        self.kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(self.kmeans_group)
        
        # Обновляем видимость групп при изменении метода
        self.global_method_combo.currentTextChanged.connect(self.update_global_params_visibility)
        self.update_global_params_visibility()
        
        # Параметры гистограммы
        hist_group = QGroupBox("Гистограмма")
        hist_layout = QVBoxLayout()
        hist_layout.setSpacing(8)
        hist_layout.setContentsMargins(10, 10, 10, 10)
        
        # Сглаживание
        self.hist_smooth_checkbox = QCheckBox("Сглаживать гистограмму")
        self.hist_smooth_window_spinbox = QSpinBox()
        self.hist_smooth_window_spinbox.setMinimum(3)
        self.hist_smooth_window_spinbox.setMaximum(51)
        self.hist_smooth_window_spinbox.setValue(5)
        self.hist_smooth_window_spinbox.setSingleStep(2)
        self.hist_smooth_window_spinbox.setEnabled(False)
        self.hist_smooth_checkbox.toggled.connect(
            lambda checked: self.hist_smooth_window_spinbox.setEnabled(checked))
        
        hist_layout.addWidget(self.hist_smooth_checkbox)
        hist_layout.addWidget(QLabel("Размер окна сглаживания:"))
        hist_layout.addWidget(self.hist_smooth_window_spinbox)
        
        # Поиск пиков
        self.hist_peaks_checkbox = QCheckBox("Показывать пики на гистограмме")
        
        hist_layout.addWidget(self.hist_peaks_checkbox)
        hist_group.setLayout(hist_layout)
        layout.addWidget(hist_group)
        
        # Подключаем обновление гистограммы
        self.hist_smooth_checkbox.toggled.connect(self.update_histogram_display)
        self.hist_smooth_window_spinbox.valueChanged.connect(self.update_histogram_display)
        self.hist_peaks_checkbox.toggled.connect(self.update_histogram_display)
        
        # Кнопка применения
        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.apply_global_threshold)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_adaptive_threshold_tab(self) -> QWidget:
        """Создаёт вкладку для адаптивной пороговой сегментации."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Тип статистики
        stat_group = QGroupBox("Тип статистики")
        stat_layout = QVBoxLayout()
        stat_layout.setSpacing(8)
        stat_layout.setContentsMargins(10, 10, 10, 10)
        self.adaptive_stat_combo = QComboBox()
        self.adaptive_stat_combo.addItems(["mean", "median", "avg_min_max"])
        stat_layout.addWidget(QLabel("Статистика:"))
        stat_layout.addWidget(self.adaptive_stat_combo)
        stat_group.setLayout(stat_layout)
        layout.addWidget(stat_group)
        
        # Размер окна
        window_group = QGroupBox("Размер окна")
        window_layout = QVBoxLayout()
        window_layout.setSpacing(8)
        window_layout.setContentsMargins(10, 10, 10, 10)
        self.adaptive_window_spinbox = QSpinBox()
        self.adaptive_window_spinbox.setMinimum(3)
        self.adaptive_window_spinbox.setMaximum(51)
        self.adaptive_window_spinbox.setValue(15)
        self.adaptive_window_spinbox.setSingleStep(2)
        window_layout.addWidget(QLabel("Размер окна:"))
        window_layout.addWidget(self.adaptive_window_spinbox)
        window_group.setLayout(window_layout)
        layout.addWidget(window_group)
        
        # Параметр C
        c_group = QGroupBox("Корректирующий параметр C")
        c_layout = QVBoxLayout()
        c_layout.setSpacing(8)
        c_layout.setContentsMargins(10, 10, 10, 10)
        self.adaptive_c_spinbox = QDoubleSpinBox()
        self.adaptive_c_spinbox.setMinimum(-50.0)
        self.adaptive_c_spinbox.setMaximum(50.0)
        self.adaptive_c_spinbox.setValue(0.0)
        c_layout.addWidget(QLabel("C:"))
        c_layout.addWidget(self.adaptive_c_spinbox)
        c_group.setLayout(c_layout)
        layout.addWidget(c_group)
        
        # Кнопка применения
        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.apply_adaptive_threshold)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_display_panel(self) -> QWidget:
        """Создаёт панель отображения."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Оригинальное изображение - равный stretch factor
        self.original_display = ImageDisplayWidget("Исходное изображение", panel)
        layout.addWidget(self.original_display, 1)  # Stretch factor = 1
        
        # Гистограмма - фиксированный размер
        self.histogram_widget = HistogramWidget()
        layout.addWidget(self.histogram_widget, 0)  # Stretch factor = 0 (фиксированный)
        
        # Результат - равный stretch factor
        self.result_display = ImageDisplayWidget("Результат сегментации", panel)
        layout.addWidget(self.result_display, 1)  # Stretch factor = 1
        
        # Сохраняем ссылку на панель для синхронизации масштабирования
        self.display_panel = panel
        
        # Сохраняем ссылки на виджеты изображений для синхронизации
        self.original_display.parent_panel = panel
        self.result_display.parent_panel = panel
        # Сохраняем ссылки друг на друга для синхронного масштабирования
        self.original_display.sibling_display = self.result_display
        self.result_display.sibling_display = self.original_display
        
        panel.setLayout(layout)
        return panel
    
    def load_image(self):
        """Загружает изображение."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)")
        
        if file_path:
            # Загружаем через cv2
            image = cv2.imread(file_path)
            if image is not None:
                self.original_image = image
                self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                self.original_display.set_image(self.gray_image)
                
                # Вычисляем и отображаем гистограмму
                self.update_histogram_display()
    
    def apply_edge_segmentation(self):
        """Применяет сегментацию по краям (оптимизированная версия)."""
        if self.gray_image is None:
            return
        
        operator = self.operator_combo.currentText()
        threshold = self.edge_threshold_spinbox.value()
        
        # Используем оптимизированную версию для sobel и roberts
        if operator in ['sobel', 'roberts']:
            result = edge_segmentation_optimized(self.gray_image, operator=operator, T_edge=threshold)
        else:
            # Для prewitt используем стандартную версию (если не оптимизирована)
            result = edge_segmentation(self.gray_image, operator=operator, T_edge=threshold)
        
        self.result_display.set_image(result)
    
    def apply_global_threshold(self):
        """Применяет глобальную пороговую сегментацию (оптимизированная версия)."""
        if self.gray_image is None:
            return
        
        method = self.global_method_combo.currentText()
        
        if method == "P-tile":
            P = self.ptile_spinbox.value()
            T, result = threshold_ptile(self.gray_image, P=P)
            print(f"Вычисленный порог: {T:.2f}")
        elif method == "Последовательные приближения":
            eps = self.iterative_eps_spinbox.value()
            # Используем оптимизированную версию
            T, result = threshold_iterative_vectorized(self.gray_image, eps=eps)
            print(f"Вычисленный порог: {T:.2f}")
        elif method == "K-средних":
            k = self.kmeans_k_spinbox.value()
            # Используем оптимизированную версию k-means
            flat = self.gray_image.flatten().astype(np.float64)
            centroids, labels = kmeans_1d_vectorized(flat, k=k)
            
            # Сортируем центроиды
            sorted_indices = np.argsort(centroids)
            sorted_centroids = centroids[sorted_indices]
            
            # Для бинаризации используем средний порог между кластерами
            if k == 2:
                T = sorted_centroids[0] + (sorted_centroids[1] - sorted_centroids[0]) / 2.0
            else:
                T = sorted_centroids[0] + (sorted_centroids[1] - sorted_centroids[0]) / 2.0
            
            # Бинаризация
            binary = np.zeros_like(self.gray_image, dtype=np.uint8)
            for i in range(k):
                orig_idx = sorted_indices[i]
                mask = (labels == orig_idx).reshape(self.gray_image.shape)
                if i >= k // 2:  # Верхние классы
                    binary[mask] = 255
            
            result = binary
            print(f"Вычисленный порог: {T:.2f}")
        elif method == "K-средних (многоуровневая)":
            k = self.kmeans_k_spinbox.value()
            thresholds, result = threshold_multilevel(self.gray_image, k=k)
            print(f"Вычисленные пороги: {thresholds}")
        
        self.result_display.set_image(result)
    
    def apply_adaptive_threshold(self):
        """Применяет адаптивную пороговую сегментацию (оптимизированная версия)."""
        if self.gray_image is None:
            return
        
        stat_type = self.adaptive_stat_combo.currentText()
        window_size = self.adaptive_window_spinbox.value()
        C = self.adaptive_c_spinbox.value()
        
        # Используем оптимизированную версию
        result = adaptive_threshold_vectorized(self.gray_image, window_size=window_size, 
                                             stat_type=stat_type, C=C)
        self.result_display.set_image(result)
    
    def update_global_params_visibility(self):
        """Обновляет видимость групп параметров в зависимости от выбранного метода."""
        method = self.global_method_combo.currentText()
        
        self.ptile_group.setVisible(method == "P-tile")
        self.iterative_group.setVisible(method == "Последовательные приближения")
        self.kmeans_group.setVisible(method == "K-средних" or method == "K-средних (многоуровневая)")
    
    def update_histogram_display(self):
        """Обновляет отображение гистограммы с учётом параметров."""
        if self.gray_image is None:
            return
        
        hist = compute_histogram(self.gray_image)
        
        # Сглаживание (используем оптимизированную версию)
        if hasattr(self, 'hist_smooth_checkbox') and self.hist_smooth_checkbox.isChecked():
            window_size = self.hist_smooth_window_spinbox.value()
            hist = smooth_histogram_vectorized(hist, window_size=window_size)
        
        # Отображаем гистограмму
        self.histogram_widget.plot_histogram(hist, "Гистограмма")
        
        # Показываем пики, если включено
        if hasattr(self, 'hist_peaks_checkbox') and self.hist_peaks_checkbox.isChecked():
            peaks = find_peaks(hist)
            if len(peaks) > 0:
                # Добавляем маркеры пиков на гистограмму
                ax = self.histogram_widget.figure.axes[0]
                ax.plot(peaks, hist[peaks], 'ro', markersize=8, label='Пики')
                ax.legend()
                self.histogram_widget.canvas.draw()
    
    def reset_settings(self):
        """Сбрасывает все настройки до исходных значений и очищает поле результатов."""
        # Сброс настроек для сегментации по краям
        if hasattr(self, 'operator_combo'):
            self.operator_combo.setCurrentIndex(0)  # sobel
        if hasattr(self, 'edge_threshold_slider'):
            self.edge_threshold_slider.setValue(50)
        if hasattr(self, 'edge_threshold_spinbox'):
            self.edge_threshold_spinbox.setValue(50.0)
        
        # Сброс настроек для глобальной пороговой сегментации
        if hasattr(self, 'global_method_combo'):
            self.global_method_combo.setCurrentIndex(0)  # P-tile
        if hasattr(self, 'ptile_slider'):
            self.ptile_slider.setValue(30)
        if hasattr(self, 'ptile_spinbox'):
            self.ptile_spinbox.setValue(30.0)
        if hasattr(self, 'iterative_eps_spinbox'):
            self.iterative_eps_spinbox.setValue(0.5)
        if hasattr(self, 'kmeans_k_spinbox'):
            self.kmeans_k_spinbox.setValue(2)
        if hasattr(self, 'hist_smooth_checkbox'):
            self.hist_smooth_checkbox.setChecked(False)
        if hasattr(self, 'hist_smooth_window_spinbox'):
            self.hist_smooth_window_spinbox.setValue(5)
        if hasattr(self, 'hist_peaks_checkbox'):
            self.hist_peaks_checkbox.setChecked(False)
        
        # Сброс настроек для адаптивной пороговой сегментации
        if hasattr(self, 'adaptive_stat_combo'):
            self.adaptive_stat_combo.setCurrentIndex(0)  # mean
        if hasattr(self, 'adaptive_window_spinbox'):
            self.adaptive_window_spinbox.setValue(15)
        if hasattr(self, 'adaptive_c_spinbox'):
            self.adaptive_c_spinbox.setValue(0.0)
        
        # Обновляем видимость параметров
        if hasattr(self, 'update_global_params_visibility'):
            self.update_global_params_visibility()
        
        # Очищаем поле результатов
        self.result_display.set_image(None)
        
        # Обновляем гистограмму с исходными настройками
        if self.gray_image is not None:
            self.update_histogram_display()


def main():
    """Главная функция."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

