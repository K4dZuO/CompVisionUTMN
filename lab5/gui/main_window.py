"""
Главное окно приложения для анализа оптического потока.

АРХИТЕКТУРА GUI:
================

Компоненты:
-----------
1. Панель управления видео (загрузка, воспроизведение, навигация)
2. Панель параметров алгоритмов (слайдеры для настройки)
3. Область отображения исходного видео
4. Область отображения результатов анализа
5. Панель управления визуализацией

Оптимизации:
------------
- Асинхронная обработка для избежания блокировки UI
- Кэширование результатов обработки
- Эффективное обновление изображений
- Оптимизация памяти при работе с видео
"""

import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QSplitter, QScrollArea, QProgressBar, QComboBox, QCheckBox,
                             QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import time
from typing import Optional, Tuple

from core.video_processor import VideoController
from core.horn_schunck import HornSchunckProcessor
from core.lucas_kanade import LucasKanadeProcessor
from core.farneback import FarnebackProcessor
from core.object_tracker import ObjectTracker
from utils.visualization import VisualizationEngine
from utils.report_generator import ReportGenerator
from gui.controls import (AlgorithmParametersWidget, VideoControlsWidget,
                         VisualizationControlsWidget)


class ResizableImageLabel(QLabel):
    """QLabel, который масштабирует изображение с сохранением пропорций."""
    
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._original_pixmap = None
    
    def setPixmap(self, pixmap):
        """Установка изображения с сохранением оригинала."""
        self._original_pixmap = pixmap
        self.update_display()
        
    def resizeEvent(self, event):
        """Обработка изменения размера виджета."""
        self.update_display()
        super().resizeEvent(event)
        
    def update_display(self):
        """Обновление отображаемого изображения с учетом текущего размера."""
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            # Масштабируем с сохранением пропорций (KeepAspectRatio)
            scaled_pixmap = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)


class ProcessingThread(QThread):
    """Поток для асинхронной обработки оптического потока."""
    
    finished = pyqtSignal(object, str)  # results, algorithm_name
    progress = pyqtSignal(int)  # progress percentage
    error = pyqtSignal(str)  # error message
    
    def __init__(self, frame1: np.ndarray, frame2: np.ndarray,
                 algorithm: str, algorithm_params: dict,
                 tracker: Optional[ObjectTracker] = None,
                 tracker_params: Optional[dict] = None):
        super().__init__()
        self.frame1 = frame1.copy()
        self.frame2 = frame2.copy()
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
        self.tracker = tracker
        self.tracker_params = tracker_params
    
    def run(self):
        """Выполнение обработки в отдельном потоке."""
        try:
            start_time = time.time()
            results = {}
            
            # 1. Вычисление оптического потока
            if self.algorithm == 'horn_schunck':
                processor = HornSchunckProcessor(
                    lambda_val=self.algorithm_params.get('lambda', 1.0),
                    num_iterations=self.algorithm_params.get('iterations', 100),
                    threshold=self.algorithm_params.get('threshold', 0.001)
                )
                u, v, magnitude, angle = processor.compute_flow_magnitude_direction(
                    self.frame1, self.frame2
                )
                results.update({
                    'u': u,
                    'v': v,
                    'magnitude': magnitude,
                    'angle': angle
                })
                
                # Подготовка маски для трекера (если нужно)
                if self.tracker_params and self.tracker_params.get('enabled'):
                    # Нормализация magnitude для лучшего порога
                    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    
                    # Адаптивный порог вместо фиксированного
                    # Используем Otsu для автоматического определения порога
                    _, mask = cv2.threshold(magnitude_norm.astype(np.uint8), 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Улучшенная морфология для объединения частей объектов
                    # 1. Закрытие (closing) - заполняет дыры внутри объектов
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                    
                    # 2. Открытие (opening) - убирает мелкий шум
                    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                    
                    # 3. Дилатация - расширяет объекты для лучшего объединения
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate, iterations=2)
                    
                    results['mask'] = mask
                
            elif self.algorithm == 'lucas_kanade':
                processor = LucasKanadeProcessor(
                    window_size=self.algorithm_params.get('window_size', 15),
                    max_level=self.algorithm_params.get('max_level', 2),
                    max_corners=self.algorithm_params.get('max_corners', 500)
                )
                points, vectors, magnitudes = processor.compute_flow_vectors(
                    self.frame1, self.frame2
                )
                results.update({
                    'points': points,
                    'vectors': vectors,
                    'magnitudes': magnitudes
                })
                
                # Подготовка маски для трекера (если нужно)
                # Для LK создаем маску из точек
                if self.tracker_params and self.tracker_params.get('enabled'):
                    mask = np.zeros(self.frame1.shape[:2], dtype=np.uint8)
                    if len(points) > 0:
                        # Рисуем точки
                        for pt in points:
                            cv2.circle(mask, tuple(pt.astype(int)), 3, 255, -1)
                        # Дилатация для объединения близких точек
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
                    results['mask'] = mask
            
            elif self.algorithm == 'farneback':
                processor = FarnebackProcessor(
                    pyr_scale=self.algorithm_params.get('pyr_scale', 0.5),
                    levels=self.algorithm_params.get('levels', 3),
                    winsize=self.algorithm_params.get('winsize', 15),
                    iterations=self.algorithm_params.get('iterations', 3),
                    poly_n=self.algorithm_params.get('poly_n', 5),
                    poly_sigma=self.algorithm_params.get('poly_sigma', 1.2)
                )
                u, v, magnitude, angle = processor.compute_flow_magnitude_direction(
                    self.frame1, self.frame2
                )
                results.update({
                    'u': u,
                    'v': v,
                    'magnitude': magnitude,
                    'angle': angle
                })
                
                # Подготовка маски для трекера (если нужно)
                if self.tracker_params and self.tracker_params.get('enabled'):
                    # Нормализация magnitude для лучшего порога
                    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                    
                    # Адаптивный порог Otsu
                    _, mask = cv2.threshold(magnitude_norm.astype(np.uint8), 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Улучшенная морфология (такая же как для Horn-Schunck)
                    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                    
                    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                    
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate, iterations=2)
                    
                    results['mask'] = mask
            
            else:
                raise ValueError(f"Неизвестный алгоритм: {self.algorithm}")
            
            # 2. Отслеживание объектов (если включено)
            if self.tracker_params and self.tracker_params.get('enabled') and self.tracker is not None:
                # Обновляем параметры трекера
                self.tracker.min_area = self.tracker_params.get('min_area', 500)
                self.tracker.tracker.max_disappeared = self.tracker_params.get('max_disappeared', 40)
                self.tracker.tracker.max_distance = self.tracker_params.get('max_distance', 50)
                
                if 'mask' in results:
                    tracked_objects, paths = self.tracker.update_from_mask(results['mask'])
                    results['tracked_objects'] = tracked_objects
                    results['paths'] = paths
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            
            self.finished.emit(results, self.algorithm)
            
        except Exception as e:
            self.error.emit(str(e))


class OpticalFlowMainWindow(QMainWindow):
    """Главное окно приложения."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ оптического потока")
        self.setGeometry(100, 100, 1400, 900)
        
        # Инициализация компонентов
        self.video_controller: Optional[VideoController] = None
        self.hs_processor: Optional[HornSchunckProcessor] = None
        self.lk_processor: Optional[LucasKanadeProcessor] = None
        self.visualization_engine = VisualizationEngine()
        self.report_generator = ReportGenerator()
        self.object_tracker = ObjectTracker()
        
        # Текущие результаты обработки
        self.current_results: Optional[dict] = None
        self.current_algorithm: Optional[str] = None
        self.processing_thread: Optional[ProcessingThread] = None
        
        # Таймер для воспроизведения
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.on_play_timer)
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Левая панель: параметры и управление (в ScrollArea)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(400)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        left_content = QWidget()
        left_layout = QVBoxLayout()
        left_content.setLayout(left_layout)
        left_scroll.setWidget(left_content)
        
        # Кнопка загрузки видео
        self.load_btn = QPushButton("Загрузить видео")
        self.load_btn.clicked.connect(self.load_video)
        left_layout.addWidget(self.load_btn)
        
        # Управление видео
        self.video_controls = VideoControlsWidget()
        self.video_controls.playClicked.connect(self.play_video)
        self.video_controls.pauseClicked.connect(self.pause_video)
        self.video_controls.stopClicked.connect(self.stop_video)
        self.video_controls.frameChanged.connect(self.on_frame_changed)
        left_layout.addWidget(self.video_controls)
        
        # Выбор алгоритма
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Хорн-Шанк", "Лукас-Канаде", "Farneback (OpenCV)"])
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        left_layout.addWidget(QLabel("Алгоритм:"))
        left_layout.addWidget(self.algorithm_combo)
        
        # Параметры алгоритмов
        self.algorithm_params = AlgorithmParametersWidget()
        left_layout.addWidget(self.algorithm_params)
        
        # Кнопка обработки
        self.process_btn = QPushButton("Обработать кадр")
        self.process_btn.clicked.connect(self.process_current_frame)
        left_layout.addWidget(self.process_btn)
        
        # Статус обработки
        self.status_label = QLabel("Готов к обработке")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        # Чекбокс автоматической повторной обработки
        self.auto_reprocess_checkbox = QCheckBox("Авто-обработка при смене кадра")
        self.auto_reprocess_checkbox.setToolTip("Автоматически обрабатывать кадр при переходе к другому кадру")
        left_layout.addWidget(self.auto_reprocess_checkbox)
        
        # Кнопка сброса трекера
        self.reset_tracker_btn = QPushButton("🔄 Сбросить историю трекинга")
        self.reset_tracker_btn.setToolTip("Очистить все траектории и начать отслеживание заново")
        self.reset_tracker_btn.clicked.connect(self.reset_tracker)
        left_layout.addWidget(self.reset_tracker_btn)
        
        # Кнопка экспорта
        self.export_btn = QPushButton("Экспорт отчёта")
        self.export_btn.clicked.connect(self.export_report)
        left_layout.addWidget(self.export_btn)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Управление визуализацией
        self.visualization_controls = VisualizationControlsWidget()
        self.visualization_controls.visualizationChanged.connect(self.on_visualization_changed)
        left_layout.addWidget(self.visualization_controls)
        
        left_layout.addStretch()
        
        main_layout.addWidget(left_scroll)
        
        # Правая панель: отображение
        right_panel = QSplitter(Qt.Horizontal)
        
        # Исходное видео
        self.original_label = ResizableImageLabel("Исходное видео")
        self.original_label.setStyleSheet("border: 1px solid gray")
        
        original_container = QWidget()
        original_layout = QVBoxLayout()
        original_layout.addWidget(self.original_label)
        original_container.setLayout(original_layout)
        
        # Результаты
        self.result_label = ResizableImageLabel("Результаты анализа")
        self.result_label.setStyleSheet("border: 1px solid gray")
        
        result_container = QWidget()
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        result_container.setLayout(result_layout)
        
        right_panel.addWidget(original_container)
        right_panel.addWidget(result_container)
        
        main_layout.addWidget(right_panel, stretch=1)
    
    def load_video(self):
        """Загрузка видеофайла."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Выберите видеофайл", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if filename:
            self.video_controller = VideoController()
            self.object_tracker = ObjectTracker()  # Сброс трекера при загрузке нового видео
            if self.video_controller.load_video(filename):
                # Обновление UI
                metadata = self.video_controller.get_metadata()
                self.video_controls.set_max_frames(metadata['frame_count'])
                self.video_controls.set_current_frame(0)
                
                # Отображение первого кадра
                frame = self.video_controller.get_current_frame()
                if frame is not None:
                    self.display_frame(frame, self.original_label)
                
                # Добавление метаданных в отчёт
                self.report_generator.add_video_metadata(metadata)
                
                QMessageBox.information(self, "Успех", "Видео загружено успешно")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить видео")

    def export_report(self):
        """Экспорт отчёта."""
        if not self.report_generator.data['frames']:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            if self.report_generator.save_report(filename):
                QMessageBox.information(self, "Успех", "Отчёт сохранён успешно")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить отчёт")

    def display_frame(self, frame: np.ndarray, label: QLabel):
        """Отображение кадра в QLabel."""
        if frame is None:
            return
            
        # Конвертация BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Создание QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Отображение
        label.setPixmap(QPixmap.fromImage(q_image))

    def play_video(self):
        """Воспроизведение видео."""
        if self.video_controller is None:
            return
        
        metadata = self.video_controller.get_metadata()
        fps = metadata.get('fps', 30)
        interval = int(1000 / fps)  # миллисекунды
        
        self.play_timer.start(interval)
        self.video_controls.play_btn.setEnabled(False)
        self.video_controls.pause_btn.setEnabled(True)
    
    def pause_video(self):
        """Пауза воспроизведения."""
        self.play_timer.stop()
        self.video_controls.play_btn.setEnabled(True)
        self.video_controls.pause_btn.setEnabled(False)
    
    def stop_video(self):
        """Остановка воспроизведения."""
        self.play_timer.stop()
        if self.video_controller is not None:
            self.video_controller.set_frame(0)
            self.video_controls.set_current_frame(0)
            frame = self.video_controller.get_current_frame()
            if frame is not None:
                self.display_frame(frame, self.original_label)
        self.video_controls.play_btn.setEnabled(True)
        self.video_controls.pause_btn.setEnabled(False)
    
    def reset_tracker(self):
        """Сброс трекера для начала новой истории отслеживания."""
        if self.object_tracker is not None:
            # Создаем новый трекер с текущими параметрами
            tracker_params = self.algorithm_params.get_tracker_params()
            self.object_tracker = ObjectTracker(
                min_area=tracker_params.get('min_area', 500),
                max_disappeared=tracker_params.get('max_disappeared', 40),
                max_distance=tracker_params.get('max_distance', 50)
            )
            
            # Обновляем статус
            self.status_label.setText("🔄 История трекинга сброшена")
            self.status_label.setStyleSheet("color: blue; font-style: italic;")
            
            # Если есть результаты, очищаем их
            if self.current_results is not None:
                self.current_results = None
                self.result_label.clear()
                self.result_label.setText("Результаты анализа\n\n(Нажмите 'Обработать кадр' для начала новой истории)")
    
    def on_play_timer(self):
        """Обработка таймера воспроизведения."""
        if self.video_controller is None:
            return
        
        frame = self.video_controller.get_next_frame()
        if frame is not None:
            self.display_frame(frame, self.original_label)
            self.video_controls.set_current_frame(self.video_controller.current_frame_idx)
        else:
            # Конец видео
            self.pause_video()
    
    def on_frame_changed(self, frame_idx: int):
        """Обработка изменения кадра."""
        if self.video_controller is not None:
            # Устанавливаем новый кадр в контроллере
            if self.video_controller.set_frame(frame_idx):
                frame = self.video_controller.get_current_frame()
                if frame is not None:
                    self.display_frame(frame, self.original_label)
                    
                    # Если включена авто-обработка и был выбран алгоритм
                    if self.auto_reprocess_checkbox.isChecked() and self.current_algorithm is not None:
                        # Автоматически обрабатываем новый кадр
                        self.process_current_frame()
                    else:
                        # НЕ сбрасываем результаты! Оставляем старую визуализацию
                        # Только обновляем статус, чтобы пользователь знал, что результаты для другого кадра
                        if self.current_results is not None:
                            # Показываем, что результаты устарели
                            old_frame = getattr(self, 'last_processed_frame', None)
                            if old_frame is not None and old_frame != frame_idx:
                                self.status_label.setText(
                                    f"⚠ Показаны результаты кадра {old_frame}, текущий кадр: {frame_idx}"
                                )
                                self.status_label.setStyleSheet("color: orange; font-style: italic;")
                        else:
                            # Нет результатов вообще
                            self.status_label.setText(f"Кадр {frame_idx} - требуется обработка")
                            self.status_label.setStyleSheet("color: gray; font-style: italic;")
    
    def on_visualization_changed(self, mode: str, params: dict):
        """Обработка изменения параметров визуализации."""
        if self.current_results is not None:
            self.update_visualization()
    
    def on_algorithm_changed(self, index: int):
        """Обработка смены алгоритма."""
        # Обновляем видимость параметров
        self.algorithm_params.set_visible_algorithm(index)
    
    def process_current_frame(self):
        """Обработка текущего кадра."""
        if self.video_controller is None:
            QMessageBox.warning(self, "Предупреждение", "Загрузите видео сначала")
            return
        
        # Получение пары кадров
        current_idx = self.video_controller.current_frame_idx
        frame_pair = self.video_controller.get_frame_pair(current_idx)
        
        if frame_pair is None:
            QMessageBox.warning(self, "Предупреждение", "Недостаточно кадров для обработки")
            return
        
        frame1, frame2 = frame_pair
        
        # Определение алгоритма
        algorithm_text = self.algorithm_combo.currentText()
        
        # Получение параметров
        if "Хорн" in algorithm_text or "horn" in algorithm_text.lower():
            algorithm_name = "horn_schunck"
            params = self.algorithm_params.get_horn_schunck_params()
        elif "Лукас" in algorithm_text or "lucas" in algorithm_text.lower():
            algorithm_name = "lucas_kanade"
            params = self.algorithm_params.get_lucas_kanade_params()
        elif "Farneback" in algorithm_text or "farneback" in algorithm_text.lower():
            algorithm_name = "farneback"
            params = self.algorithm_params.get_farneback_params()
        else:
            # Fallback
            algorithm_name = "horn_schunck"
            params = self.algorithm_params.get_horn_schunck_params()
            
        tracker_params = self.algorithm_params.get_tracker_params()
        
        # Добавление параметров в отчёт
        self.report_generator.add_algorithm_parameters(algorithm_name, params)
        
        # Обновление статуса
        self.status_label.setText(f"Обработка кадра {current_idx}...")
        self.status_label.setStyleSheet("color: blue; font-style: italic;")
        
        # Запуск обработки в отдельном потоке
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Бесконечный прогресс
        
        # НЕ создаем новый трекер! Используем существующий для накопления траекторий
        # Трекер сбрасывается только при загрузке нового видео
        
        self.processing_thread = ProcessingThread(
            frame1, frame2, algorithm_name, params, 
            self.object_tracker, tracker_params
        )
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_processing_finished(self, results: dict, algorithm_name: str):
        """Обработка завершения обработки."""
        self.current_results = results
        self.current_algorithm = algorithm_name
        
        # Сохраняем номер обработанного кадра
        self.last_processed_frame = self.video_controller.current_frame_idx
        
        # Добавление метрик в отчёт
        if 'execution_time' in results:
            self.report_generator.add_metrics(algorithm_name, {
                'execution_time': results['execution_time']
            })
        
        # Визуализация результатов
        self.update_visualization()
        
        # Обновление статуса
        exec_time = results.get('execution_time', 0)
        num_objects = len(results.get('tracked_objects', {})) if 'tracked_objects' in results else 0
        
        if num_objects > 0:
            self.status_label.setText(
                f"✓ Кадр {self.last_processed_frame}: {num_objects} объект(ов), {exec_time:.2f}с"
            )
        else:
            self.status_label.setText(f"✓ Обработано за {exec_time:.2f}с")
        self.status_label.setStyleSheet("color: green; font-style: normal; font-weight: bold;")
        
        # Обновление UI
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def on_processing_error(self, error_msg: str):
        """Обработка ошибки обработки."""
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        
        # Обновление статуса
        self.status_label.setText(f"✗ Ошибка: {error_msg[:50]}...")
        self.status_label.setStyleSheet("color: red; font-style: normal; font-weight: bold;")
        
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def update_visualization(self):
        """Обновление визуализации результатов."""
        if self.current_results is None or self.video_controller is None:
            return
        
        # Получение текущего кадра
        frame = self.video_controller.get_current_frame()
        if frame is None:
            return
        
        # Получение параметров визуализации
        mode, vis_params = self.visualization_controls.get_visualization_params()
        scale = vis_params['scale']
        
        # Визуализация в зависимости от алгоритма
        if self.current_algorithm == 'horn_schunck':
            u = self.current_results['u']
            v = self.current_results['v']
            magnitude = self.current_results['magnitude']
            
            if mode == "HSV плотный поток":
                vis_image = self.visualization_engine.visualize_dense_flow_hsv(u, v, magnitude_scale=scale * 10)
            elif mode == "Стрелки на сетке":
                vis_image = self.visualization_engine.visualize_flow_grid(frame, u, v, step=20, scale=scale)
            elif mode == "Heat map":
                vis_image = self.visualization_engine.visualize_heatmap(magnitude)
            else:
                vis_image = frame
                
        elif self.current_algorithm == 'lucas_kanade':
            points = self.current_results['points']
            vectors = self.current_results['vectors']
            magnitudes = self.current_results['magnitudes']
            
            if len(points) > 0:
                vis_image = self.visualization_engine.visualize_sparse_flow(
                    frame, points, vectors, magnitudes, scale=scale
                )
            else:
                vis_image = frame
        
        elif self.current_algorithm == 'farneback':
            # Farneback дает плотный поток, как Horn-Schunck
            u = self.current_results['u']
            v = self.current_results['v']
            magnitude = self.current_results['magnitude']
            
            if mode == "HSV плотный поток":
                vis_image = self.visualization_engine.visualize_dense_flow_hsv(u, v, magnitude_scale=scale * 10)
            elif mode == "Стрелки на сетке":
                vis_image = self.visualization_engine.visualize_flow_grid(frame, u, v, step=20, scale=scale)
            elif mode == "Heat map":
                vis_image = self.visualization_engine.visualize_heatmap(magnitude)
            else:
                vis_image = frame
        
        else:
            vis_image = frame
        
        # Наложение результатов трекинга (если есть)
        if 'tracked_objects' in self.current_results:
            tracked_objects = self.current_results['tracked_objects']
            paths = self.current_results.get('paths')
            u = self.current_results.get('u')
            v = self.current_results.get('v')
            
            vis_image = self.visualization_engine.visualize_tracked_objects(
                vis_image, tracked_objects, paths, None, u, v
            )
        
        # Отображение результата
        self.display_frame(vis_image, self.result_label)

    def display_frame(self, frame: np.ndarray, label: QLabel):
        """
        Отображение кадра в QLabel.
        
        Args:
            frame: Кадр изображения (BGR)
            label: QLabel для отображения
        """
        if frame is None:
            return
        
        # Преобразование BGR в RGB
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Масштабирование для отображения
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
    
    def export_report(self):
        """Экспорт отчёта."""
        if self.current_results is None:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов для экспорта")
            return
        
        # Экспорт JSON
        json_path = self.report_generator.export_json()
        
        # Сохранение визуализации если есть
        if self.video_controller is not None:
            frame = self.video_controller.get_current_frame()
            if frame is not None and self.current_results is not None:
                # Обновляем визуализацию для экспорта
                self.update_visualization()
                
                # Получаем изображение из label (упрощённая версия)
                # В реальном приложении лучше сохранять напрямую
                pass
        
        QMessageBox.information(self, "Успех", f"Отчёт сохранён: {json_path}")
    
    def closeEvent(self, event):
        """Обработка закрытия окна."""
        if self.processing_thread is not None and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
        
        if self.video_controller is not None:
            self.video_controller.release()
        
        event.accept()
