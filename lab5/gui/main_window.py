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
                             QSplitter, QScrollArea, QProgressBar, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import time
from typing import Optional, Tuple

from core.video_processor import VideoController
from core.horn_schunck import HornSchunckProcessor
from core.lucas_kanade import LucasKanadeProcessor
from utils.visualization import VisualizationEngine
from utils.report_generator import ReportGenerator
from gui.controls import (AlgorithmParametersWidget, VideoControlsWidget,
                         VisualizationControlsWidget)


class ProcessingThread(QThread):
    """Поток для асинхронной обработки оптического потока."""
    
    finished = pyqtSignal(object, str)  # results, algorithm_name
    progress = pyqtSignal(int)  # progress percentage
    error = pyqtSignal(str)  # error message
    
    def __init__(self, frame1: np.ndarray, frame2: np.ndarray,
                 algorithm: str, algorithm_params: dict):
        super().__init__()
        self.frame1 = frame1.copy()
        self.frame2 = frame2.copy()
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
    
    def run(self):
        """Выполнение обработки в отдельном потоке."""
        try:
            start_time = time.time()
            
            if self.algorithm == 'horn_schunck':
                processor = HornSchunckProcessor(
                    lambda_val=self.algorithm_params.get('lambda', 1.0),
                    num_iterations=self.algorithm_params.get('iterations', 100),
                    threshold=self.algorithm_params.get('threshold', 0.001)
                )
                u, v, magnitude, angle = processor.compute_flow_magnitude_direction(
                    self.frame1, self.frame2
                )
                results = {
                    'u': u,
                    'v': v,
                    'magnitude': magnitude,
                    'angle': angle
                }
                
            elif self.algorithm == 'lucas_kanade':
                processor = LucasKanadeProcessor(
                    window_size=self.algorithm_params.get('window_size', 15),
                    max_level=self.algorithm_params.get('max_level', 2),
                    max_corners=self.algorithm_params.get('max_corners', 500)
                )
                points, vectors, magnitudes = processor.compute_flow_vectors(
                    self.frame1, self.frame2
                )
                results = {
                    'points': points,
                    'vectors': vectors,
                    'magnitudes': magnitudes
                }
            else:
                raise ValueError(f"Неизвестный алгоритм: {self.algorithm}")
            
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
        
        # Левая панель: параметры и управление
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
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
        
        # Параметры алгоритмов
        self.algorithm_params = AlgorithmParametersWidget()
        left_layout.addWidget(self.algorithm_params)
        
        # Управление визуализацией
        self.visualization_controls = VisualizationControlsWidget()
        self.visualization_controls.visualizationChanged.connect(self.on_visualization_changed)
        left_layout.addWidget(self.visualization_controls)
        
        # Кнопка обработки
        self.process_btn = QPushButton("Обработать кадр")
        self.process_btn.clicked.connect(self.process_current_frame)
        left_layout.addWidget(self.process_btn)
        
        # Выбор алгоритма
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["Хорн-Шанк", "Лукас-Канаде"])
        left_layout.addWidget(QLabel("Алгоритм:"))
        left_layout.addWidget(self.algorithm_combo)
        
        # Кнопка экспорта
        self.export_btn = QPushButton("Экспорт отчёта")
        self.export_btn.clicked.connect(self.export_report)
        left_layout.addWidget(self.export_btn)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        
        main_layout.addWidget(left_panel)
        
        # Правая панель: отображение
        right_panel = QSplitter(Qt.Horizontal)
        
        # Исходное видео
        self.original_label = QLabel("Исходное видео")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("border: 1px solid gray")
        scroll_original = QScrollArea()
        scroll_original.setWidget(self.original_label)
        scroll_original.setWidgetResizable(True)
        
        # Результаты
        self.result_label = QLabel("Результаты анализа")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(400, 300)
        self.result_label.setStyleSheet("border: 1px solid gray")
        scroll_result = QScrollArea()
        scroll_result.setWidget(self.result_label)
        scroll_result.setWidgetResizable(True)
        
        right_panel.addWidget(scroll_original)
        right_panel.addWidget(scroll_result)
        
        main_layout.addWidget(right_panel, stretch=1)
    
    def load_video(self):
        """Загрузка видеофайла."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Выберите видеофайл", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if filename:
            self.video_controller = VideoController()
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
            frame = self.video_controller.get_frame(frame_idx)
            if frame is not None:
                self.display_frame(frame, self.original_label)
    
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
        else:
            algorithm_name = "lucas_kanade"
            params = self.algorithm_params.get_lucas_kanade_params()
        
        # Добавление параметров в отчёт
        self.report_generator.add_algorithm_parameters(algorithm_name, params)
        
        # Запуск обработки в отдельном потоке
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Неопределённый прогресс
        
        self.processing_thread = ProcessingThread(frame1, frame2, algorithm_name, params)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_processing_finished(self, results: dict, algorithm_name: str):
        """Обработка завершения обработки."""
        self.current_results = results
        self.current_algorithm = algorithm_name
        
        # Добавление метрик в отчёт
        if 'execution_time' in results:
            self.report_generator.add_metrics(algorithm_name, {
                'execution_time': results['execution_time']
            })
        
        # Визуализация результатов
        self.update_visualization()
        
        # Обновление UI
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def on_processing_error(self, error_msg: str):
        """Обработка ошибки обработки."""
        QMessageBox.critical(self, "Ошибка обработки", error_msg)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def on_visualization_changed(self, mode: str, params: dict):
        """Обработка изменения параметров визуализации."""
        if self.current_results is not None:
            self.update_visualization()
    
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
        alpha = vis_params['alpha']
        
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
        else:
            vis_image = frame
        
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
