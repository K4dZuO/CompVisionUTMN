# -*- coding: utf-8 -*-
import sys
import numpy as np
from PIL import Image, ImageQt
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, 
                              QVBoxLayout, QWidget, QCheckBox, QButtonGroup, 
                              QRadioButton, QFrame, QSlider, QSpinBox, QDoubleSpinBox)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
# scipy.ndimage удален - используем только собственные реализации

from ui_main import Ui_Imchanger 

# Helpers
from helpers_0 import (
    logarithmic_transform, power_transform, binary_transform, brightness_range_cutout,
    rectangular_filter, median_filter, gaussian_filter, sigma_filter,
    absolute_difference_map, unsharp_masking, add_gaussian_noise, add_salt_pepper_noise
)


class ImchangerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Imchanger()
        self.ui.setupUi(self)

        self.frames_setup()
        self.setup_connections()
        
        self.original_image = None
        self.original_array = None
        self.chroma_array = None
        self.smooth_array = None
        self.noised_array = None
        self.clarity_array = None

        # Параметры по умолчанию
        self.gamma_value = 1.0
        self.binary_threshold = 128
        self.brightness_min = 0
        self.brightness_max = 255
        self.constant_value = 0
        self.sigma_value = 1.0
        self.k_value = 3
        self.lambda_value = 1.0

    def frames_setup(self):
        # Главная страница
        self.origin_img_label = QLabel(self.ui.origin_img_frame)
        self.origin_img_label.resize(self.ui.origin_img_frame.size()) # размер фрейма
        self.origin_img_label.setScaledContents(True) # растягивание из-ия под под размер фрейма
        self.origin_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # 

        self.chroma_img_label = QLabel(self.ui.chroma_img_frame)
        self.chroma_img_label.resize(self.ui.chroma_img_frame.size())
        self.chroma_img_label.setScaledContents(True)
        self.chroma_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.smooth_img_label = QLabel(self.ui.smooth_img_frame)
        self.smooth_img_label.resize(self.ui.smooth_img_frame.size())
        self.smooth_img_label.setScaledContents(True)
        self.smooth_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.clarity_img_label = QLabel(self.ui.clarity_img_frame)
        self.clarity_img_label.resize(self.ui.clarity_img_frame.size())
        self.clarity_img_label.setScaledContents(True)
        self.clarity_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Страница цветности
        self.chroma_label = QLabel(self.ui.chroma_frame)
        self.chroma_label.resize(self.ui.chroma_frame.size())
        self.chroma_label.setScaledContents(True)
        self.chroma_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Страница сглаживания
        self.smooth_label = QLabel(self.ui.smooth_frame)
        self.smooth_label.resize(self.ui.smooth_frame.size())
        self.smooth_label.setScaledContents(True)
        self.smooth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Страница резкости
        self.clarity_label = QLabel(self.ui.clarity_frame)
        self.clarity_label.resize(self.ui.clarity_frame.size())
        self.clarity_label.setScaledContents(True)
        self.clarity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setup_connections(self):
        """Настройка соединений для всех кнопок и слайдеров"""
        self.ui.load_button.clicked.connect(self.load_image)
        self.ui.reset_button.clicked.connect(self.reset_all_changes)
        
        # Цветность
        self.ui.log_button.clicked.connect(self.apply_logarithmic)
        self.ui.power_button.clicked.connect(self.apply_power_transform)
        self.ui.binary_button.clicked.connect(self.apply_binary_transform)
        self.ui.range_button.clicked.connect(self.apply_brightness_range)
        
        # Слайдеры и спинбоксы для цветности
        self.ui.gamma_slider.valueChanged.connect(self.gamma_slider_changed)
        self.ui.gamma_spin.valueChanged.connect(self.gamma_spin_changed)
        self.ui.threshold_slider.valueChanged.connect(self.threshold_slider_changed)
        self.ui.threshold_spin.valueChanged.connect(self.threshold_spin_changed)
        self.ui.min_brightness_slider.valueChanged.connect(self.min_brightness_changed)
        self.ui.min_brightness_spin.valueChanged.connect(self.min_brightness_changed)
        self.ui.max_brightness_slider.valueChanged.connect(self.max_brightness_changed)
        self.ui.max_brightness_spin.valueChanged.connect(self.max_brightness_changed)
        self.ui.constant_value_slider.valueChanged.connect(self.constant_value_changed)
        self.ui.constant_value_spin.valueChanged.connect(self.constant_value_changed)
        
        # Сглаживание
        self.ui.add_noise_button.clicked.connect(self.add_noise)
        self.ui.rectangular_button.clicked.connect(self.apply_rectangular_filter)
        self.ui.median_button.clicked.connect(self.apply_median_filter)
        self.ui.gaussian_button.clicked.connect(self.apply_gaussian_filter)
        self.ui.sigma_button.clicked.connect(self.apply_sigma_filter)
        self.ui.difference_button.clicked.connect(self.show_difference_map)
        
        # Слайдеры для сглаживания
        self.ui.sigma_slider.valueChanged.connect(self.sigma_slider_changed)
        self.ui.sigma_spin.valueChanged.connect(self.sigma_spin_changed)
        
        # Резкость
        self.ui.unsharp_button.clicked.connect(self.apply_unsharp_masking)
        
        # Слайдеры для резкости
        self.ui.k_slider.valueChanged.connect(self.k_slider_changed)
        self.ui.k_spin.valueChanged.connect(self.k_spin_changed)
        self.ui.lambda_slider.valueChanged.connect(self.lambda_slider_changed)
        self.ui.lambda_spin.valueChanged.connect(self.lambda_spin_changed)

    def load_image(self):
        """Загружает изображение и отображает его."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                self.original_image = Image.open(file_path).convert("L")  # Конвертируем в градации серого
                self.original_array = np.array(self.original_image)
                self.chroma_array = self.original_array.copy()
                self.smooth_array = self.original_array.copy()
                self.clarity_array = self.original_array.copy()
                
                self.display_original_image_in_frame()
                self.update_chroma_display()
                self.update_smooth_display()
                self.update_clarity_display()

            except Exception as e:
                print(f"Ошибка загрузки изображения: {e}")

    def reset_all_changes(self):
        """Сбрасывает все изменения и возвращает исходное изображение"""
        if self.original_array is not None:
            # Сбрасываем все массивы к оригиналу
            self.chroma_array = self.original_array.copy()
            self.smooth_array = self.original_array.copy()
            self.clarity_array = self.original_array.copy()
            self.noised_array = None
            
            # Сбрасываем параметры к значениям по умолчанию
            self.reset_parameters_to_default()
            
            # Обновляем все отображения
            self.update_all_displays()
            
            print("Все изменения сброшены!")

    def reset_parameters_to_default(self):
        """Сбрасывает все параметры к значениям по умолчанию"""
        # Сбрасываем значения переменных
        self.gamma_value = 1.0
        self.binary_threshold = 128
        self.brightness_min = 0
        self.brightness_max = 255
        self.constant_value = 0
        self.sigma_value = 1.0
        self.k_value = 3
        self.lambda_value = 1.0
        
        # Сбрасываем UI элементы
        self.ui.gamma_slider.setValue(10)  # 10 / 10 = 1.0
        self.ui.gamma_spin.setValue(1.0)
        
        self.ui.threshold_slider.setValue(128)
        self.ui.threshold_spin.setValue(128)
        
        self.ui.min_brightness_slider.setValue(0)
        self.ui.min_brightness_spin.setValue(0)
        
        self.ui.max_brightness_slider.setValue(255)
        self.ui.max_brightness_spin.setValue(255)
        
        self.ui.constant_value_slider.setValue(0)
        self.ui.constant_value_spin.setValue(0)
        
        self.ui.sigma_slider.setValue(10)  # 10 / 10 = 1.0
        self.ui.sigma_spin.setValue(1.0)
        
        self.ui.k_slider.setValue(3)
        self.ui.k_spin.setValue(3)
        
        self.ui.lambda_slider.setValue(10)  # 10 / 10 = 1.0
        self.ui.lambda_spin.setValue(1.0)
        
        # Сбрасываем радио-кнопки
        self.ui.constant_radio.setChecked(True)
        self.ui.kernel3_radio.setChecked(True)

    def update_all_displays(self):
        """Обновляет все отображения изображений"""
        self.update_chroma_display()
        self.update_smooth_display()
        self.update_clarity_display()
        self.display_original_image_in_frame()

    def display_original_image_in_frame(self):
        if self.original_image is None:
            return
        pixmap = self.original_image.toqpixmap()
        self.origin_img_label.setPixmap(pixmap.scaled(self.origin_img_label.size()))

    def array_to_pixmap(self, array):
        """Конвертирует numpy array в QPixmap"""
        image = Image.fromarray(array)
        return ImageQt.toqpixmap(image)

    # ========== ЦВЕТНОСТЬ ==========
    
    def update_chroma_display(self):
        """Обновляет отображение на вкладке цветности"""
        if self.chroma_array is not None:
            pixmap = self.array_to_pixmap(self.chroma_array)
            self.chroma_label.setPixmap(pixmap.scaled(self.chroma_label.size()))
            # Также обновляем миниатюру на главной странице
            self.chroma_img_label.setPixmap(pixmap.scaled(self.chroma_img_label.size()))

    def apply_logarithmic(self):
        if self.original_array is not None:
            self.chroma_array = logarithmic_transform(self.original_array)
            self.update_chroma_display()

    def apply_power_transform(self):
        if self.original_array is not None:
            self.chroma_array = power_transform(self.original_array, self.gamma_value)
            self.update_chroma_display()

    def apply_binary_transform(self):
        if self.original_array is not None:
            self.chroma_array = binary_transform(self.original_array, self.binary_threshold)
            self.update_chroma_display()

    def apply_brightness_range(self):
        if self.original_array is not None:
            use_constant = self.ui.constant_radio.isChecked()
            constant_val = self.constant_value if use_constant else None
            self.chroma_array = brightness_range_cutout(image = self.original_array, 
                                                       min_val=self.brightness_min, 
                                                       max_val=self.brightness_max, 
                                                       constant_value=constant_val)
            self.update_chroma_display()

    # Обработчики слайдеров для цветности
    def gamma_slider_changed(self, value):
        self.gamma_value = value / 10.0
        self.ui.gamma_spin.setValue(self.gamma_value)

    def gamma_spin_changed(self, value):
        self.gamma_value = value
        self.ui.gamma_slider.setValue(int(value * 10))

    def threshold_slider_changed(self, value):
        self.binary_threshold = value
        self.ui.threshold_spin.setValue(value)

    def threshold_spin_changed(self, value):
        self.binary_threshold = value
        self.ui.threshold_slider.setValue(value)

    def min_brightness_changed(self, value):
        self.brightness_min = value
        self.ui.min_brightness_spin.setValue(value)

    def max_brightness_changed(self, value):
        self.brightness_max = value
        self.ui.max_brightness_spin.setValue(value)

    def constant_value_changed(self, value):
        self.constant_value = value
        self.ui.constant_value_spin.setValue(value)
        

    # ========== СГЛАЖИВАНИЕ ==========
    
    def update_smooth_display(self):
        """Обновляет отображение на вкладке сглаживания"""
        if self.smooth_array is not None:
            pixmap = self.array_to_pixmap(self.smooth_array)
            self.smooth_label.setPixmap(pixmap.scaled(self.smooth_label.size()))
            # Обновляем миниатюру на главной странице
            self.smooth_img_label.setPixmap(pixmap.scaled(self.smooth_img_label.size()))

    def add_noise(self):
        if self.original_array is not None:
            # Добавляем смешанный шум для демонстрации
            noisy = add_gaussian_noise(self.original_array, sigma=20)
            self.noised_array = add_salt_pepper_noise(noisy, 0.02, 0.02)
            self.smooth_array = self.noised_array.copy()
            self.update_smooth_display()

    def apply_rectangular_filter(self):
        if self.noised_array is not None:
            kernel_size = 3 if self.ui.kernel3_radio.isChecked() else 5
            self.smooth_array = rectangular_filter(self.noised_array, kernel_size)
            self.update_smooth_display()

    def apply_median_filter(self):
        if self.noised_array is not None:
            kernel_size = 3 if self.ui.kernel3_radio.isChecked() else 5
            self.smooth_array = median_filter(self.noised_array, kernel_size)
            self.update_smooth_display()

    def apply_gaussian_filter(self):
        if self.noised_array is not None:
            self.smooth_array = gaussian_filter(self.noised_array, self.sigma_value)
            self.update_smooth_display()

    def apply_sigma_filter(self):
        if self.noised_array is not None:
            window_size = 3 if self.ui.kernel3_radio.isChecked() else 5
            self.smooth_array = sigma_filter(self.noised_array, self.sigma_value, window_size)
            self.update_smooth_display()

    def show_difference_map(self):
        if self.noised_array is not None and self.smooth_array is not None:
            diff_map = absolute_difference_map(self.noised_array, self.smooth_array)
            pixmap = self.array_to_pixmap(diff_map)
            self.smooth_label.setPixmap(pixmap.scaled(self.smooth_label.size()))

    def sigma_slider_changed(self, value):
        self.sigma_value = value / 10.0
        self.ui.sigma_spin.setValue(self.sigma_value)
        # Убрали автоматическое применение - только при нажатии кнопки

    def sigma_spin_changed(self, value):
        self.sigma_value = value
        self.ui.sigma_slider.setValue(int(value * 10))
        # Убрали автоматическое применение - только при нажатии кнопки

    # ========== РЕЗКОСТЬ ==========
    
    def update_clarity_display(self):
        """Обновляет отображение на вкладке резкости"""
        if self.clarity_array is not None:
            pixmap = self.array_to_pixmap(self.clarity_array)
            self.clarity_label.setPixmap(pixmap.scaled(self.clarity_label.size()))
            # Обновляем миниатюру на главной странице
            self.clarity_img_label.setPixmap(pixmap.scaled(self.clarity_img_label.size()))

    def apply_unsharp_masking(self):
        if self.original_array is not None:
            self.clarity_array = unsharp_masking(self.original_array, self.k_value, self.lambda_value)
            self.update_clarity_display()

    def k_slider_changed(self, value):
        self.k_value = value
        self.ui.k_spin.setValue(value)
        # Убрали автоматическое применение - только при нажатии кнопки

    def k_spin_changed(self, value):
        self.k_value = value
        self.ui.k_slider.setValue(value)
        # Убрали автоматическое применение - только при нажатии кнопки

    def lambda_slider_changed(self, value):
        self.lambda_value = value / 10.0
        self.ui.lambda_spin.setValue(self.lambda_value)
        # Убрали автоматическое применение - только при нажатии кнопки

    def lambda_spin_changed(self, value):
        self.lambda_value = value
        self.ui.lambda_slider.setValue(int(value * 10))
        # Убрали автоматическое применение - только при нажатии кнопки


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImchangerApp()
    window.show()
    sys.exit(app.exec())
