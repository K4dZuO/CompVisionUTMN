import os
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
from PySide6.QtCore import Qt
from PIL import Image
from PIL.ImageQt import ImageQt
import numpy as np

from ui_main import Ui_Imchanger
from helpers import (get_histogram, draw_histogram_image,
                     gaussian_filter, avarage_filter, get_hf_simple,
                     apply_convolution_filter, parse_kernel_from_string, get_standard_kernels,
                     harris_corner_detection, shi_tomasi_corner_detection,
                     sobel_edge_detection, canny_edge_detection)


class MainWindow(QMainWindow, Ui_Imchanger):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("ImChanger")

        self.frames_setup()
        self.load_button.clicked.connect(self.load_file)
        self.save_button.clicked.connect(self.save_file)


        self.reset_button.clicked.connect(self.reset_image)
        self.negative_button.clicked.connect(self.apply_negative)
        self.rotate_90_button.clicked.connect(self.apply_90_rotation)
        self.reflect_by_v_button.clicked.connect(self.apply_flip_vertical)
        self.rgb_to_grb_button.clicked.connect(self.apply_from_rgb_to_grb)
        self.grb_to_rgb_button.clicked.connect(self.apply_from_grb_to_rgb)
        
        self.lf_simple_button.clicked.connect(self.set_lf_image_simple)
        self.hf_simple_button.clicked.connect(self.set_hf_image_simple)

        # self.brightness_slider.valueChanged.connect(self.apply_transformations)
        # self.contrast_slider.valueChanged.connect(self.apply_transformations)
        self.brightness_slider.valueChanged.connect(self.update_brightness_label)
        self.contrast_slider.valueChanged.connect(self.update_contrast_label)
        self.c_slider.valueChanged.connect(self.update_c_label)

        self.brightness_slider.sliderReleased.connect(self.apply_transformations)
        self.contrast_slider.sliderReleased.connect(self.apply_transformations)

        # self.pseudo_pillow_button.clicked.connect(self.apply_pseudo_pillow)
        # self.pseudo_manual_button.clicked.connect(self.apply_pseudo_manual)

        self.current_pil = None
        self.current_np = None
        self.brightness_value = 0
        self.contrast_value = 10.0
        
        self.original_pil = None
        self.original_np = None

        self.cursor_size = 13
        self.cursor_scale = 11

        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.on_image_mouse_move
        
        # Инициализируем дополнительные элементы управления
        self.setup_additional_controls()

    @staticmethod
    def pil_to_qpixmap(pil: Image.Image) -> QPixmap:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        qim = ImageQt(pil)  # PIL -> QImage
        return QPixmap.fromImage(qim)

    def frames_setup(self):
        self.image_label = QLabel(self.img_frame)
        self.image_label.resize(self.img_frame.size())
        self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.gray_label = QLabel(self.gray_frame)
        self.gray_label.resize(self.gray_frame.size())
        self.gray_label.setScaledContents(True)

        self.gray_hist_label = QLabel(self.gray_hist)
        self.gray_hist_label.resize(self.gray_hist.size())
        self.gray_hist_label.setScaledContents(True)

        self.red_label = QLabel(self.red_frame)
        self.red_label.resize(self.red_frame.size())
        self.red_label.setScaledContents(True)

        self.red_hist_label = QLabel(self.red_hist)
        self.red_hist_label.resize(self.red_hist.size())
        self.red_hist_label.setScaledContents(True)

        self.blue_label = QLabel(self.blue_frame)
        self.blue_label.resize(self.blue_frame.size())
        self.blue_label.setScaledContents(True)

        self.blue_hist_label = QLabel(self.blue_hist)
        self.blue_hist_label.resize(self.blue_hist.size())
        self.blue_hist_label.setScaledContents(True)

        self.green_label = QLabel(self.green_frame)
        self.green_label.resize(self.green_frame.size())
        self.green_label.setScaledContents(True)

        self.green_hist_label = QLabel(self.green_hist)
        self.green_hist_label.resize(self.green_hist.size())
        self.green_hist_label.setScaledContents(True)

        self.cursor_label = QLabel(self.cursor_frame)
        self.cursor_label.resize(self.cursor_frame.size())
        self.cursor_label.setScaledContents(False)
        self.cursor_label.setAlignment(Qt.AlignCenter)
        self.cursor_label.setStyleSheet("border: 1px solid gray; background: black;")
        self.cursor_label.setText("Наведите курсор\nна изображение")
    
    def setup_additional_controls(self):
        """Настраивает связи событий для дополнительных элементов управления"""
        # Вызываем метод из UI класса для инициализации элементов
        super().setup_additional_controls()
        
        # Настраиваем связи событий
        self.kernel_combo.currentTextChanged.connect(self.on_kernel_combo_changed)
        self.apply_convolution_button.clicked.connect(self.apply_convolution_filter)
        self.harris_button.clicked.connect(self.apply_harris_corners)
        self.shi_tomasi_button.clicked.connect(self.apply_shi_tomasi_corners)
        self.sobel_button.clicked.connect(self.apply_sobel_edges)
        self.canny_button.clicked.connect(self.apply_canny_edges)

    def display_original_image_in_frame(self):
        if self.original_pil is None:
            return
        pixmap = self.pil_to_qpixmap(self.original_pil)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
                                                 Qt.AspectRatioMode.KeepAspectRatio, # Масштабирование с сохранением пропорций
                                                 Qt.TransformationMode.SmoothTransformation)) # сглаживание по сути

    # загрузка / сохранение
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Изображения (*.png *.bmp *.tiff *.tif);;Все файлы (*)")
        if not file_path:
            return
        try:
            self.original_pil = Image.open(file_path).convert("RGB")
            self.original_np = np.array(self.original_pil)

            # сразу создаём current_np / current_pil
            self.current_np = self.original_np.copy()
            self.current_pil = self.original_pil.copy()
            self.brightness_value = 0
            self.contrast_value = 10.0

            self.filename_label.setText(os.path.basename(file_path))
            self.display_original_image_in_frame()
            self.update_previews()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")

    def save_file(self):
        if self.current_pil is None:
            QMessageBox.warning(self, "Attention", "No image for save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            "",
            "PNG (*.png);;BMP (*.bmp);;TIFF (*.tiff *.tif);;JPEG (*.jpg *.jpeg);;Все файлы (*)"
        )
        if not file_path:
            return

        try:
            self.current_pil.save(file_path)
            QMessageBox.information(self, "OK", f"Image was saved:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "ERROR", f"Can't save an image:\n{str(e)}")


    def on_image_mouse_move(self, event):
        if self.original_np is None:
            return
        if hasattr(event, "position"):
            px = event.position().x()
            py = event.position().y()
        else:
            px = event.x(); py = event.y()
        img_w = self.original_np.shape[1]; img_h = self.original_np.shape[0]
        lbl_w = max(1, self.image_label.width()); lbl_h = max(1, self.image_label.height())
        # image_label сейчас масштабирует картинку в рамки QLabel (scaledContents True) либо KeepAspectRatio used in display
        # Простейшее соответствие — нормировать по размерам QLabel:
        cx = int(px * img_w / lbl_w)
        cy = int(py * img_h / lbl_h)
        cx = max(0, min(img_w - 1, cx))
        cy = max(0, min(img_h - 1, cy))
        r, g, b = self.original_np[cy, cx]
        intensity = (r + g + b) / 3
        preview_stats = f"Coords: ({cx}, {cy}), \n RGB=({r},{g},{b}), I={intensity:.1f}"
        #self.coords_label.setText(f"Coords: ({int(px)}, {int(py)})")
        self.update_cursor_window(cx, cy, preview_stats)

    def update_cursor_window(self, cx, cy, preview_stats):
        if self.original_np is None:
            return
        img = self.current_np if self.current_np is not None else self.original_np

        # размер окна лупы
        half = self.cursor_size // 2
        h, w = img.shape[:2]
        x1, x2 = max(0, cx - half), min(w, cx + half + 1)
        y1, y2 = max(0, cy - half), min(h, cy + half + 1)
        window = img[y1:y2, x1:x2]

        if window.size == 0:
            return
        if window.shape[0] != self.cursor_size or window.shape[1] != self.cursor_size:
            full = np.zeros((self.cursor_size, self.cursor_size, 3), dtype=np.uint8)
            h_part, w_part = window.shape[:2]
            y_off = (self.cursor_size - h_part) // 2
            x_off = (self.cursor_size - w_part) // 2
            full[y_off:y_off+h_part, x_off:x_off+w_part] = window
            window = full

        # создаём QPixmap для лупы
        pil_window = Image.fromarray(window).resize(
            (self.cursor_size * self.cursor_scale, self.cursor_size * self.cursor_scale),
            Image.Resampling.NEAREST
        )
        pix = self.pil_to_qpixmap(pil_window)

        # рисуем сетку лупы
        painter = QPainter(pix)
        pen = QPen(QColor(200, 200, 200, 200))
        pen.setWidth(1)
        painter.setPen(pen)
        step = self.cursor_scale
        for i in range(0, pix.width(), step):
            painter.drawLine(i, 0, i, pix.height())
        for j in range(0, pix.height(), step):
            painter.drawLine(0, j, pix.width(), j)
        painter.end()

        self.cursor_label.setPixmap(pix)
        self.cursor_label.setFixedSize(pix.width(), pix.height())

        # --- рисуем рамку на основном изображении ---
        main_pixmap = self.pil_to_qpixmap(Image.fromarray(img)).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        painter = QPainter(main_pixmap)
        pen = QPen(QColor(255, 0, 0, 180))
        pen.setWidth(2)
        painter.setPen(pen)

        # пересчёт координат с учётом масштабирования QLabel
        lbl_w, lbl_h = self.image_label.width(), self.image_label.height()
        img_h, img_w = img.shape[:2]
        scale = min(lbl_w / img_w, lbl_h / img_h)
        offset_x = (lbl_w - img_w * scale) / 2
        offset_y = (lbl_h - img_h * scale) / 2

        rect_x1 = int(x1 * scale + offset_x)
        rect_y1 = int(y1 * scale + offset_y)
        rect_x2 = int(x2 * scale + offset_x)
        rect_y2 = int(y2 * scale + offset_y)

        painter.drawRect(rect_x1, rect_y1, rect_x2 - rect_x1, rect_y2 - rect_y1)
        painter.end()

        self.image_label.setPixmap(main_pixmap)
        # После отрисовки превью добавляем статистику
        stats_text = self.update_cursor_stats(window)
        self.coords_label.setText(preview_stats + '\n' + stats_text)


    def reset_image(self):
        """Сбрасывает все изменения"""
        if self.original_pil is None:
            return
        self.current_pil = self.original_pil.copy()
        self.current_np = self.original_np.copy()
        self.brightness_value = 0
        self.contrast_value = 10.0
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(10)  # 10 = 1.0
        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
        
    def apply_negative(self):
        """Инверсия цветов"""
        if self.current_np is None:
            return
        self.current_np = 255 - self.current_np
        self.current_pil = Image.fromarray(self.current_np)
        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
    
    def apply_90_rotation(self):
        """Поворот изображения на 90° против часовой стрелки"""
        if self.current_np is None:
            return
        self.current_np = np.rot90(self.current_np, k=1) # меняет (i; j) на (n-i; j)
        self.current_pil = Image.fromarray(self.current_np)

        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
        
    def apply_flip_vertical(self):
        """Вертикальное отражение """
        if self.current_np is None:
            return
        self.current_np = np.flipud(self.current_np) # меняет (i; j) и (n-i; j) строки
        self.current_pil = Image.fromarray(self.current_np)

        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
        
    def apply_from_rgb_to_grb(self):
        if self.current_np is None:
            return
        img = self.current_np.copy()
        only_red = img[:,:,0]
        only_green = img[:,:,1]
        self.current_np[:,:,1] = only_red
        self.current_np[:,:,0] = only_green
        
        self.current_pil = Image.fromarray(self.current_np)
        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
        
        
        
    def apply_from_grb_to_rgb(self):
        if self.current_np is None:
            return
        img = self.current_np.copy()
        only_red = img[:,:,1]
        only_green = img[:,:,0]
        self.current_np[:,:,0] = only_red
        self.current_np[:,:,1] = only_green
        
        self.current_pil = Image.fromarray(self.current_np)
        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()


    def apply_transformations(self):
        """Применяет яркость/контраст поверх оригинала"""
        if self.original_np is None:
            return

        self.brightness_value = self.brightness_slider.value()
        self.contrast_value = self.contrast_slider.value() / 10.0  # 0.1 .. 3.0

        img = self.original_np.astype(np.float32)

        # контраст
        img = 255 / (1 + np.exp(self.contrast_value * (0.5 - img / 255.0)))
        
        # яркость
        img += self.brightness_value

        img = np.clip(img, 0, 255).astype(np.uint8)
        self.current_np = img
        self.current_pil = Image.fromarray(img)

        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()

    def update_stats(self):
        """Обновляет статистику для текущего изображения"""
        if self.current_np is None:
            self.stats_label.setText("Нет данных")
            return

        img = self.current_np
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        min_v = img.min(axis=(0, 1))
        max_v = img.max(axis=(0, 1))

        text = (f"Статистика (RGB):\n"
                f"Avg: {mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}\n"
                f"Std: {std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}\n"
                f"Min: {min_v[0]}, {min_v[1]}, {min_v[2]}\n"
                f"Max: {max_v[0]}, {max_v[1]}, {max_v[2]}")
        self.stats_label.setText(text)

    def update_brightness_label(self, value):
        self.brightness_label.setText(f"Brightness: {value}")

    def update_contrast_label(self, value):
        self.contrast_label.setText(f"Contrast: {value/10:.1f}")
    
    def update_c_label(self):
        self.c_label.setText(f"{self.c_slider.value()/10}")
            

    def update_previews(self):
        if self.current_np is None:
            return

        # основное изображение
        self.image_label.setPixmap(
            self.pil_to_qpixmap(Image.fromarray(self.current_np)).scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        
        # превью W/B
        gray = np.dot(self.current_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        gray_pil = Image.fromarray(gray)
        self.gray_label.setPixmap(
            self.pil_to_qpixmap(gray_pil).scaled(
                self.gray_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        gray_hist = get_histogram(gray)
        gray_hist_image = draw_histogram_image(gray_hist, color='gray')
        self.gray_hist_label.setPixmap(self.pil_to_qpixmap(gray_hist_image).scaled(self.gray_hist_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


        # отдельные каналы
        red_only = np.zeros_like(self.current_np); red_only[:, :, 0] = self.current_np[:, :, 0]
        green_only = np.zeros_like(self.current_np); green_only[:, :, 1] = self.current_np[:, :, 1]
        blue_only = np.zeros_like(self.current_np); blue_only[:, :, 2] = self.current_np[:, :, 2]

        self.red_label.setPixmap(
            self.pil_to_qpixmap(Image.fromarray(red_only)).scaled(
                self.red_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        red_hist = get_histogram(red_only[..., 0])
        red_hist_image = draw_histogram_image(red_hist, color='red')
        self.red_hist_label.setPixmap(self.pil_to_qpixmap(red_hist_image).scaled(self.red_hist_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        self.green_label.setPixmap(
            self.pil_to_qpixmap(Image.fromarray(green_only)).scaled(
                self.green_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        green_hist = get_histogram(self.current_np[..., 1])
        green_hist_image = draw_histogram_image(green_hist, color='green')
        self.green_hist_label.setPixmap(self.pil_to_qpixmap(green_hist_image).scaled(self.green_hist_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        
        self.blue_label.setPixmap(
            self.pil_to_qpixmap(Image.fromarray(blue_only)).scaled(
                self.blue_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        blue_hist = get_histogram(self.current_np[..., 2])
        blue_hist_image = draw_histogram_image(blue_hist, color='blue')
        self.blue_hist_label.setPixmap(self.pil_to_qpixmap(blue_hist_image).scaled(self.blue_hist_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


    def update_cursor_stats(self, window):
        """Считает статистику для превью 13×13"""
        if window is None or window.size == 0:
            return "Нет данных"
        
        # Для RGB каналов
        stats_text = ""
        
        # По каналам R, G, B
        for i, color in enumerate(['R', 'G', 'B']):
            channel = window[:, :, i]
            stats_text += (f"{color}: avg {channel.mean():.1f} "
                          f"std: {channel.std():.1f} "
                          f"var : {channel.var():.1f}"
                          f"min {channel.min()} max {channel.max()}\n")
    
    
        return stats_text
        
    def set_modified_image(self, np_img):
        """Обновляет текущее изображение после применения фильтров"""
        if np_img is None:
            return
        self.current_np = np_img
        self.current_pil = Image.fromarray(np_img)

        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
        
    def set_lf_image_simple(self):
        if self.current_np is None:
            return
        
        if self.average_blur_rbutton.isChecked():
            smooth_func = avarage_filter
        elif self.gauss_filter_rbutton.isChecked():
            smooth_func = gaussian_filter
        else:
            return
        
        if self.average_blur_rbutton.isChecked():
            self.current_np = smooth_func(self.current_np, 5)
        elif self.gauss_filter_rbutton.isChecked():
            self.current_np = smooth_func(self.current_np, 1.0)  # sigma для гаусса
        self.current_pil = Image.fromarray(self.current_np)

        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
    
    def set_hf_image_simple(self):
        if self.current_np is None:
            return
        
        self.current_np = get_hf_simple(original_image=self.original_np,
                                        lf_image=self.current_np,
                                        c = self.c_slider.value()/10)
        self.current_pil = Image.fromarray(self.current_np)

        self.display_original_image_in_frame()
        
        self.update_stats()
        self.update_previews()
    
    def on_kernel_combo_changed(self, kernel_name):
        """Обновляет текстовое поле при выборе стандартного ядра"""
        standard_kernels = get_standard_kernels()
        if kernel_name in standard_kernels:
            kernel = standard_kernels[kernel_name]
            size = kernel.shape[0]
            self.kernel_size_edit.setText(str(size))
            
            # Преобразуем матрицу в строку
            kernel_str = ""
            for i in range(size):
                row = " ".join([str(int(kernel[i, j])) if kernel[i, j] == int(kernel[i, j]) else f"{kernel[i, j]:.1f}" for j in range(size)])
                kernel_str += row + "\n"
            self.kernel_text_edit.setPlainText(kernel_str.strip())
    
    def apply_convolution_filter(self):
        """Применяет свёртку с пользовательской матрицей"""
        if self.current_np is None:
            return
        
        try:
            # Получаем размер ядра
            size = int(self.kernel_size_edit.text())
            
            # Парсим ядро из текстового поля
            kernel_str = self.kernel_text_edit.toPlainText()
            kernel = parse_kernel_from_string(kernel_str, size)
            
            # Получаем параметры
            normalize = self.normalize_checkbox.isChecked()
            add_128 = self.add_128_checkbox.isChecked()
            
            # Применяем фильтр
            result = apply_convolution_filter(self.current_np, kernel, normalize, add_128)
            self.set_modified_image(result)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при применении свёртки:\n{str(e)}")
    
    def apply_harris_corners(self):
        """Применяет детекцию углов методом Харриса"""
        if self.current_np is None:
            return
        
        try:
            threshold = self.corner_threshold_slider.value() / 1000.0  # Преобразуем в диапазон 0.001-0.1
            result = harris_corner_detection(self.current_np, threshold=threshold)
            self.set_modified_image(result)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при детекции углов Харриса:\n{str(e)}")
    
    def apply_shi_tomasi_corners(self):
        """Применяет детекцию углов методом Shi-Tomasi"""
        if self.current_np is None:
            return
        
        try:
            quality_level = self.corner_threshold_slider.value() / 1000.0  # Преобразуем в диапазон 0.001-0.1
            result = shi_tomasi_corner_detection(self.current_np, quality_level=quality_level)
            self.set_modified_image(result)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при детекции углов Shi-Tomasi:\n{str(e)}")
    
    def apply_sobel_edges(self):
        """Применяет детекцию границ оператором Собеля"""
        if self.current_np is None:
            return
        
        try:
            add_128 = self.add_128_checkbox.isChecked()
            result = sobel_edge_detection(self.current_np, add_128=add_128)
            self.set_modified_image(result)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при детекции границ Собеля:\n{str(e)}")
    
    def apply_canny_edges(self):
        """Применяет детекцию границ алгоритмом Канни"""
        if self.current_np is None:
            return
        
        try:
            low_threshold = self.canny_low_slider.value()
            high_threshold = self.canny_high_slider.value()
            result = canny_edge_detection(self.current_np, low_threshold, high_threshold)
            self.set_modified_image(result)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при детекции границ Канни:\n{str(e)}")
        
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
