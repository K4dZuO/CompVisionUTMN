import os
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel
)
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPixmap, QImage
from PIL import Image, ImageDraw
import numpy as np

# Импортируем сгенерированный UI
from ui_main import Ui_Imchanger


class MainWindow(QMainWindow, Ui_Imchanger):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("ImChanger — Обработка изображений")

        # Подготовка отображения изображений
        self.frames_setup()

        # Кнопки
        self.load_button.clicked.connect(self.load_file)
        self.save_button.clicked.connect(self.save_image)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        self.negative_button.clicked.connect(self.create_negative)

        # Данные изображений
        self.original_pil = None
        self.original_np = None
        self.current_pil = None
        self.current_np = None

        # Наведение курсора
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.current_pixel_info = None

        # Метки для статистики
        self.global_stats_label = QLabel(self)
        self.global_stats_label.setGeometry(QRect(560, 570, 400, 60))
        self.global_stats_label.setStyleSheet("border: 1px solid gray; padding: 3px;")

    # ---------------- Глобальная статистика ----------------
    def update_global_stats(self):
        if self.current_np is None:
            self.global_stats_label.setText("")
            return
        mean = np.mean(self.current_np, axis=(0, 1))
        std = np.std(self.current_np, axis=(0, 1))
        text = (f"Глобальное среднее: R={mean[0]:.1f}, G={mean[1]:.1f}, B={mean[2]:.1f}\n"
                f"Глобальное ст.откл: R={std[0]:.1f}, G={std[1]:.1f}, B={std[2]:.1f}")
        self.global_stats_label.setText(text)

    # ---------------- Фильтры ----------------
    def create_negative(self):
        if self.current_np is None:
            return
        negative = 255 - self.current_np
        self.update_display(Image.fromarray(negative))

    def adjust_brightness(self, value):
        if self.current_np is None:
            return
        adjusted = np.clip(self.current_np.astype(np.int16) + int(value), 0, 255).astype(np.uint8)
        self.update_display(Image.fromarray(adjusted))

    def adjust_contrast(self, value):
        if self.current_np is None:
            return
        mean = np.mean(self.current_np, axis=(0, 1))
        factor = value if isinstance(value, (int, float)) else 1.0
        adjusted = np.clip((self.current_np - mean) * factor + mean, 0, 255).astype(np.uint8)
        self.update_display(Image.fromarray(adjusted))

    def swap_channels(self, order):
        if self.current_np is None:
            return
        swapped = self.current_np[:, :, order]
        self.update_display(Image.fromarray(swapped))

    def flip_image(self, horizontal=True):
        if self.current_np is None:
            return
        if horizontal:
            flipped_np = self.current_np[:, ::-1].copy()
        else:
            flipped_np = self.current_np[::-1].copy()
        self.update_display(Image.fromarray(flipped_np))

    # ---------------- Служебные методы ----------------
    @staticmethod
    def pil2pixmap(im: Image.Image) -> QPixmap:
        if im.mode != "RGB":
            im = im.convert("RGB")
        data = im.tobytes("raw", "RGB")
        qimage = QImage(data, im.width, im.height, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def update_display(self, image: Image.Image):
        if image is None:
            return
        self.current_pil = image
        self.current_np = np.array(image)

        pixmap = self.pil2pixmap(image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        self.update_global_stats()
        self.update_channel_previews()
        self.update_histograms()

    # ---------------- Работа с файлами ----------------
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "",
            "Изображения (*.png *.bmp *.tiff *.tif);;Все файлы (*)"
        )
        if not file_path:
            return
        try:
            self.original_pil = Image.open(file_path).convert("RGB")
            self.original_np = np.array(self.original_pil)
            self.current_pil = self.original_pil.copy()
            self.current_np = np.array(self.current_pil)
            self.filename_label.setText(os.path.basename(file_path))
            self.update_display(self.current_pil)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")

    def save_image(self):
        if self.current_pil is None:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для сохранения")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "",
            "PNG (*.png);;BMP (*.bmp);;TIFF (*.tiff);;Все файлы (*)"
        )
        if file_path:
            try:
                self.current_pil.save(file_path)
                QMessageBox.information(self, "Успех", "Изображение сохранено")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {str(e)}")

    def reset_image(self):
        if self.original_pil is None:
            return
        self.update_display(self.original_pil.copy())

    # ---------------- Обработка курсора ----------------
    def mouse_move_event(self, event):
        if self.current_np is None or self.current_pil is None:
            return
        try:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            orig_w, orig_h = self.current_pil.width, self.current_pil.height
            label_w, label_h = self.image_label.width(), self.image_label.height()
            scale = min(label_w / orig_w, label_h / orig_h)
            disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
            x_offset = (label_w - disp_w) // 2
            y_offset = (label_h - disp_h) // 2
            lx, ly = pos.x(), pos.y()
            if lx < x_offset or lx >= x_offset + disp_w or ly < y_offset or ly >= y_offset + disp_h:
                self.coords_label.setText("Координаты: за границами")
                self.cursor_label.clear()
                return
            img_x = int((lx - x_offset) / scale)
            img_y = int((ly - y_offset) / scale)
            img_x = max(0, min(orig_w - 1, img_x))
            img_y = max(0, min(orig_h - 1, img_y))
            self.show_pixel_info(img_x, img_y)
            self.show_window_frame(img_x, img_y)
            self.update_cursor_window(img_x, img_y)
            self.coords_label.setText(f"Координаты: ({img_x}, {img_y})")
        except Exception as e:
            print(f"Ошибка в обработке мыши: {e}")

    def show_pixel_info(self, x, y):
        if self.current_np is None:
            return
        r, g, b = map(int, self.current_np[y, x])
        intensity = (r + g + b) / 3.0
        info_text = f"Координаты: ({x}, {y})\nRGB: ({r}, {g}, {b})\nИнтенсивность: {intensity:.1f}"
        self.pixel_info_label.setText(info_text)

    def show_window_frame(self, x, y):
        if self.current_pil is None:
            return
        try:
            img_copy = self.current_pil.copy()
            draw = ImageDraw.Draw(img_copy)
            left = max(0, x - 6)
            top = max(0, y - 6)
            right = min(img_copy.width - 1, x + 6)
            bottom = min(img_copy.height - 1, y + 6)
            draw.rectangle([left, top, right, bottom], outline="yellow", width=2)
            pixmap = self.pil2pixmap(img_copy)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Ошибка в show_window_frame: {e}")

    def update_cursor_window(self, x, y):
        if self.current_np is None:
            self.cursor_label.clear()
            return
        try:
            h, w = self.current_np.shape[:2]
            WINDOW_RADIUS = 6
            WINDOW_SIZE = 2 * WINDOW_RADIUS + 1

            # Координаты окна с проверкой выхода за границы
            x1 = max(0, x - WINDOW_RADIUS)
            y1 = max(0, y - WINDOW_RADIUS)
            x2 = min(w, x1 + WINDOW_SIZE)
            y2 = min(h, y1 + WINDOW_SIZE)
            x1 = max(0, x2 - WINDOW_SIZE)
            y1 = max(0, y2 - WINDOW_SIZE)

            window = self.current_np[y1:y2, x1:x2]
            if window.size == 0:
                self.stats_label.setText("Окно выходит за границы")
                self.cursor_label.clear()
                return

            # Статистика окна
            if window.ndim == 3:
                mean = np.mean(window, axis=(0, 1))
                std = np.std(window, axis=(0, 1))
                stats_text = (f"Среднее: R={mean[0]:.1f}, G={mean[1]:.1f}, B={mean[2]:.1f}\n"
                              f"Станд. отклонение: R={std[0]:.1f}, G={std[1]:.1f}, B={std[2]:.1f}")
            else:
                mean = np.mean(window)
                std = np.std(window)
                stats_text = f"Среднее: {mean:.2f}\nСтанд. отклонение: {std:.2f}"
            self.stats_label.setText(stats_text)

            # Увеличение окна
            SCALE = 20
            window_pil = self.add_grid_to_window(window, scale=SCALE)

            # QLabel под размер увеличенного окна
            self.cursor_label.setScaledContents(False)
            self.cursor_label.setPixmap(self.pil2pixmap(window_pil))
            w, h = window_pil.size
            self.cursor_label.setFixedSize(w, h)
            self.cursor_label.repaint()  # чтобы сразу обновилось

        except Exception as e:
            print(f"Ошибка в update_cursor_window: {e}")
            self.cursor_label.clear()



    def add_grid_to_window(self, window, scale=20):
        """
        window: numpy array (h,w,3) или (h,w)
        scale: сколько увеличиваем каждый пиксель
        """
        pil_img = Image.fromarray(window)
        w, h = pil_img.size
        big_w, big_h = w * scale, h * scale

        # увеличиваем без интерполяции, пиксели квадратные
        pil_big = pil_img.resize((big_w, big_h), Image.Resampling.NEAREST)

        draw = ImageDraw.Draw(pil_big)
        line_color = (200, 200, 200) if pil_big.mode == "RGB" else 200

        # вертикальные линии
        for i in range(w + 1):
            draw.line([(i * scale, 0), (i * scale, big_h)], fill=line_color)

        # горизонтальные линии
        for j in range(h + 1):
            draw.line([(0, j * scale), (big_w, j * scale)], fill=line_color)

        return pil_big


    # ---------------- Гистограммы и каналы ----------------
    def calculate_histogram(self, image_array):
        if len(image_array.shape) == 3:
            hist = np.zeros((256, 3))
            for i in range(3):
                hist[:, i] = np.histogram(image_array[:, :, i], bins=256, range=(0, 255))[0]
        else:
            hist = np.histogram(image_array, bins=256, range=(0, 255))[0]
        return hist

    def plot_histogram(self, hist_data, is_color=False):
        height, width = 150, 256
        hist_img = np.ones((height, width, 3), dtype=np.uint8) * 240
        if is_color and len(hist_data.shape) > 1:
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            for i in range(3):
                if hist_data[:, i].max() > 0:
                    normalized = (hist_data[:, i] / hist_data[:, i].max() * (height - 20)).astype(int)
                    for x in range(width):
                        if x < len(normalized):
                            y_start = height - 10 - normalized[x]
                            hist_img[y_start:height-10, x, :] = colors[i]  # уже есть, но можно добавить
                            hist_img[y_start:height-10, x, :] = colors[i]

        else:
            if hist_data.max() > 0:
                normalized = (hist_data / hist_data.max() * (height - 20)).astype(int)
                for x in range(width):
                    if x < len(normalized) and normalized[x] > 0:
                        y_start = height - 10 - normalized[x]
                        hist_img[y_start:height - 10, x, :] = [100, 100, 100]
        hist_img[height - 10:height, :, :] = [0, 0, 0]
        return Image.fromarray(hist_img)

    def update_histograms(self):
        if self.current_np is None:
            return
        main_hist = self.calculate_histogram(self.current_np)
        self.main_hist_label.setPixmap(self.pil2pixmap(self.plot_histogram(main_hist, True)))
        gray = np.dot(self.current_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        gray_hist = self.calculate_histogram(gray)
        self.gray_hist_label.setPixmap(self.pil2pixmap(self.plot_histogram(gray_hist)))
        for i, frame in enumerate([self.red_hist_label, self.green_hist_label, self.blue_hist_label]):
            channel_hist = self.calculate_histogram(self.current_np[:, :, i])
            frame.setPixmap(self.pil2pixmap(self.plot_histogram(channel_hist)))

    def update_channel_previews(self):
        if self.current_np is None:
            return
        img = self.current_np
        gray_only = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        self.gray_label.setPixmap(self.pil2pixmap(Image.fromarray(gray_only)))
        red = np.zeros_like(img); red[:, :, 0] = img[:, :, 0]
        self.red_label.setPixmap(self.pil2pixmap(Image.fromarray(red)))
        green = np.zeros_like(img); green[:, :, 1] = img[:, :, 1]
        self.green_label.setPixmap(self.pil2pixmap(Image.fromarray(green)))
        blue = np.zeros_like(img); blue[:, :, 2] = img[:, :, 2]
        self.blue_label.setPixmap(self.pil2pixmap(Image.fromarray(blue)))

    # ---------------- Размещение элементов ----------------
    def frames_setup(self):
        self.image_label = QLabel(self.img_frame)
        self.image_label.resize(self.img_frame.size())
        self.image_label.setScaledContents(False)
        self.image_label.setMouseTracking(True)
        self.main_hist_label = QLabel(self.centralwidget)
        self.main_hist_label.setGeometry(QRect(20, 570, 512, 100))
        self.main_hist_label.setScaledContents(True)
        self.main_hist_label.setStyleSheet("border: 1px solid gray;")

        self.window_stats_label = QLabel(self.centralwidget)
        self.window_stats_label.setGeometry(QRect(550, 570, 200, 50))
        self.window_stats_label.setStyleSheet("border: 1px solid gray;")
        self.window_stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        
        self.gray_label = QLabel(self.gray_frame); self.gray_label.resize(self.gray_frame.size()); self.gray_label.setScaledContents(True)
        self.gray_hist_label = QLabel(self.gray_hist); self.gray_hist_label.resize(self.gray_hist.size()); self.gray_hist_label.setScaledContents(True)
        self.red_label = QLabel(self.red_frame); self.red_label.resize(self.red_frame.size()); self.red_label.setScaledContents(True)
        self.red_hist_label = QLabel(self.red_hist); self.red_hist_label.resize(self.red_hist.size()); self.red_hist_label.setScaledContents(True)
        self.blue_label = QLabel(self.blue_frame); self.blue_label.resize(self.blue_frame.size()); self.blue_label.setScaledContents(True)
        self.blue_hist_label = QLabel(self.blue_hist); self.blue_hist_label.resize(self.blue_hist.size()); self.blue_hist_label.setScaledContents(True)
        self.green_label = QLabel(self.green_frame); self.green_label.resize(self.green_frame.size()); self.green_label.setScaledContents(True)
        self.green_hist_label = QLabel(self.green_hist); self.green_hist_label.resize(self.green_hist.size()); self.green_hist_label.setScaledContents(True)
        self.cursor_label = QLabel(self.cursor_frame)
        self.cursor_label.resize(self.cursor_frame.size())
        self.cursor_label.setScaledContents(False)
        self.cursor_label.setAlignment(Qt.AlignCenter)
        self.cursor_label.setStyleSheet("border: 2px solid red; background-color: black;")
        self.cursor_label.setText("Наведите курсор\nна изображение")


# === Запуск приложения ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
