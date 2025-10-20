# -*- coding: utf-8 -*-
import sys
import numpy as np
from PIL import Image, ImageQt
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QCheckBox, QButtonGroup, QRadioButton, QFrame
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from ui_main import Ui_Imchanger 


class ImchangerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Imchanger()
        self.ui.setupUi(self)

        self.frames_setup()
        
        self.original_image = None
        self.chroma_image = None
        self.smooth_image = None
        self.clarity_image = None

        self.ui.load_button.clicked.connect(self.load_image)
        

    def frames_setup(self):
        # главная страница
        self.origin_img_label = QLabel(self.ui.origin_img_frame)
        self.origin_img_label.resize(self.ui.origin_img_frame.size())
        self.origin_img_label.setScaledContents(True)
        self.origin_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

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
        
        # страница цветности
        self.chroma_label = QLabel(self.ui.chroma_frame)
        self.chroma_label.resize(self.ui.chroma_frame.size())
        self.chroma_label.setScaledContents(True)
        self.chroma_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    

    def load_image(self):
        """Загружает изображение и отображает его."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                # 1. Загружаем изображение с помощью Pillow
                self.original_image = Image.open(file_path)#.convert("L") # Конвертируем в градации серого
                self.chroma_image = self.original_image.copy()
                self.smooth_image = self.original_image.copy()
                self.clarity_image = self.original_image.copy()

                # 2. Отображаем оригинальное изображение на главной вкладке
                self.display_original_image_in_frame()
                # self.display_image_on_frame(self.original_image, self.ui.origin_img_frame)
                self.display_image_on_frame(self.original_image, self.ui.chroma_img_frame)
                # self.display_image_on_frame(self.original_image, self.ui.origin_img_frame)
                # self.display_image_on_frame(self.original_image, self.ui.origin_img_frame)

            except Exception as e:
                print(f"Ошибка загрузки изображения: {e}")

    def display_original_image_in_frame(self):
        if self.original_image is None:
            return
        pixmap = self.original_image.toqpixmap()
        self.origin_img_label.setPixmap(pixmap.scaled(self.origin_img_label.size(), 
                                                 Qt.KeepAspectRatio, # Масштабирование с сохранением пропорций
                                                 Qt.SmoothTransformation)) # сглаживание по сути

    def display_image_on_frame(self, pil_image: Image.Image, frame: QFrame):
        """Отображает изображение PIL на заданном QFrame."""
        # Если pil_image пустое, просто очищаем
        if pil_image is None:
            return

        pixmap = pil_image.toqpixmap()

        # Создаем QLabel и устанавливаем ему QPixmap
        label = QLabel(frame)
        label.setPixmap(pixmap.scaled(frame.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setGeometry(0, 0, frame.width(), frame.height())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImchangerApp()
    window.show()
    sys.exit(app.exec())
