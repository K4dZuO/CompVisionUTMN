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
        
        self.dct_images = {}

        self.ui.load_button.clicked.connect(self.load_image)
        

    def frames_setup(self):
        # главная страница
        self.origin_main_label = QLabel(self.ui.origin_main_frame)
        self.origin_main_label.resize(self.ui.origin_main_frame.size())
        self.origin_main_label.setScaledContents(True)
        self.origin_main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.chroma_main_label = QLabel(self.ui.chroma_main_frame)
        self.chroma_main_label.resize(self.ui.chroma_main_frame.size())
        self.chroma_main_label.setScaledContents(True)
        self.chroma_main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.smooth_main_label = QLabel(self.ui.smooth_main_frame)
        self.smooth_main_label.resize(self.ui.smooth_main_frame.size())
        self.smooth_main_label.setScaledContents(True)
        self.smooth_main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.clarity_main_label = QLabel(self.ui.clarity_main_frame)
        self.clarity_main_label.resize(self.ui.clarity_main_frame.size())
        self.clarity_main_label.setScaledContents(True)
        self.clarity_main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # страница цветности
        self.chroma_label = QLabel(self.ui.chroma_frame)
        self.chroma_label.resize(self.ui.chroma_frame.size())
        self.chroma_label.setScaledContents(True)
        self.chroma_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
        # страница сглаживания
        self.smooth_label = QLabel(self.ui.smooth_frame)
        self.smooth_label.resize(self.ui.smooth_frame.size())
        self.smooth_label.setScaledContents(True)
        self.smooth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # страница резкости
        self.clarity_label = QLabel(self.ui.clarity_frame)
        self.clarity_label.resize(self.ui.clarity_frame.size())
        self.clarity_label.setScaledContents(True)
        self.clarity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def load_image(self):
        """Загружает изображение и отображает его."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                # 1. Загружаем изображение с помощью Pillow
                self.original_image = Image.open(file_path).convert("L") # Конвертируем в градации серого
                self.chroma_image = self.original_image.copy()
                self.smooth_image = self.original_image.copy()
                self.clarity_image = self.original_image.copy()

                # 2. Отображаем оригинальное изображение на главной вкладке
                # self.display_original_image_in_frame()
                # self.display_image_on_frame(self.original_image, self.ui.origin_main_frame)
                self.display_image_on_frame(self.original_image, "chroma_main_frame")
                self.display_image_on_frame(self.original_image, "original_main_frame")
                self.display_image_on_frame(self.original_image, "smooth_main_frame")
                self.display_image_on_frame(self.original_image, "clarity_main_frame")
                self.display_image_on_frame(self.original_image, "clarity_main_frame")
                self.display_image_on_frame(self.original_image, "chroma_frame")
                self.display_image_on_frame(self.original_image, "smooth_frame")
                self.display_image_on_frame(self.original_image, "clarity_frame")
                
                # self.display_image_on_frame(self.original_image, self.ui.)
                # self.display_image_on_frame(self.original_image, self.ui.origin_main_frame)

            except Exception as e:
                print(f"Ошибка загрузки изображения: {e}")

    def display_original_image_in_frame(self):
        if self.original_image is None:
            return
        pixmap = self.original_image.toqpixmap()
        self.origin_main_label.setPixmap(pixmap.scaled(self.origin_main_label.size(), 
                                                 Qt.KeepAspectRatio, # Масштабирование с сохранением пропорций
                                                 Qt.SmoothTransformation)) # сглаживание по сути
    
    def display_image_on_frame(self, pil_image: Image.Image, name_frame: str):
        pixmap = pil_image.toqpixmap()
        print(pixmap)
        
        match name_frame:
            case "original_main_frame":
                self.origin_main_label.setPixmap(pixmap.scaled(self.origin_main_label.size()))
            case "chroma_main_frame":
                self.chroma_main_label.setPixmap(pixmap.scaled(self.chroma_main_label.size()))
            case "smooth_main_frame":
                self.smooth_main_label.setPixmap(pixmap.scaled(self.smooth_main_label.size()))
            case "clarity_main_frame":
                self.clarity_main_label.setPixmap(pixmap.scaled(self.clarity_main_label.size()))
            case "chroma_frame":
                self.chroma_label.setPixmap(pixmap.scaled(self.chroma_label.size()))
            case "smooth_frame":
                self.smooth_label.setPixmap(pixmap.scaled(self.smooth_label.size()))
            case "clarity_frame":
                self.clarity_label.setPixmap(pixmap.scaled(self.clarity_label.size()))
            case _:
                print("Неизвестный фрейм")
                
            
                

    # def display_image_on_frame(self, pil_image: Image.Image, frame: QFrame):
    #     """Отображает изображение PIL на заданном QFrame."""
    #     # Если pil_image пустое, просто очищаем
    #     if pil_image is None:
    #         return

    #     pixmap = pil_image.toqpixmap()

    #     # Создаем QLabel и устанавливаем ему QPixmap
    #     label = QLabel(frame)
    #     label.setPixmap(pixmap.scaled(frame.size(), 
    #                                   Qt.AspectRatioMode.KeepAspectRatio, 
    #                                   Qt.TransformationMode.SmoothTransformation))
    #     label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #     label.setGeometry(0, 0, frame.width(), frame.height())
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImchangerApp()
    window.show()
    sys.exit(app.exec())
