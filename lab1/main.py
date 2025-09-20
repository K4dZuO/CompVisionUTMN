import os
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QLabel, QFrame
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PIL import Image
import numpy as np

# Импортируем сгенерированный UI
from ui_main import Ui_Imchanger


class MainWindow(QMainWindow, Ui_Imchanger):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Устанавливаем UI из .ui файла
        self.setWindowTitle("ImChanger — Обработка изображений")

        # Подготовка отображения изображений
        self.frames_setup()

        # Настройка кнопки загрузки
        self.load_button.clicked.connect(self.load_file)

        # Переменная для хранения оригинала
        self.original_pil = None
        self.original_np = None
        
        self.existed_labels = {'main_img': None,
                               'red_frame': None}
        
    def frames_setup(self):
        # главное из-ие
        self.image_label = QLabel(self.img_frame)  # QLabel внутри img_frame для отображения
        self.image_label.resize(self.img_frame.size())
        self.image_label.setScaledContents(True)  # Автоматическое масштабирование
        
        # черно-белое из-ие
        self.gray_label = QLabel(self.gray_frame)
        self.gray_label.resize(self.gray_frame.size())
        self.gray_label.setScaledContents(True)
        
        
        
        # красное из-ие
        self.red_label = QLabel(self.red_frame)
        self.red_label.resize(self.red_frame.size())
        self.red_label.setScaledContents(True) 
        
        # синее из-ие
        self.blue_label = QLabel(self.blue_frame)
        self.blue_label.resize(self.blue_frame.size())
        self.blue_label.setScaledContents(True) 
        
        # зеленое из-ие
        self.green_label = QLabel(self.green_frame)
        self.green_label.resize(self.green_frame.size())
        self.green_label.setScaledContents(True) 
        
        # курсорное окно
        self.cursor_label = QLabel(self.cursor_frame)
        self.green_label.resize(self.cursor_frame.size())
        self.green_label.setScaledContents(True) 
        
    
    def display_original_image_in_frame(self):
        """Отображает текущее изображение в img_frame с масштабированием"""
        if self.original_pil is None:
            return
        pixmap = self.original_pil.toqpixmap()
        self.image_label.setPixmap(pixmap)
     
    
    def update_channel_previews(self):
        """Обновляет превью: ч/б и три цветовых канала"""
        if self.original_np is None:
            return

        img = self.original_np  # (H, W, 3) — RGB

        # === 1. Ч/Б (вручную): Y = 0.299*R + 0.587*G + 0.114*B ===
        gray_only = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        gray_pil = Image.fromarray(gray_only)
        gray_pixmap = gray_pil.toqpixmap()
        self.gray_label.setPixmap(gray_pixmap)

        # === 2. Красный канал: R, G=0, B=0 ===
        red_only = np.zeros_like(img)
        red_only[:, :, 0] = img[:, :, 0]
        red_pil = Image.fromarray(red_only)
        red_pixmap = red_pil.toqpixmap()
        self.red_label.setPixmap(red_pixmap)

        # === 3. Зелёный канал: G, R=0, B=0 ===
        green_only = np.zeros_like(img)
        green_only[:, :, 1] = img[:, :, 1]
        green_pil = Image.fromarray(green_only)
        green_pixmap = green_pil.toqpixmap()
        self.green_label.setPixmap(green_pixmap)

        # === 4. Синий канал: B, R=0, G=0 ===
        blue_only = np.zeros_like(img)
        blue_only[:, :, 2] = img[:, :, 2]
        blue_pil = Image.fromarray(blue_only)
        blue_pixmap = blue_pil.toqpixmap()
        self.blue_label.setPixmap(blue_pixmap)
        
        
    def load_file(self):
        """Открывает диалог выбора файла и загружает изображение"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть изображение",
            "",
            "Изображения (*.png *.bmp *.tiff *.tif);;Все файлы (*)"
        )

        if not file_path:
            return  # Пользователь отменил выбор

        try:
            self.original_pil = Image.open(file_path).convert("RGB") # для отображения
            self.original_np = np.array(self.original_pil)  # для анализа

            self.filename_label.setText(os.path.basename(file_path)) # показатьназвание файла
            self.display_original_image_in_frame()
            self.update_channel_previews()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")

    

# === Запуск приложения ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
