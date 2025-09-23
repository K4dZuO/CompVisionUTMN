# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'design.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QMenuBar, QPushButton, QScrollArea, QSizePolicy,
    QStatusBar, QWidget, QSlider, QGroupBox, QVBoxLayout,
    QHBoxLayout, QComboBox)

class Ui_Imchanger(object):
    def setupUi(self, Imchanger):
        if not Imchanger.objectName():
            Imchanger.setObjectName(u"Imchanger")
        Imchanger.resize(1400, 900)
        
        self.actionload_file = QAction(Imchanger)
        self.actionload_file.setObjectName(u"actionload_file")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave))
        self.actionload_file.setIcon(icon)
        self.actionload_file.setMenuRole(QAction.MenuRole.NoRole)
        
        self.centralwidget = QWidget(Imchanger)
        self.centralwidget.setObjectName(u"centralwidget")
        
        # Основное изображение
        self.img_frame = QFrame(self.centralwidget)
        self.img_frame.setObjectName(u"img_frame")
        self.img_frame.setGeometry(QRect(20, 50, 512, 512))
        self.img_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.img_frame.setFrameShadow(QFrame.Shadow.Raised)
        
        # Кнопки загрузки и сохранения
        self.load_button = QPushButton(self.centralwidget)
        self.load_button.setObjectName(u"load_button")
        self.load_button.setGeometry(QRect(20, 10, 120, 30))
        
        self.save_button = QPushButton(self.centralwidget)
        self.save_button.setObjectName(u"save_button")
        self.save_button.setGeometry(QRect(150, 10, 120, 30))
        
        # Информация о файле
        self.filename_label = QLabel(self.centralwidget)
        self.filename_label.setObjectName(u"filename_label")
        self.filename_label.setGeometry(QRect(280, 10, 251, 31))
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        
        # Превью каналов
        self.gray_frame = QFrame(self.centralwidget)
        self.gray_frame.setObjectName(u"gray_frame")
        self.gray_frame.setGeometry(QRect(550, 50, 150, 150))
        
        self.red_frame = QFrame(self.centralwidget)
        self.red_frame.setObjectName(u"red_frame")
        self.red_frame.setGeometry(QRect(550, 220, 150, 150))
        
        self.blue_frame = QFrame(self.centralwidget)
        self.blue_frame.setObjectName(u"blue_frame")
        self.blue_frame.setGeometry(QRect(550, 390, 150, 150))
        
        self.green_frame = QFrame(self.centralwidget)
        self.green_frame.setObjectName(u"green_frame")
        self.green_frame.setGeometry(QRect(550, 560, 150, 150))
        
        # Гистограммы
        self.gray_hist = QFrame(self.centralwidget)
        self.gray_hist.setObjectName(u"gray_hist")
        self.gray_hist.setGeometry(QRect(710, 50, 251, 151))
        
        self.red_hist = QFrame(self.centralwidget)
        self.red_hist.setObjectName(u"red_hist")
        self.red_hist.setGeometry(QRect(710, 220, 251, 151))
        
        self.blue_hist = QFrame(self.centralwidget)
        self.blue_hist.setObjectName(u"blue_hist")
        self.blue_hist.setGeometry(QRect(710, 390, 251, 151))
        
        self.green_hist = QFrame(self.centralwidget)
        self.green_hist.setObjectName(u"green_hist")
        self.green_hist.setGeometry(QRect(710, 560, 251, 151))
        
        # Окно курсора
        self.cursor_frame = QFrame(self.centralwidget)
        self.cursor_frame.setObjectName(u"cursor_frame")
        self.cursor_frame.setGeometry(QRect(980, 50, 161, 161))
        
        # Информация о пикселе
        self.pixel_info_group = QGroupBox(self.centralwidget)
        self.pixel_info_group.setObjectName(u"pixel_info_group")
        self.pixel_info_group.setGeometry(QRect(980, 220, 300, 120))
        
        self.pixel_info_label = QLabel(self.pixel_info_group)
        self.pixel_info_label.setObjectName(u"pixel_info_label")
        self.pixel_info_label.setGeometry(QRect(10, 20, 280, 90))
        
        # Статистика окна
        self.stats_group = QGroupBox(self.centralwidget)
        self.stats_group.setObjectName(u"stats_group")
        self.stats_group.setGeometry(QRect(980, 350, 300, 100))
        
        self.stats_label = QLabel(self.stats_group)
        self.stats_label.setObjectName(u"stats_label")
        self.stats_label.setGeometry(QRect(10, 20, 280, 70))
        
        # Панель редактирования
        self.edit_group = QGroupBox(self.centralwidget)
        self.edit_group.setObjectName(u"edit_group")
        self.edit_group.setGeometry(QRect(980, 460, 400, 400))
        
        # Яркость
        self.brightness_label = QLabel(self.edit_group)
        self.brightness_label.setObjectName(u"brightness_label")
        self.brightness_label.setGeometry(QRect(20, 30, 100, 20))
        
        self.brightness_slider = QSlider(self.edit_group)
        self.brightness_slider.setObjectName(u"brightness_slider")
        self.brightness_slider.setGeometry(QRect(130, 30, 200, 20))
        self.brightness_slider.setOrientation(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        
        # Контрастность
        self.contrast_label = QLabel(self.edit_group)
        self.contrast_label.setObjectName(u"contrast_label")
        self.contrast_label.setGeometry(QRect(20, 70, 100, 20))
        
        self.contrast_slider = QSlider(self.edit_group)
        self.contrast_slider.setObjectName(u"contrast_slider")
        self.contrast_slider.setGeometry(QRect(130, 70, 200, 20))
        self.contrast_slider.setOrientation(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        
        # Кнопки редактирования
        self.negative_button = QPushButton(self.edit_group)
        self.negative_button.setObjectName(u"negative_button")
        self.negative_button.setGeometry(QRect(20, 110, 150, 30))
        
        self.flip_h_button = QPushButton(self.edit_group)
        self.flip_h_button.setObjectName(u"flip_h_button")
        self.flip_h_button.setGeometry(QRect(180, 110, 150, 30))
        
        self.flip_v_button = QPushButton(self.edit_group)
        self.flip_v_button.setObjectName(u"flip_v_button")
        self.flip_v_button.setGeometry(QRect(20, 150, 150, 30))
        
        # Обмен каналов
        self.channel_swap_label = QLabel(self.edit_group)
        self.channel_swap_label.setObjectName(u"channel_swap_label")
        self.channel_swap_label.setGeometry(QRect(20, 200, 150, 20))
        
        self.channel_combo = QComboBox(self.edit_group)
        self.channel_combo.setObjectName(u"channel_combo")
        self.channel_combo.setGeometry(QRect(180, 200, 150, 25))
        self.channel_combo.addItems(["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"])
        
        self.swap_button = QPushButton(self.edit_group)
        self.swap_button.setObjectName(u"swap_button")
        self.swap_button.setGeometry(QRect(20, 240, 150, 30))
        
        # Сброс изменений
        self.reset_button = QPushButton(self.edit_group)
        self.reset_button.setObjectName(u"reset_button")
        self.reset_button.setGeometry(QRect(180, 240, 150, 30))
        
        # Метки для превью
        self.gray_text = QLabel(self.centralwidget)
        self.gray_text.setObjectName(u"gray_text")
        self.gray_text.setGeometry(QRect(550, 30, 150, 20))
        self.gray_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.red_text = QLabel(self.centralwidget)
        self.red_text.setObjectName(u"red_text")
        self.red_text.setGeometry(QRect(550, 200, 150, 20))
        self.red_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.blue_text = QLabel(self.centralwidget)
        self.blue_text.setObjectName(u"blue_text")
        self.blue_text.setGeometry(QRect(550, 370, 150, 20))
        self.blue_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.green_text = QLabel(self.centralwidget)
        self.green_text.setObjectName(u"green_text")
        self.green_text.setGeometry(QRect(550, 540, 150, 20))
        self.green_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Координаты
        self.coords_label = QLabel(self.centralwidget)
        self.coords_label.setObjectName(u"coords_label")
        self.coords_label.setGeometry(QRect(980, 30, 161, 20))
        self.coords_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        Imchanger.setCentralWidget(self.centralwidget)
        
        self.menubar = QMenuBar(Imchanger)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1400, 23))
        Imchanger.setMenuBar(self.menubar)
        
        self.statusbar = QStatusBar(Imchanger)
        self.statusbar.setObjectName(u"statusbar")
        Imchanger.setStatusBar(self.statusbar)

        self.retranslateUi(Imchanger)
        QMetaObject.connectSlotsByName(Imchanger)

    def retranslateUi(self, Imchanger):
        Imchanger.setWindowTitle(QCoreApplication.translate("Imchanger", u"ImChanger - Обработка изображений", None))
        self.actionload_file.setText(QCoreApplication.translate("Imchanger", u"Загрузить файл", None))
        self.filename_label.setText(QCoreApplication.translate("Imchanger", u"Файл не выбран", None))
        self.load_button.setText(QCoreApplication.translate("Imchanger", u"Загрузить", None))
        self.save_button.setText(QCoreApplication.translate("Imchanger", u"Сохранить", None))
        
        self.pixel_info_group.setTitle(QCoreApplication.translate("Imchanger", u"Информация о пикселе", None))
        self.pixel_info_label.setText(QCoreApplication.translate("Imchanger", u"Наведите курсор на изображение", None))
        
        self.stats_group.setTitle(QCoreApplication.translate("Imchanger", u"Статистика окна 11x11", None))
        self.stats_label.setText(QCoreApplication.translate("Imchanger", u"Окно не доступно", None))
        
        self.edit_group.setTitle(QCoreApplication.translate("Imchanger", u"Редактирование изображения", None))
        self.brightness_label.setText(QCoreApplication.translate("Imchanger", u"Яркость:", None))
        self.contrast_label.setText(QCoreApplication.translate("Imchanger", u"Контрастность:", None))
        self.negative_button.setText(QCoreApplication.translate("Imchanger", u"Негатив", None))
        self.flip_h_button.setText(QCoreApplication.translate("Imchanger", u"Отразить по гориз.", None))
        self.flip_v_button.setText(QCoreApplication.translate("Imchanger", u"Отразить по верт.", None))
        self.channel_swap_label.setText(QCoreApplication.translate("Imchanger", u"Порядок каналов:", None))
        self.swap_button.setText(QCoreApplication.translate("Imchanger", u"Применить обмен", None))
        self.reset_button.setText(QCoreApplication.translate("Imchanger", u"Сбросить изменения", None))
        
        self.gray_text.setText(QCoreApplication.translate("Imchanger", u"Ч/Б превью", None))
        self.red_text.setText(QCoreApplication.translate("Imchanger", u"Красный канал", None))
        self.blue_text.setText(QCoreApplication.translate("Imchanger", u"Синий канал", None))
        self.green_text.setText(QCoreApplication.translate("Imchanger", u"Зеленый канал", None))
        self.coords_label.setText(QCoreApplication.translate("Imchanger", u"Окно курсора", None))