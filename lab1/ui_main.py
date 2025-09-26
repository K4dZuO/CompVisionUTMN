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
    QStatusBar, QWidget, QSlider)

class Ui_Imchanger(object):
    def setupUi(self, Imchanger):
        if not Imchanger.objectName():
            Imchanger.setObjectName(u"Imchanger")
        Imchanger.resize(1545, 874)
        self.actionload_file = QAction(Imchanger)
        self.actionload_file.setObjectName(u"actionload_file")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave))
        self.actionload_file.setIcon(icon)
        self.actionload_file.setMenuRole(QAction.MenuRole.NoRole)
        self.centralwidget = QWidget(Imchanger)
        self.centralwidget.setObjectName(u"centralwidget")
        self.img_frame = QFrame(self.centralwidget)
        self.img_frame.setObjectName(u"img_frame")
        self.img_frame.setGeometry(QRect(20, 50, 512, 512))
        self.img_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.img_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.filename_label = QLabel(self.centralwidget)
        self.filename_label.setObjectName(u"filename_label")
        self.filename_label.setGeometry(QRect(200, 10, 331, 31))
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        '''
        self.modified = QScrollArea(self.centralwidget)
        self.modified.setObjectName(u"modified")
        self.modified.setGeometry(QRect(20, 590, 571, 181))
        self.modified.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 569, 179))
        self.modified.setWidget(self.scrollAreaWidgetContents)
        '''
        self.load_button = QPushButton(self.centralwidget)
        self.load_button.setObjectName(u"load_button")
        self.load_button.setGeometry(QRect(20, 10, 161, 26))
        self.save_button = QPushButton(self.centralwidget)
        self.save_button.setObjectName(u"save_button")
        self.save_button.setGeometry(QRect(320, 10, 161, 26))
        self.gray_frame = QFrame(self.centralwidget)
        self.gray_frame.setObjectName(u"gray_frame")
        self.gray_frame.setGeometry(QRect(910, 10, 201, 211))
        self.gray_text = QLabel(self.centralwidget)
        self.gray_text.setObjectName(u"gray_text")
        self.gray_text.setGeometry(QRect(830, 70, 71, 91))
        self.gray_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gray_text.setWordWrap(True)
        self.red_text = QLabel(self.centralwidget)
        self.red_text.setObjectName(u"red_text")
        self.red_text.setGeometry(QRect(820, 490, 66, 41))
        self.red_text.setWordWrap(True)
        self.blue_text = QLabel(self.centralwidget)
        self.blue_text.setObjectName(u"blue_text")
        self.blue_text.setGeometry(QRect(830, 290, 66, 41))
        self.blue_text.setWordWrap(True)
        self.green_text = QLabel(self.centralwidget)
        self.green_text.setObjectName(u"green_text")
        self.green_text.setGeometry(QRect(800, 700, 81, 41))
        self.green_text.setWordWrap(True)
        self.red_frame = QFrame(self.centralwidget)
        self.red_frame.setObjectName(u"red_frame")
        self.red_frame.setGeometry(QRect(910, 430, 201, 181))
        self.blue_frame = QFrame(self.centralwidget)
        self.blue_frame.setObjectName(u"blue_frame")
        self.blue_frame.setGeometry(QRect(910, 230, 201, 191))
        self.green_frame = QFrame(self.centralwidget)
        self.green_frame.setObjectName(u"green_frame")
        self.green_frame.setGeometry(QRect(910, 620, 201, 201))
        self.cursor_frame = QFrame(self.centralwidget)
        self.cursor_frame.setObjectName(u"cursor_frame")
        self.cursor_frame.setGeometry(QRect(560, 110, 161, 161))
        self.cursor_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.cursor_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gray_hist = QFrame(self.centralwidget)
        self.gray_hist.setObjectName(u"gray_hist")
        self.gray_hist.setGeometry(QRect(1130, 10, 391, 211))
        self.gray_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.gray_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.blue_hist = QFrame(self.centralwidget)
        self.blue_hist.setObjectName(u"blue_hist")
        self.blue_hist.setGeometry(QRect(1130, 230, 391, 191))
        self.blue_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.blue_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.red_hist = QFrame(self.centralwidget)
        self.red_hist.setObjectName(u"red_hist")
        self.red_hist.setGeometry(QRect(1130, 430, 391, 181))
        self.red_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.red_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.green_hist = QFrame(self.centralwidget)
        self.green_hist.setObjectName(u"green_hist")
        self.green_hist.setGeometry(QRect(1130, 620, 391, 201))
        self.green_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.green_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.coords_label = QLabel(self.centralwidget)
        self.coords_label.setObjectName(u"coords_label")
        self.coords_label.setGeometry(QRect(560, 25, 200, 100))
        self.coords_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        Imchanger.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Imchanger)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1545, 23))
        Imchanger.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Imchanger)
        self.statusbar.setObjectName(u"statusbar")
        Imchanger.setStatusBar(self.statusbar)
        
        self.negative_button = QPushButton(self.centralwidget)
        self.negative_button.setObjectName(u"negative_button")
        self.negative_button.setGeometry(QRect(560, 280, 111, 31))

        self.reset_button = QPushButton(self.centralwidget)
        self.reset_button.setObjectName(u"reset_button")
        self.reset_button.setGeometry(QRect(680, 280, 111, 31))
        
        self.rotate_90_button = QPushButton(self.centralwidget)
        self.rotate_90_button.setObjectName(u"rotate_90_button")
        self.rotate_90_button.setGeometry(QRect(560, 325, 111, 31))

        self.reflect_by_v_button = QPushButton(self.centralwidget)
        self.reflect_by_v_button.setObjectName(u"reflect_by_v_button")
        self.reflect_by_v_button.setGeometry(QRect(680, 325, 111, 31))
        
        self.rgb_to_grb_button = QPushButton(self.centralwidget)
        self.rgb_to_grb_button.setObjectName(u"rgb_to_grb_button")
        self.rgb_to_grb_button.setGeometry(QRect(560, 360, 111, 31))

        self.grb_to_rgb_button = QPushButton(self.centralwidget)
        self.grb_to_rgb_button.setObjectName(u"grb_to_rgb_button")
        self.grb_to_rgb_button.setGeometry(QRect(680, 360, 111, 31))
        
        #self.pseudo_pillow_button.clicked.connect(self.apply_pseudo_pillow)
        #self.pseudo_pillow_button.clicked.setObjectName(u"pil_pseudo_button")
        #self.pseudo_manual_button.clicked.connect(self.apply_pseudo_manual)
        
        self.brightness_slider = QSlider(self.centralwidget)
        self.brightness_slider.setObjectName(u"brightness_slider")
        self.brightness_slider.setGeometry(QRect(560, 430, 231, 22))
        self.brightness_slider.setOrientation(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)

        self.brightness_label = QLabel(self.centralwidget)
        self.brightness_label.setObjectName(u"brightness_label")
        self.brightness_label.setGeometry(QRect(800, 425, 81, 31))
        self.brightness_label.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)

        self.contrast_slider = QSlider(self.centralwidget)
        self.contrast_slider.setObjectName(u"contrast_slider")
        self.contrast_slider.setGeometry(QRect(560, 470, 231, 22))
        self.contrast_slider.setOrientation(Qt.Horizontal)
        self.contrast_slider.setMinimum(1)
        self.contrast_slider.setMaximum(30)
        self.contrast_slider.setValue(10)

        self.contrast_label = QLabel(self.centralwidget)
        self.contrast_label.setObjectName(u"contrast_label")
        self.contrast_label.setGeometry(QRect(800, 465, 81, 31))
        self.contrast_label.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
        
        self.stats_area = QScrollArea(self.centralwidget)
        self.stats_area.setGeometry(QRect(35, 600, 571, 80))
        self.stats_area.setWidgetResizable(True)
        
        '''
        self.stats_label = QLabel(self.centralwidget)
        self.stats_label.setObjectName(u"stats_label")
        self.stats_label.setGeometry(QRect(35, 600, 571, 40))
        self.stats_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.stats_label.setWordWrap(True)
        '''
        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
        self.stats_label.setWordWrap(True)
        self.stats_area.setWidget(self.stats_label)
        
        self.pseudo_pillow_button = QPushButton(self.centralwidget)
        self.pseudo_pillow_button.setObjectName(u"pseudo_pillow_button")
        self.pseudo_pillow_button.setGeometry(QRect(35, 700, 120, 25))
        self.pseudo_manual_button = QPushButton(self.centralwidget)
        self.pseudo_manual_button.setObjectName(u"pseudo_manual_button")
        self.pseudo_manual_button.setGeometry(QRect(35, 725, 120, 25))
        

        
        self.retranslateUi(Imchanger)

        QMetaObject.connectSlotsByName(Imchanger)
    # setupUi

    def retranslateUi(self, Imchanger):
        Imchanger.setWindowTitle(QCoreApplication.translate("Imchanger", u"MainWindow", None))
        self.actionload_file.setText(QCoreApplication.translate("Imchanger", u"load_file", None))
        self.filename_label.setText(QCoreApplication.translate("Imchanger", u"Your file", None))
        self.load_button.setText(QCoreApplication.translate("Imchanger", u"Load File", None))
        self.save_button.setText(QCoreApplication.translate("Imchanger", u"Save File", None))
        self.gray_text.setText(QCoreApplication.translate("Imchanger", u"W/B preview", None))
        self.red_text.setText(QCoreApplication.translate("Imchanger", u"Only red preview", None))
        self.blue_text.setText(QCoreApplication.translate("Imchanger", u"Only blue preview", None))
        self.green_text.setText(QCoreApplication.translate("Imchanger", u"Only green preview", None))
        self.coords_label.setText(QCoreApplication.translate("Imchanger", u"Coords", None))
        self.reset_button.setText(QCoreApplication.translate("Imchanger", u"Reset", None))
        self.negative_button.setText(QCoreApplication.translate("Imchanger", u"Negative", None))
        self.rotate_90_button.setText(QCoreApplication.translate("Imchanger", u"↺", None))
        self.rgb_to_grb_button.setText(QCoreApplication.translate("Imchanger", u"RGB -> GRB", None))
        self.grb_to_rgb_button.setText(QCoreApplication.translate("Imchanger", u"GRB -> RGB", None))
        self.reflect_by_v_button.setText(QCoreApplication.translate("Imchanger", u"Reflect V", None))
        self.brightness_label.setText(QCoreApplication.translate("Imchanger", u"Brightness", None))
        self.contrast_label.setText(QCoreApplication.translate("Imchanger", u"Contrast", None))
        self.stats_label.setText(QCoreApplication.translate("Imchanger", u"Stats", None))
        self.pseudo_pillow_button.setText(QCoreApplication.translate("Imchanger", u"pseudocolors PIL", None))
        self.pseudo_manual_button.setText(QCoreApplication.translate("Imchanger", u"pseudocolors manual", None))
    # retranslateUi

