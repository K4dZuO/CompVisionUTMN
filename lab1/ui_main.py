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
    QStatusBar, QWidget)

class Ui_Imchanger(object):
    def setupUi(self, Imchanger):
        if not Imchanger.objectName():
            Imchanger.setObjectName(u"Imchanger")
        Imchanger.resize(1275, 832)
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
        self.modified = QScrollArea(self.centralwidget)
        self.modified.setObjectName(u"modified")
        self.modified.setGeometry(QRect(20, 590, 571, 181))
        self.modified.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 569, 179))
        self.modified.setWidget(self.scrollAreaWidgetContents)
        self.load_button = QPushButton(self.centralwidget)
        self.load_button.setObjectName(u"load_button")
        self.load_button.setGeometry(QRect(20, 10, 161, 26))
        self.gray_frame = QFrame(self.centralwidget)
        self.gray_frame.setObjectName(u"gray_frame")
        self.gray_frame.setGeometry(QRect(850, 30, 150, 150))
        self.gray_text = QLabel(self.centralwidget)
        self.gray_text.setObjectName(u"gray_text")
        self.gray_text.setGeometry(QRect(740, 70, 121, 91))
        self.gray_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.red_text = QLabel(self.centralwidget)
        self.red_text.setObjectName(u"red_text")
        self.red_text.setGeometry(QRect(780, 420, 66, 41))
        self.red_text.setWordWrap(True)
        self.blue_text = QLabel(self.centralwidget)
        self.blue_text.setObjectName(u"blue_text")
        self.blue_text.setGeometry(QRect(770, 250, 66, 41))
        self.blue_text.setWordWrap(True)
        self.green_text = QLabel(self.centralwidget)
        self.green_text.setObjectName(u"green_text")
        self.green_text.setGeometry(QRect(770, 600, 81, 41))
        self.green_text.setWordWrap(True)
        self.red_frame = QFrame(self.centralwidget)
        self.red_frame.setObjectName(u"red_frame")
        self.red_frame.setGeometry(QRect(850, 370, 150, 150))
        self.blue_frame = QFrame(self.centralwidget)
        self.blue_frame.setObjectName(u"blue_frame")
        self.blue_frame.setGeometry(QRect(850, 200, 150, 150))
        self.green_frame = QFrame(self.centralwidget)
        self.green_frame.setObjectName(u"green_frame")
        self.green_frame.setGeometry(QRect(850, 550, 150, 150))
        self.cursor_frame = QFrame(self.centralwidget)
        self.cursor_frame.setObjectName(u"cursor_frame")
        self.cursor_frame.setGeometry(QRect(560, 100, 161, 161))
        self.cursor_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.cursor_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gray_hist = QFrame(self.centralwidget)
        self.gray_hist.setObjectName(u"gray_hist")
        self.gray_hist.setGeometry(QRect(1010, 30, 251, 151))
        self.gray_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.gray_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.blue_hist = QFrame(self.centralwidget)
        self.blue_hist.setObjectName(u"blue_hist")
        self.blue_hist.setGeometry(QRect(1010, 200, 251, 151))
        self.blue_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.blue_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.red_hist = QFrame(self.centralwidget)
        self.red_hist.setObjectName(u"red_hist")
        self.red_hist.setGeometry(QRect(1010, 370, 251, 151))
        self.red_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.red_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.green_hist = QFrame(self.centralwidget)
        self.green_hist.setObjectName(u"green_hist")
        self.green_hist.setGeometry(QRect(1010, 550, 251, 151))
        self.green_hist.setFrameShape(QFrame.Shape.StyledPanel)
        self.green_hist.setFrameShadow(QFrame.Shadow.Raised)
        self.coords_label = QLabel(self.centralwidget)
        self.coords_label.setObjectName(u"coords_label")
        self.coords_label.setGeometry(QRect(570, 47, 141, 31))
        self.coords_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        Imchanger.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Imchanger)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1275, 23))
        Imchanger.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Imchanger)
        self.statusbar.setObjectName(u"statusbar")
        Imchanger.setStatusBar(self.statusbar)

        self.retranslateUi(Imchanger)

        QMetaObject.connectSlotsByName(Imchanger)
    # setupUi

    def retranslateUi(self, Imchanger):
        Imchanger.setWindowTitle(QCoreApplication.translate("Imchanger", u"MainWindow", None))
        self.actionload_file.setText(QCoreApplication.translate("Imchanger", u"load_file", None))
        self.filename_label.setText(QCoreApplication.translate("Imchanger", u"Your file", None))
        self.load_button.setText(QCoreApplication.translate("Imchanger", u"Load File", None))
        self.gray_text.setText(QCoreApplication.translate("Imchanger", u"W/B preview", None))
        self.red_text.setText(QCoreApplication.translate("Imchanger", u"Only red preview", None))
        self.blue_text.setText(QCoreApplication.translate("Imchanger", u"Only blue preview", None))
        self.green_text.setText(QCoreApplication.translate("Imchanger", u"Only green preview", None))
        self.coords_label.setText(QCoreApplication.translate("Imchanger", u"Coords", None))
    # retranslateUi

