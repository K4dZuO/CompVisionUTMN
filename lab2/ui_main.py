# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'design_2lab.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QLabel,
    QMainWindow, QPushButton, QSizePolicy, QStatusBar,
    QTabWidget, QWidget)

class Ui_Imchanger(object):
    def setupUi(self, Imchanger):
        if not Imchanger.objectName():
            Imchanger.setObjectName(u"Imchanger")
        Imchanger.resize(1255, 903)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Imchanger.sizePolicy().hasHeightForWidth())
        Imchanger.setSizePolicy(sizePolicy)
        self.actionload_file = QAction(Imchanger)
        self.actionload_file.setObjectName(u"actionload_file")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave))
        self.actionload_file.setIcon(icon)
        self.actionload_file.setMenuRole(QAction.MenuRole.NoRole)
        self.centralwidget = QWidget(Imchanger)
        self.centralwidget.setObjectName(u"centralwidget")
        self.filename_label = QLabel(self.centralwidget)
        self.filename_label.setObjectName(u"filename_label")
        self.filename_label.setGeometry(QRect(190, 10, 331, 31))
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        self.load_button = QPushButton(self.centralwidget)
        self.load_button.setObjectName(u"load_button")
        self.load_button.setGeometry(QRect(20, 10, 161, 26))
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(20, 60, 1211, 781))
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.main_tab = QWidget()
        self.main_tab.setObjectName(u"main_tab")
        self.origin_img_frame = QFrame(self.main_tab)
        self.origin_img_frame.setObjectName(u"origin_img_frame")
        self.origin_img_frame.setGeometry(QRect(270, 30, 350, 350))
        self.origin_img_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.origin_img_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.chroma_img_frame = QFrame(self.main_tab)
        self.chroma_img_frame.setObjectName(u"chroma_img_frame")
        self.chroma_img_frame.setGeometry(QRect(270, 380, 350, 350))
        self.chroma_img_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.chroma_img_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.smooth_img_frame = QFrame(self.main_tab)
        self.smooth_img_frame.setObjectName(u"smooth_img_frame")
        self.smooth_img_frame.setGeometry(QRect(620, 30, 350, 350))
        self.smooth_img_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.smooth_img_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.clarity_img_frame = QFrame(self.main_tab)
        self.clarity_img_frame.setObjectName(u"clarity_img_frame")
        self.clarity_img_frame.setGeometry(QRect(620, 380, 350, 350))
        self.clarity_img_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.clarity_img_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.label1 = QLabel(self.main_tab)
        self.label1.setObjectName(u"label1")
        self.label1.setGeometry(QRect(110, 190, 121, 51))
        font = QFont()
        font.setPointSize(16)
        self.label1.setFont(font)
        self.label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label2 = QLabel(self.main_tab)
        self.label2.setObjectName(u"label2")
        self.label2.setGeometry(QRect(89, 520, 121, 51))
        self.label2.setFont(font)
        self.label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label3 = QLabel(self.main_tab)
        self.label3.setObjectName(u"label3")
        self.label3.setGeometry(QRect(1020, 190, 121, 51))
        self.label3.setFont(font)
        self.label3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label4 = QLabel(self.main_tab)
        self.label4.setObjectName(u"label4")
        self.label4.setGeometry(QRect(1030, 540, 121, 51))
        self.label4.setFont(font)
        self.label4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tabWidget.addTab(self.main_tab, "")
        self.chroma_tab = QWidget()
        self.chroma_tab.setObjectName(u"chroma_tab")
        self.chroma_frame = QFrame(self.chroma_tab)
        self.chroma_frame.setObjectName(u"chroma_frame")
        self.chroma_frame.setGeometry(QRect(80, 70, 600, 600))
        self.chroma_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.chroma_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.label = QLabel(self.chroma_tab)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(730, 150, 211, 20))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.checkBox = QCheckBox(self.chroma_tab)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(780, 90, 131, 23))
        self.tabWidget.addTab(self.chroma_tab, "")
        self.smooth_tab = QWidget()
        self.smooth_tab.setObjectName(u"smooth_tab")
        self.tabWidget.addTab(self.smooth_tab, "")
        self.clarity_tab = QWidget()
        self.clarity_tab.setObjectName(u"clarity_tab")
        self.tabWidget.addTab(self.clarity_tab, "")
        Imchanger.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(Imchanger)
        self.statusbar.setObjectName(u"statusbar")
        Imchanger.setStatusBar(self.statusbar)

        self.retranslateUi(Imchanger)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Imchanger)
    # setupUi

    def retranslateUi(self, Imchanger):
        Imchanger.setWindowTitle(QCoreApplication.translate("Imchanger", u"MainWindow", None))
        self.actionload_file.setText(QCoreApplication.translate("Imchanger", u"load_file", None))
        self.filename_label.setText(QCoreApplication.translate("Imchanger", u"Your file", None))
        self.load_button.setText(QCoreApplication.translate("Imchanger", u"Load File", None))
        self.label1.setText(QCoreApplication.translate("Imchanger", u"Original", None))
        self.label2.setText(QCoreApplication.translate("Imchanger", u"Cromaticity", None))
        self.label3.setText(QCoreApplication.translate("Imchanger", u"Smoothing", None))
        self.label4.setText(QCoreApplication.translate("Imchanger", u"Clarity", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.main_tab), QCoreApplication.translate("Imchanger", u"Main page", None))
        self.label.setText("")
        self.checkBox.setText(QCoreApplication.translate("Imchanger", u"Ln transform", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.chroma_tab), QCoreApplication.translate("Imchanger", u"\u0421hromaticity", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.smooth_tab), QCoreApplication.translate("Imchanger", u"Smoothing", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.clarity_tab), QCoreApplication.translate("Imchanger", u"Clarity", None))
    # retranslateUi

