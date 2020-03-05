# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

import CV404Filters as backend
import CV404Histograms as hg
from PyQt5 import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QFileDialog
from qtpy.QtGui import QPixmap
import qimage2ndarray

class Ui_MainWindow(object):
    
    def load_image_A_hybrid(self):
        try:
            options = QFileDialog.Options()
            img_A, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(img_A)
            pixmap = pixmap.scaled(self.label_histograms_input_2.width(),self.label_histograms_input_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_input_2.setPixmap(pixmap)
           
        except Exception as err:
            print(err)
    def load_image_B_hybrid(self):
        try:
            options = QFileDialog.Options()
            img_B, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(img_B)
            pixmap = pixmap.scaled(self.label_histograms_hinput_2.width(),self.label_histograms_hinput_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_hinput_2.setPixmap(pixmap)
            
        except Exception as err:
            print(err)
    def histogram_load_btn(self):
        try:
            options = QFileDialog.Options()
            fileName_histo, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(fileName_histo)
            pixmap = pixmap.scaled(self.label_filters_input.width(),self.label_filters_input.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_input.setPixmap(pixmap)
            histo,_= hg.histogram(fileName_histo)
        except Exception as err:
            print(err)
    def filters_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.fileName, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.label_filters_input.width(),self.label_filters_input.height(), QtCore.Qt.KeepAspectRatio)
            self.label_filters_input.setPixmap(pixmap)
            self.comboBox.setEnabled(True)
            self.label_filters_output.clear()
        except Exception as err:
            print(err)
    def getImageFromArray(self,array):
            qimg = qimage2ndarray.array2qimage(array, normalize=True)
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(self.label_filters_output.width(),self.label_filters_output.height(), QtCore.Qt.KeepAspectRatio)
            self.label_filters_output.setPixmap(pixmap)
    def combo_selection(self):
        value = self.comboBox.currentText()

        if value == 'Average Filter':
            img = backend.averageFilter(self.fileName)
            self.getImageFromArray(img)

        elif value == 'Gaussian Filter':
            img = backend.gaussianFilter(self.fileName)
            self.getImageFromArray(img)

        elif value == 'Median Filter':
            img = backend.medianFilter(self.fileName)
            self.getImageFromArray(img)

        elif value == 'Uniform Noise':
            img = backend.uniformNoise(self.fileName)
            self.getImageFromArray(img)
           
        elif value == 'Gaussian Noise':
            img = backend.gaussianNoise(self.fileName)
            self.getImageFromArray(img)    

        elif value == 'Salt & Pepper Noise':
            img = backend.saltNpepperNoise(self.fileName)
            self.getImageFromArray(img)  

        elif value == 'Sobel ED':
            img = backend.sobel(self.fileName)
            self.getImageFromArray(img)

        elif value == 'Roberts ED':
            img = backend.roberts_edge_detection(self.fileName)
            self.getImageFromArray(img)

        elif value == 'Prewitt ED':
            img = backend.prewitt(self.fileName)
            self.getImageFromArray(img)
            
        elif value == 'Canny ED':
            pass
        
                
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(978, 667)
        MainWindow.setMinimumSize(QtCore.QSize(800, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(10, 20, 951, 601))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_filters = QtWidgets.QWidget()
        self.tab_filters.setObjectName("tab_filters")
        self.label_filters_input = QtWidgets.QLabel(self.tab_filters)
        self.label_filters_input.setGeometry(QtCore.QRect(150, 30, 361, 341))
        self.label_filters_input.setFrameShape(QtWidgets.QFrame.Box)
        self.label_filters_input.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_filters_input.setObjectName("label_filters_input")
        self.label_filters_output = QtWidgets.QLabel(self.tab_filters)
        self.label_filters_output.setGeometry(QtCore.QRect(550, 30, 361, 341))
        self.label_filters_output.setFrameShape(QtWidgets.QFrame.Box)
        self.label_filters_output.setTextFormat(QtCore.Qt.PlainText)
        self.label_filters_output.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_filters_output.setObjectName("label_filters_output")
        self.pushButton_filters_load = QtWidgets.QPushButton(self.tab_filters)
        self.pushButton_filters_load.setGeometry(QtCore.QRect(10, 30, 121, 81))
        self.pushButton_filters_load.setObjectName("pushButton_filters_load")
        self.label = QtWidgets.QLabel(self.tab_filters)
        self.label.setGeometry(QtCore.QRect(10, 140, 71, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab_filters)
        self.label_2.setGeometry(QtCore.QRect(10, 170, 71, 21))
        self.label_2.setObjectName("label_2")
        self.groupBox = QtWidgets.QGroupBox(self.tab_filters)
        self.groupBox.setGeometry(QtCore.QRect(140, 420, 771, 111))
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 40, 311, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout.addWidget(self.comboBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(380, 30, 311, 71))
        self.groupBox_2.setObjectName("groupBox_2")
        self.tabWidget.addTab(self.tab_filters, "")
        self.tab_histograms = QtWidgets.QWidget()
        self.tab_histograms.setObjectName("tab_histograms")
        self.label_histograms_houtput = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_houtput.setGeometry(QtCore.QRect(560, 270, 361, 281))
        self.label_histograms_houtput.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_houtput.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_houtput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_houtput.setObjectName("label_histograms_houtput")
        self.pushButton_histograms_load = QtWidgets.QPushButton(self.tab_histograms)
        self.pushButton_histograms_load.setGeometry(QtCore.QRect(40, 20, 121, 81))
        self.pushButton_histograms_load.setObjectName("pushButton_histograms_load")
        self.label_histograms_input = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_input.setGeometry(QtCore.QRect(200, 20, 341, 241))
        self.label_histograms_input.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_input.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_input.setObjectName("label_histograms_input")
        self.label_histograms_output = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_output.setGeometry(QtCore.QRect(560, 20, 361, 241))
        self.label_histograms_output.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_output.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_output.setObjectName("label_histograms_output")
        self.label_10 = QtWidgets.QLabel(self.tab_histograms)
        self.label_10.setGeometry(QtCore.QRect(50, 160, 71, 21))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.tab_histograms)
        self.label_11.setGeometry(QtCore.QRect(50, 130, 71, 21))
        self.label_11.setObjectName("label_11")
        self.label_histograms_hinput = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_hinput.setGeometry(QtCore.QRect(200, 270, 341, 281))
        self.label_histograms_hinput.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_hinput.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_hinput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_hinput.setObjectName("label_histograms_hinput")
        self.tabWidget.addTab(self.tab_histograms, "")
        self.tab_hybrid = QtWidgets.QWidget()
        self.tab_hybrid.setObjectName("tab_hybrid")
        self.label_histograms_input_2 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_histograms_input_2.setGeometry(QtCore.QRect(180, 20, 301, 241))
        self.label_histograms_input_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_input_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_input_2.setObjectName("label_histograms_input_2")
        self.label_histograms_hinput_2 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_histograms_hinput_2.setGeometry(QtCore.QRect(180, 270, 301, 241))
        self.label_histograms_hinput_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_hinput_2.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_hinput_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_hinput_2.setObjectName("label_histograms_hinput_2")
        self.label_histograms_output_2 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_histograms_output_2.setGeometry(QtCore.QRect(490, 20, 431, 491))
        self.label_histograms_output_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_output_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_output_2.setObjectName("label_histograms_output_2")
        self.label_12 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_12.setGeometry(QtCore.QRect(30, 110, 71, 21))
        self.label_12.setObjectName("label_12")
        self.pushButton_histograms_load_2 = QtWidgets.QPushButton(self.tab_hybrid)
        self.pushButton_histograms_load_2.setGeometry(QtCore.QRect(20, 20, 121, 81))
        self.pushButton_histograms_load_2.setObjectName("pushButton_histograms_load_2")
        self.label_13 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_13.setGeometry(QtCore.QRect(30, 130, 71, 21))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_14.setGeometry(QtCore.QRect(30, 310, 71, 21))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_15.setGeometry(QtCore.QRect(30, 290, 71, 21))
        self.label_15.setObjectName("label_15")
        self.pushButton_histograms_load_3 = QtWidgets.QPushButton(self.tab_hybrid)
        self.pushButton_histograms_load_3.setGeometry(QtCore.QRect(20, 200, 121, 81))
        self.pushButton_histograms_load_3.setObjectName("pushButton_histograms_load_3")
        self.pushButton_histograms_load_4 = QtWidgets.QPushButton(self.tab_hybrid)
        self.pushButton_histograms_load_4.setGeometry(QtCore.QRect(20, 350, 121, 41))
        self.pushButton_histograms_load_4.setObjectName("pushButton_histograms_load_4")
        self.tabWidget.addTab(self.tab_hybrid, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 978, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.comboBox.addItems(['Tab to select','Average Filter','Gaussian Filter', 'Median Filter'])
        self.comboBox.addItems(['Uniform Noise','Gaussian Noise', 'Salt & Pepper Noise'])
        self.comboBox.addItems(['Sobel ED','Roberts ED', 'Prewitt ED', 'Canny ED'])
        self.comboBox.setEnabled(False)

        


        self.comboBox.currentTextChanged.connect(self.combo_selection)
        self.pushButton_filters_load.clicked.connect(self.filters_load_btn)
        self.pushButton_histograms_load.clicked.connect(self.histogram_load_btn)
        self.pushButton_histograms_load_2.clicked.connect(self.load_image_A_hybrid)
        self.pushButton_histograms_load_3.clicked.connect(self.load_image_B_hybrid)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_filters_input.setText(_translate("MainWindow", "Input image"))
        self.label_filters_output.setText(_translate("MainWindow", "Output image"))
        self.pushButton_filters_load.setText(_translate("MainWindow", "Load Image"))
        self.label.setText(_translate("MainWindow", "Name:"))
        self.label_2.setText(_translate("MainWindow", "Size:"))
        self.groupBox.setTitle(_translate("MainWindow", "Filter Settings"))
        self.label_3.setText(_translate("MainWindow", "Select Filter"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Additional Parameters"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_filters), _translate("MainWindow", "Filters"))
        self.label_histograms_houtput.setText(_translate("MainWindow", "Output Histogram"))
        self.pushButton_histograms_load.setText(_translate("MainWindow", "Load image"))
        self.label_histograms_input.setText(_translate("MainWindow", "Input image"))
        self.label_histograms_output.setText(_translate("MainWindow", "Output image"))
        self.label_10.setText(_translate("MainWindow", "Size:"))
        self.label_11.setText(_translate("MainWindow", "Name:"))
        self.label_histograms_hinput.setText(_translate("MainWindow", "Input Histogram"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_histograms), _translate("MainWindow", "Histograms"))
        self.label_histograms_input_2.setText(_translate("MainWindow", "Input image A"))
        self.label_histograms_hinput_2.setText(_translate("MainWindow", "Input image B"))
        self.label_histograms_output_2.setText(_translate("MainWindow", "Output image"))
        self.label_12.setText(_translate("MainWindow", "Name:"))
        self.pushButton_histograms_load_2.setText(_translate("MainWindow", "Load image A"))
        self.label_13.setText(_translate("MainWindow", "Size:"))
        self.label_14.setText(_translate("MainWindow", "Size:"))
        self.label_15.setText(_translate("MainWindow", "Name:"))
        self.pushButton_histograms_load_3.setText(_translate("MainWindow", "Load image B"))
        self.pushButton_histograms_load_4.setText(_translate("MainWindow", "Make Hybrid"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_hybrid), _translate("MainWindow", "Hybrid"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
