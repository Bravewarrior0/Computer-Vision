# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1043, 668)
        MainWindow.setMinimumSize(QtCore.QSize(800, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(10, 20, 1011, 601))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_filters = QtWidgets.QWidget()
        self.tab_filters.setObjectName("tab_filters")
        self.groupBox = QtWidgets.QGroupBox(self.tab_filters)
        self.groupBox.setGeometry(QtCore.QRect(190, 410, 781, 141))
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
        self.comboBox_filters = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox_filters.setObjectName("comboBox_filters")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.addItem("")
        self.comboBox_filters.setEnabled(False)
        self.horizontalLayout.addWidget(self.comboBox_filters)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(390, 20, 371, 111))
        self.groupBox_2.setObjectName("groupBox_2")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 30, 141, 31))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.spinBox_filter_size = QtWidgets.QSpinBox(self.layoutWidget)
        self.spinBox_filter_size.setMinimum(3)
        self.spinBox_filter_size.setMaximum(51)
        self.spinBox_filter_size.setSingleStep(2)
        self.spinBox_filter_size.setObjectName("spinBox_filter_size")
        self.horizontalLayout_5.addWidget(self.spinBox_filter_size)
        self.layoutWidget1 = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 70, 141, 31))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.doubleSpinBox_sigma = QtWidgets.QDoubleSpinBox(self.layoutWidget1)
        self.doubleSpinBox_sigma.setObjectName("doubleSpinBox_sigma")
        self.horizontalLayout_4.addWidget(self.doubleSpinBox_sigma)
        self.layoutWidget2 = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget2.setGeometry(QtCore.QRect(210, 70, 141, 31))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_3.addWidget(self.label_6)
        self.doubleSpinBox_mu = QtWidgets.QDoubleSpinBox(self.layoutWidget2)
        self.doubleSpinBox_mu.setObjectName("doubleSpinBox_mu")
        self.horizontalLayout_3.addWidget(self.doubleSpinBox_mu)
        self.layoutWidget3 = QtWidgets.QWidget(self.groupBox_2)
        self.layoutWidget3.setGeometry(QtCore.QRect(210, 30, 141, 31))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.doubleSpinBox_low = QtWidgets.QDoubleSpinBox(self.layoutWidget3)
        self.doubleSpinBox_low.setMaximum(1.0)
        self.doubleSpinBox_low.setSingleStep(0.01)
        self.doubleSpinBox_low.setObjectName("doubleSpinBox_low")
        self.horizontalLayout_2.addWidget(self.doubleSpinBox_low)
        self.widget = QtWidgets.QWidget(self.tab_filters)
        self.widget.setGeometry(QtCore.QRect(200, 10, 791, 361))
        self.widget.setObjectName("widget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_filters_input = QtWidgets.QLabel(self.widget)
        self.label_filters_input.setFrameShape(QtWidgets.QFrame.Box)
        self.label_filters_input.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_filters_input.setObjectName("label_filters_input")
        self.gridLayout_5.addWidget(self.label_filters_input, 0, 0, 1, 1)
        self.label_filters_output = QtWidgets.QLabel(self.widget)
        self.label_filters_output.setFrameShape(QtWidgets.QFrame.Box)
        self.label_filters_output.setTextFormat(QtCore.Qt.PlainText)
        self.label_filters_output.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_filters_output.setObjectName("label_filters_output")
        self.gridLayout_5.addWidget(self.label_filters_output, 0, 1, 1, 1)
        self.widget1 = QtWidgets.QWidget(self.tab_filters)
        self.widget1.setGeometry(QtCore.QRect(10, 140, 171, 131))
        self.widget1.setObjectName("widget1")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.widget1)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_2 = QtWidgets.QLabel(self.widget1)
        self.label_2.setObjectName("label_2")
        self.gridLayout_6.addWidget(self.label_2, 3, 0, 1, 1)
        self.pushButton_filters_load = QtWidgets.QPushButton(self.widget1)
        self.pushButton_filters_load.setObjectName("pushButton_filters_load")
        self.gridLayout_6.addWidget(self.pushButton_filters_load, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget1)
        self.label.setObjectName("label")
        self.gridLayout_6.addWidget(self.label, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tab_filters, "")
        self.tab_histograms = QtWidgets.QWidget()
        self.tab_histograms.setObjectName("tab_histograms")
        self.widget2 = QtWidgets.QWidget(self.tab_histograms)
        self.widget2.setGeometry(QtCore.QRect(280, 10, 701, 531))
        self.widget2.setObjectName("widget2")
        self.gridLayout = QtWidgets.QGridLayout(self.widget2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_histograms_input = QtWidgets.QLabel(self.widget2)
        self.label_histograms_input.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_input.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_input.setObjectName("label_histograms_input")
        self.gridLayout.addWidget(self.label_histograms_input, 0, 0, 1, 1)
        self.label_histograms_output = QtWidgets.QLabel(self.widget2)
        self.label_histograms_output.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_output.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_output.setObjectName("label_histograms_output")
        self.gridLayout.addWidget(self.label_histograms_output, 0, 1, 1, 1)
        self.label_histograms_hinput = QtWidgets.QLabel(self.widget2)
        self.label_histograms_hinput.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_hinput.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_hinput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_hinput.setObjectName("label_histograms_hinput")
        self.gridLayout.addWidget(self.label_histograms_hinput, 1, 0, 1, 1)
        self.label_histograms_houtput = QtWidgets.QLabel(self.widget2)
        self.label_histograms_houtput.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_houtput.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_houtput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_houtput.setObjectName("label_histograms_houtput")
        self.gridLayout.addWidget(self.label_histograms_houtput, 1, 1, 1, 1)
        self.widget3 = QtWidgets.QWidget(self.tab_histograms)
        self.widget3.setGeometry(QtCore.QRect(40, 200, 201, 131))
        self.widget3.setObjectName("widget3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget3)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_Auto_global_thershold = QtWidgets.QPushButton(self.widget3)
        self.pushButton_Auto_global_thershold.setObjectName("pushButton_Auto_global_thershold")
        self.gridLayout_2.addWidget(self.pushButton_Auto_global_thershold, 0, 0, 1, 1)
        self.pushButton_globalThershold = QtWidgets.QPushButton(self.widget3)
        self.pushButton_globalThershold.setObjectName("pushButton_globalThershold")
        self.gridLayout_2.addWidget(self.pushButton_globalThershold, 1, 0, 1, 1)
        self.doubleSpinBox_Thershold_value = QtWidgets.QDoubleSpinBox(self.widget3)
        self.doubleSpinBox_Thershold_value.setMaximum(1.0)
        self.doubleSpinBox_Thershold_value.setSingleStep(0.01)
        self.doubleSpinBox_Thershold_value.setProperty("value", 0.5)
        self.doubleSpinBox_Thershold_value.setObjectName("doubleSpinBox_Thershold_value")
        self.gridLayout_2.addWidget(self.doubleSpinBox_Thershold_value, 2, 0, 1, 1)
        self.widget4 = QtWidgets.QWidget(self.tab_histograms)
        self.widget4.setGeometry(QtCore.QRect(40, 350, 201, 131))
        self.widget4.setObjectName("widget4")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget4)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_local_thershold = QtWidgets.QPushButton(self.widget4)
        self.pushButton_local_thershold.setObjectName("pushButton_local_thershold")
        self.gridLayout_3.addWidget(self.pushButton_local_thershold, 0, 0, 1, 2)
        self.spinBox_thershlod_filter_size = QtWidgets.QSpinBox(self.widget4)
        self.spinBox_thershlod_filter_size.setMinimum(3)
        self.spinBox_thershlod_filter_size.setMaximum(41)
        self.spinBox_thershlod_filter_size.setSingleStep(2)
        self.spinBox_thershlod_filter_size.setProperty("value", 7)
        self.spinBox_thershlod_filter_size.setObjectName("spinBox_thershlod_filter_size")
        self.gridLayout_3.addWidget(self.spinBox_thershlod_filter_size, 1, 0, 1, 1)
        self.comboBox_thersholding_filter = QtWidgets.QComboBox(self.widget4)
        self.comboBox_thersholding_filter.setObjectName("comboBox_thersholding_filter")
        self.comboBox_thersholding_filter.addItem("")
        self.comboBox_thersholding_filter.addItem("")
        self.comboBox_thersholding_filter.addItem("")
        self.gridLayout_3.addWidget(self.comboBox_thersholding_filter, 1, 1, 1, 1)
        self.widget5 = QtWidgets.QWidget(self.tab_histograms)
        self.widget5.setGeometry(QtCore.QRect(40, 20, 201, 171))
        self.widget5.setObjectName("widget5")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget5)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pushButton_histograms_load = QtWidgets.QPushButton(self.widget5)
        self.pushButton_histograms_load.setObjectName("pushButton_histograms_load")
        self.gridLayout_4.addWidget(self.pushButton_histograms_load, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.widget5)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.widget5)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tab_histograms, "")
        self.tab_hybrid = QtWidgets.QWidget()
        self.tab_hybrid.setObjectName("tab_hybrid")
        self.widget6 = QtWidgets.QWidget(self.tab_hybrid)
        self.widget6.setGeometry(QtCore.QRect(180, 20, 811, 521))
        self.widget6.setObjectName("widget6")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.widget6)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_histograms_input_2 = QtWidgets.QLabel(self.widget6)
        self.label_histograms_input_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_input_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_input_2.setObjectName("label_histograms_input_2")
        self.gridLayout_7.addWidget(self.label_histograms_input_2, 0, 0, 1, 1)
        self.label_histograms_output_2 = QtWidgets.QLabel(self.widget6)
        self.label_histograms_output_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_output_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_output_2.setObjectName("label_histograms_output_2")
        self.gridLayout_7.addWidget(self.label_histograms_output_2, 0, 1, 2, 1)
        self.label_histograms_hinput_2 = QtWidgets.QLabel(self.widget6)
        self.label_histograms_hinput_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_hinput_2.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_hinput_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_hinput_2.setObjectName("label_histograms_hinput_2")
        self.gridLayout_7.addWidget(self.label_histograms_hinput_2, 1, 0, 1, 1)
        self.widget7 = QtWidgets.QWidget(self.tab_hybrid)
        self.widget7.setGeometry(QtCore.QRect(20, 20, 131, 121))
        self.widget7.setObjectName("widget7")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.widget7)
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.pushButton_histograms_load_2 = QtWidgets.QPushButton(self.widget7)
        self.pushButton_histograms_load_2.setObjectName("pushButton_histograms_load_2")
        self.gridLayout_8.addWidget(self.pushButton_histograms_load_2, 0, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.widget7)
        self.label_12.setObjectName("label_12")
        self.gridLayout_8.addWidget(self.label_12, 1, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.widget7)
        self.label_13.setObjectName("label_13")
        self.gridLayout_8.addWidget(self.label_13, 2, 0, 1, 1)
        self.widget8 = QtWidgets.QWidget(self.tab_hybrid)
        self.widget8.setGeometry(QtCore.QRect(20, 170, 131, 131))
        self.widget8.setObjectName("widget8")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.widget8)
        self.gridLayout_9.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.pushButton_histograms_load_3 = QtWidgets.QPushButton(self.widget8)
        self.pushButton_histograms_load_3.setObjectName("pushButton_histograms_load_3")
        self.gridLayout_9.addWidget(self.pushButton_histograms_load_3, 0, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.widget8)
        self.label_15.setObjectName("label_15")
        self.gridLayout_9.addWidget(self.label_15, 1, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.widget8)
        self.label_14.setObjectName("label_14")
        self.gridLayout_9.addWidget(self.label_14, 2, 0, 1, 1)
        self.widget9 = QtWidgets.QWidget(self.tab_hybrid)
        self.widget9.setGeometry(QtCore.QRect(30, 350, 131, 81))
        self.widget9.setObjectName("widget9")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.widget9)
        self.gridLayout_10.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_8 = QtWidgets.QLabel(self.widget9)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_6.addWidget(self.label_8)
        self.doubleSpinBox_alpha = QtWidgets.QDoubleSpinBox(self.widget9)
        self.doubleSpinBox_alpha.setMinimum(0.1)
        self.doubleSpinBox_alpha.setMaximum(0.9)
        self.doubleSpinBox_alpha.setSingleStep(0.05)
        self.doubleSpinBox_alpha.setProperty("value", 0.5)
        self.doubleSpinBox_alpha.setObjectName("doubleSpinBox_alpha")
        self.horizontalLayout_6.addWidget(self.doubleSpinBox_alpha)
        self.gridLayout_10.addLayout(self.horizontalLayout_6, 0, 0, 1, 1)
        self.pushButton_histograms_load_4 = QtWidgets.QPushButton(self.widget9)
        self.pushButton_histograms_load_4.setObjectName("pushButton_histograms_load_4")
        self.gridLayout_10.addWidget(self.pushButton_histograms_load_4, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_hybrid, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1043, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Filter Settings"))
        self.label_3.setText(_translate("MainWindow", "Select Filter"))
        self.comboBox_filters.setItemText(0, _translate("MainWindow", "Tab to Select"))
        self.comboBox_filters.setItemText(1, _translate("MainWindow", "Gaussian noise"))
        self.comboBox_filters.setItemText(2, _translate("MainWindow", "Uniform noise"))
        self.comboBox_filters.setItemText(3, _translate("MainWindow", "Salt & pepper noise"))
        self.comboBox_filters.setItemText(4, _translate("MainWindow", "Gaussian filter"))
        self.comboBox_filters.setItemText(5, _translate("MainWindow", "Median filter"))
        self.comboBox_filters.setItemText(6, _translate("MainWindow", "Average filter"))
        self.comboBox_filters.setItemText(7, _translate("MainWindow", "Sobel ED"))
        self.comboBox_filters.setItemText(8, _translate("MainWindow", "Roberts ED"))
        self.comboBox_filters.setItemText(9, _translate("MainWindow", "Prewitt ED"))
        self.comboBox_filters.setItemText(10, _translate("MainWindow", "Canny ED"))
        self.comboBox_filters.setItemText(11, _translate("MainWindow", "Laplacian of Gaussian"))
        self.comboBox_filters.setItemText(12, _translate("MainWindow", "Laplacian1"))
        self.comboBox_filters.setItemText(13, _translate("MainWindow", "Laplacian2"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Additional Parameters"))
        self.label_4.setText(_translate("MainWindow", "filter size"))
        self.label_5.setText(_translate("MainWindow", "sigma"))
        self.label_6.setText(_translate("MainWindow", "mu"))
        self.label_7.setText(_translate("MainWindow", "low"))
        self.label_filters_input.setText(_translate("MainWindow", "Input image"))
        self.label_filters_output.setText(_translate("MainWindow", "Output image"))
        self.label_2.setText(_translate("MainWindow", "Size:"))
        self.pushButton_filters_load.setText(_translate("MainWindow", "Load Image"))
        self.label.setText(_translate("MainWindow", "Name:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_filters), _translate("MainWindow", "Filters"))
        self.label_histograms_input.setText(_translate("MainWindow", "Input image"))
        self.label_histograms_output.setText(_translate("MainWindow", "Output image"))
        self.label_histograms_hinput.setText(_translate("MainWindow", "Input Histogram"))
        self.label_histograms_houtput.setText(_translate("MainWindow", "Output Histogram"))
        self.pushButton_Auto_global_thershold.setText(_translate("MainWindow", "Auto Global Thersholding"))
        self.pushButton_globalThershold.setText(_translate("MainWindow", "Global Thersholding"))
        self.pushButton_local_thershold.setText(_translate("MainWindow", "Local Thersholding"))
        self.comboBox_thersholding_filter.setItemText(0, _translate("MainWindow", "Gaussian"))
        self.comboBox_thersholding_filter.setItemText(1, _translate("MainWindow", "Average"))
        self.comboBox_thersholding_filter.setItemText(2, _translate("MainWindow", "Median"))
        self.pushButton_histograms_load.setText(_translate("MainWindow", "Load image"))
        self.label_11.setText(_translate("MainWindow", "Name:"))
        self.label_10.setText(_translate("MainWindow", "Size:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_histograms), _translate("MainWindow", "Histograms"))
        self.label_histograms_input_2.setText(_translate("MainWindow", "Input image A"))
        self.label_histograms_output_2.setText(_translate("MainWindow", "Output image"))
        self.label_histograms_hinput_2.setText(_translate("MainWindow", "Input image B"))
        self.pushButton_histograms_load_2.setText(_translate("MainWindow", "Load image A"))
        self.label_12.setText(_translate("MainWindow", "Name:"))
        self.label_13.setText(_translate("MainWindow", "Size:"))
        self.pushButton_histograms_load_3.setText(_translate("MainWindow", "Load image B"))
        self.label_15.setText(_translate("MainWindow", "Name:"))
        self.label_14.setText(_translate("MainWindow", "Size:"))
        self.label_8.setText(_translate("MainWindow", "Alpha"))
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
