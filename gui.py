from PyQt5 import QtWidgets
from MainWindow import Ui_MainWindow
import sys
import CV404Filters as backend
import CV404Histograms as hg
import CV404Frequency as freq
from PyQt5 import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QFileDialog
from qtpy.QtGui import QPixmap
import qimage2ndarray
from functools import partial

class ApplicationWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,parent=None):
        super(ApplicationWindow, self).__init__(parent) 
        self.setupUi(self)

        self.low = self.doubleSpinBox_low.value()
        self.filterSize = self.spinBox_filter_size.value()
        self.mu = self.doubleSpinBox_mu.value()
        self.sigma = self.doubleSpinBox_sigma.value()
        self.alpha = self.doubleSpinBox_alpha.value()

        self.comboBox_filters.currentTextChanged.connect(self.combo_selection)
        self.pushButton_filters_load.clicked.connect(self.filters_load_btn) 
        self.pushButton_histograms_load.clicked.connect(self.histo_load_btn)
        self.pushButton_histograms_load_2.clicked.connect(self.imgA_load_btn)
        self.pushButton_histograms_load_3.clicked.connect(self.imgB_load_btn)
        self.pushButton_histograms_load_4.clicked.connect(self.hybrid)
        self.doubleSpinBox_low.valueChanged.connect(self.saltNpepperFun)
        self.spinBox_filter_size.valueChanged.connect(self.FilterSize)
        self.doubleSpinBox_sigma.valueChanged.connect(self.sigmaFun)
        self.doubleSpinBox_mu.valueChanged.connect(self.muFun)
        self.doubleSpinBox_alpha.valueChanged.connect(self.alphaFun)
    def alphaFun(self):
        self.alpha= self.doubleSpinBox_alpha.value()
        self.hybrid()
    def muFun(self):
        self.mu = self.doubleSpinBox_mu.value()
        self.gaussiaNoise()
    def sigmaFun(self):
        self.sigma = self.doubleSpinBox_sigma.value()
        if self.comboBox_filters.currentText() == 'Gaussian noise':
            self.gaussiaNoise()
        if self.comboBox_filters.currentText() == 'Gaussian filter':
            self.gaussianFilter()         
    def saltNpepperFun(self):
        self.low = self.doubleSpinBox_low.value()
        if self.comboBox_filters.currentText() == 'Salt & pepper noise':
            self.saltNpepper()
    def FilterSize(self):
        self.filterSize = self.spinBox_filter_size.value()
        if self.comboBox_filters.currentText() == 'Median filter':
            self.medianFilter()  
        if self.comboBox_filters.currentText() == 'Average filter':  
            self.averageFilter()
        if self.comboBox_filters.currentText() == 'Gaussian filter': 
            self.gaussianFilter()  

    def hybrid(self):
        hybrid_img = freq.hybrid(self.hyb1, self.hyb2, self.alpha)
        self.getImageFromArray(hybrid_img,self.label_histograms_output_2)
        
    def imgB_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.hyb2, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(self.hyb2)
            pixmap = pixmap.scaled(self.label_histograms_hinput_2.width(),self.label_histograms_hinput_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_hinput_2.setPixmap(pixmap)
            
        except Exception as err:
            print(err)      
    def imgA_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.hyb1, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(self.hyb1)
            pixmap = pixmap.scaled(self.label_histograms_input_2.width(),self.label_histograms_input_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_input_2.setPixmap(pixmap)
            
        except Exception as err:
            print(err)      
    def histo_load_btn(self):
        try:   
            options = QFileDialog.Options()
            self.histo_fileName, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(self.histo_fileName)
            pixmap = pixmap.scaled(self.label_histograms_input.width(),self.label_histograms_input.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_input.setPixmap(pixmap)
            
        except Exception as err:
            print(err)       
    def filters_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.fileName, _ = QFileDialog.getOpenFileName(None, 'Upload Image', '', '*.png *.jpg *.jpeg',options=options)
            pixmap = QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.label_filters_input.width(),self.label_filters_input.height(), QtCore.Qt.KeepAspectRatio)
            self.label_filters_input.setPixmap(pixmap)
            self.comboBox_filters.setEnabled(True)
            self.label_filters_output.clear()
        except Exception as err:
            print(err)
    

    def getImageFromArray(self,array,outlabel):
        qimg = qimage2ndarray.array2qimage(array, normalize=True)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(outlabel.width(),outlabel.height(), QtCore.Qt.KeepAspectRatio)
        outlabel.setPixmap(pixmap)
    

    def combo_selection(self):
        value = self.comboBox_filters.currentText()
        if value == 'Average filter':
           self.averageFilter()

        elif value == 'Gaussian filter':
           self.gaussianFilter()

        elif value == 'Median filter':
            self.medianFilter()

        elif value == 'Uniform noise':
            img = backend.uniformNoise(self.fileName)
            self.getImageFromArray(img,self.label_filters_output)
    
        elif value == 'Gaussian noise':
            self.gaussiaNoise()

        elif value == 'Salt & pepper noise':
           self.saltNpepper()

        elif value == 'Sobel ED':
            img = backend.sobel(self.fileName)
            self.getImageFromArray(img,self.label_filters_output)

        elif value == 'Roberts ED':
            img = backend.roberts_edge_detection(self.fileName)
            self.getImageFromArray(img,self.label_filters_output)

        elif value == 'Prewitt ED':
            img = backend.prewitt(self.fileName)
            self.getImageFromArray(img,self.label_filters_output)

        elif value == 'Canny ED':
            pass    
         
    def saltNpepper (self):
        img = backend.saltNpepper(self.fileName, self.low)
        self.getImageFromArray(img,self.label_filters_output)  
    def medianFilter(self):
        img = backend.median_filter(self.fileName,self.filterSize)
        self.getImageFromArray(img,self.label_filters_output)  
    def gaussiaNoise(self):
        img = backend.add_gaussian_noise(self.mu,self.sigma,self.fileName)
        self.getImageFromArray(img,self.label_filters_output)   
    def averageFilter(self):
        img = backend.average_filter(self.fileName,self.filterSize)
        self.getImageFromArray(img,self.label_filters_output)     
    def gaussianFilter(self):
        img = backend.img_gaussian_filter(self.fileName,self.filterSize,self.sigma)
        self.getImageFromArray(img,self.label_filters_output)
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()