from PyQt5 import QtWidgets
from MainWindow import Ui_MainWindow
import sys
import CV404Filters as backend
import CV404Histograms as hg
import CV404Frequency as freq
import CV404Harris as harris
import CV404Hough as hough
import CV404ActiveContour as ac
from PyQt5 import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QFileDialog
from qtpy.QtGui import QPixmap
import qimage2ndarray
from functools import partial
import matplotlib.image as mpimg
import Canny
import numpy as np
import cv2
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from PyQt5.QtGui import QPainter, QBrush, QPen
from scipy import ndimage, signal, interpolate
from PyQt5.QtCore import Qt

class ApplicationWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ApplicationWindow, self).__init__(parent)
        self.setupUi(self)
        self.arr = []
        self.AC = ""
       
        self.alpha_AC =self.doubleSpinBox_AC_Alpha.value()
        self.beta_AC =self.doubleSpinBox_AC_Beta.value()
        self.gamma_AC =self.doubleSpinBox_AC_Gamma.value()
        self.segma_ac = self.doubleSpinBox_sigma_AC.value()
        self.harris_fileName = None
        self.hough_fileName = None
        self.histo_fileName = 'images\Bikesgray.jpg'
        self.low = self.doubleSpinBox_low.value()
        self.filterSize = self.spinBox_filter_size.value()
        self.mu = self.doubleSpinBox_mu.value()
        self.sigma = self.doubleSpinBox_sigma.value()
        self.alpha = self.doubleSpinBox_alpha.value()
        self.threshold_value = self.doubleSpinBox_Thershold_value.value()
        self.thershlod_filter_size = self.spinBox_thershlod_filter_size.value()
        self.thershold_filter_type = self.comboBox_thersholding_filter.currentText()
        self.effect = self.comboBox_hybrid.currentText()
        self.comboBox_filters.currentTextChanged.connect(self.combo_selection)
        self.comboBox_thersholding_filter.currentTextChanged.connect(
            self.changeThersholdFilterType)
        self.pushButton_filters_load.clicked.connect(self.filters_load_btn)
        self.pushButton_histograms_load.clicked.connect(self.histo_load_btn)
        self.pushButton_histograms_load_2.clicked.connect(self.imgA_load_btn)
        self.pushButton_histograms_load_3.clicked.connect(self.imgB_load_btn)
        self.pushButton_histograms_load_4.clicked.connect(self.hybrid)
        self.pushButton_globalThershold.clicked.connect(
            self.globalThershold_btn)
        self.pushButton_Auto_global_thershold.clicked.connect(
            self.autoGlobalThershold_btn)
        self.pushButton_local_thershold.clicked.connect(
            self.localThershold_btn)
        self.doubleSpinBox_low.valueChanged.connect(self.saltNpepperFun)
        self.spinBox_filter_size.valueChanged.connect(self.FilterSize)
        self.spinBox_thershlod_filter_size.valueChanged.connect(
            self.changeThersholdFilterSize)
        self.doubleSpinBox_sigma.valueChanged.connect(self.sigmaFun)
        self.doubleSpinBox_mu.valueChanged.connect(self.muFun)
        self.doubleSpinBox_alpha.valueChanged.connect(self.alphaFun)
        self.doubleSpinBox_Thershold_value.valueChanged.connect(
            self.changeThersholdValue)
        self.pushButton_frequency_load.clicked.connect(self.frequency_load)
        self.comboBox_pass_filter.currentTextChanged.connect(
            self.combo_pass_filter)
        self.comboBox_hybrid.currentTextChanged.connect(self.hybrid_effect)

        #Snakes
        self.pushButton_AC_load.clicked.connect(self.AC_load)
        self.doubleSpinBox_sigma_AC.valueChanged.connect(self.sigma_AC)
        self.comboBox_AC.currentTextChanged.connect(self.ED_AC)
        self.pushButton_AC_apply.clicked.connect(self.AC_execution)

        # Harris tab
        self.pushButton_harris_load.clicked.connect(self.harris_load_btn)
        self.pushButton_harris_apply.clicked.connect(self.harris_apply_btn)

        # Hough
        self.pushButton_hough_load.clicked.connect(self.hough_load_btn)
        self.pushButton_hough_apply.clicked.connect(self.hough_apply_btn)

    def AC_execution(self):
        self.radius = ((self.center[0] - self.tip[0]) ** 2 + (self.center[1]-self.tip[1])**2)**.5
        self.theta=np.linspace(0, 2*np.pi, 200) # min, max, number of divisions
        self.x_AC=self.center[0]+self.radius*np.cos(self.theta)
        self.y_AC=self.center[1]+self.radius*np.sin(self.theta)
        self.x_rep_AC = ac.circ_replicate(self.x_AC)
        self.y_rep_AC = ac.circ_replicate(self.y_AC)
        self.newContour=ac.compute_energy(self.x_rep_AC,self.y_rep_AC, self.alpha_AC, self.beta_AC, self.gamma_AC,self.img_norm)
        self.drawNewCont(self.newContour)

    def drawNewCont(self, points):
       
        x = points[0]
        y = points[1]
        painter = QPainter(self.label_AC.pixmap())
        painter.setPen(QPen(Qt.red,  1, Qt.SolidLine))   
        for i in range (len(x)):
            painter.drawPoint(x[i], y[i])
        painter.end()
        self.update()   

    def ED_AC(self):
        self.total_settings_AC()  
    def sigma_AC (self):
        self.total_settings_AC()
    def min_Canny_AC(self):
        self.total_settings_AC()
    def max_Canny_AC(self):
        self.total_settings_AC()
    def total_settings_AC(self):

        self.imgFiltered=signal.convolve2d(self.img_AC, ac.gaussian_Filter_AC(self.segma_ac, (3,3)), mode='same')
        value = self.comboBox_AC.currentText()
        if value == 'Canny ED':  
            self.img_norm = cv2.Canny( self.img_AC, self.Td_Low_AC.value() , self.Td_High_AC.value())
            #self.img_norm=-backend.normalize(self.img_grad,0,1)
        elif value == 'Sobel ED':
            self.edgeX = backend.sobel_h(self.imgFiltered)
            self.edgeY = backend.sobel_v(self.imgFiltered)
            self.img_grad=np.hypot(self.edgeX,self.edgeY)
            self.img_norm=-ac.normalize(self.img_grad,0,1)
        
        self.getImageFromArray( self.img_norm, self.label_AC)
    def AC_load(self):
        try:
            options = QFileDialog.Options()
            self.AC, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            self.img_AC = self.getGrayImage(self.AC)    
            
            self.total_settings_AC()
            

        except Exception as err:
            print(err)

    def mousePressEvent(self, e):
        if self.AC != "" and ( ( (e.x() > self.label_AC.x()) and ( e.x() < self.label_AC.x()+self.label_AC.width() ) ) and ( ( e.y() > self.label_AC.y()) and ( e.y() < self.label_AC.y()+self.label_AC.height())) ):
            try:    
                painter = QPainter(self.label_AC.pixmap())
                mappedPoint = self.label_AC.mapFromParent(e.pos())
                self.arr.append([mappedPoint.x(), mappedPoint.y()])
                painter.setPen(QPen(Qt.blue,  1, Qt.DashLine))
                painter.drawPoint(mappedPoint.x(), mappedPoint.y())
                #painter.drawPoint(e.x(), e.y())
                if len(self.arr) % 2 == 0 and len(self.arr) != 0:
                    self.center = self.arr[-2]
                    self.tip = self.arr[-1]
                    self.radius = ((self.center[0] - self.tip[0])
                                   ** 2 + (self.center[1]-self.tip[1])**2)**.5
                    #painter.drawEllipse(self.label_AC.mapFromParent(e.pos()) , 2*self.radius, 2*self.radius)
                    painter.drawEllipse(
                        self.center[0]-self.radius, self.center[1]-self.radius, 2*self.radius, 2*self.radius)
                    #print(self.arr[-1][0], " || " , self.arr[-1][1])
                painter.end()
                self.update()
            except Exception as err:
                print(err)

    def hybrid_effect(self):
        self.effect = self.comboBox_pass_filter.currentText()
        self.hybrid()

    def alphaFun(self):
        self.alpha = self.doubleSpinBox_alpha.value()
        self.hybrid()

    def muFun(self):
        value = self.comboBox_filters.currentText()
        self.mu = self.doubleSpinBox_mu.value()
        if value == 'Gaussian noise':
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

    def frequency_load(self):
        try:
            options = QFileDialog.Options()
            self.freq, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            pixmap = QPixmap(self.freq)
            pixmap = pixmap.scaled(self.label_pass_input.width(
            ), self.label_pass_input.height(), QtCore.Qt.KeepAspectRatio)
            self.label_pass_input.setPixmap(pixmap)
            self.label_pass_output.clear()
        except Exception as err:
            print(err)

    def imgB_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.hyb2, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            pixmap = QPixmap(self.hyb2)
            pixmap = pixmap.scaled(self.label_histograms_hinput_2.width(
            ), self.label_histograms_hinput_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_hinput_2.setPixmap(pixmap)

        except Exception as err:
            print(err)

    def imgA_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.hyb1, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            pixmap = QPixmap(self.hyb1)
            pixmap = pixmap.scaled(self.label_histograms_input_2.width(
            ), self.label_histograms_input_2.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_input_2.setPixmap(pixmap)

        except Exception as err:
            print(err)

    def harris_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.harris_fileName, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            pixmap = QPixmap(self.harris_fileName)
            if(not pixmap.isNull()):
                pixmap = pixmap.scaled(self.label_harris_input.width(
                ), self.label_harris_input.height(), QtCore.Qt.KeepAspectRatio)
                self.label_harris_input.setPixmap(pixmap)
        except Exception as err:
            print(err)
    #hough load
    def hough_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.hough_fileName, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            pixmap = QPixmap(self.hough_fileName)
            if(not pixmap.isNull()):
                pixmap = pixmap.scaled(self.label_hough_in.width(
                ), self.label_hough_in.height(), QtCore.Qt.KeepAspectRatio)
                self.label_hough_in.setPixmap(pixmap)
        except Exception as err:
            print(err)

    #hough apply
    def hough_apply_btn(self):
        if(self.hough_fileName == None):
            return
        if(self.radioButton_line.isChecked()):
            img = self.getImage(self.hough_fileName)
            thershold = self.doubleSpinBox_hough_thershold.value()
            out = hough.get_hough_lines(img,thershold)
            self.getImageFromArray(out, self.label_hough_out)
            # origin = np.array((0, img.shape[1]))
            # painter = QPainter(self.label_hough_out.pixmap())
            # pen = QPen()
            # pen.setColor(QtGui.QColor('red'))
            # painter.setPen(pen)
            # for _, angle, dist in zip(*hough.get_hough_lines(img)):
            #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            #     painter.drawLine(int(origin[0]),int(origin[1]), int(y0), int(y1))
            # painter.end()
            # self.update()
        
    def equalization_histograms(self, img):
        bins = np.arange(257)
        #img = self.getGrayImage(self.histo_fileName)
        histoNormal = hg.histogram(img)
        equalized_array = hg.equalization(img)
        histoEqualized = hg.histogram(equalized_array)
        self.getImageFromArray(equalized_array, self.label_histograms_output)
        self.histo_input.plot(bins+1, histoNormal,
                              stepMode=True, fillLevel=0, brush='r')
        self.histo_output.plot(bins+1, histoEqualized,
                               stepMode=True, fillLevel=0, brush='b')

    def histo_load_btn(self):
        current = self.histo_fileName
        try:
            options = QFileDialog.Options()
            self.histo_fileName, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            pixmap = QPixmap(self.histo_fileName)
            pixmap = pixmap.scaled(self.label_histograms_input.width(
            ), self.label_histograms_input.height(), QtCore.Qt.KeepAspectRatio)
            self.label_histograms_input.setPixmap(pixmap)
            img = self.getGrayImage(self.histo_fileName)
            self.equalization_histograms(img)

        except Exception as err:
            self.histo_fileName = current
            print(err)

    def filters_load_btn(self):
        try:
            options = QFileDialog.Options()
            self.fileName, _ = QFileDialog.getOpenFileName(
                None, 'Upload Image', '', '*.png *.jpg *.jpeg', options=options)
            pixmap = QPixmap(self.fileName)
            pixmap = pixmap.scaled(self.label_filters_input.width(
            ), self.label_filters_input.height(), QtCore.Qt.KeepAspectRatio)
            self.label_filters_input.setPixmap(pixmap)
            self.comboBox_filters.setEnabled(True)
            self.label_filters_output.clear()
        except Exception as err:
            print(err)

    def harris_apply_btn(self):
        if(self.harris_fileName == None):
            return
        img = self.getImage(self.harris_fileName)
        k = self.doubleSpinBox_harris_k.value()
        w = self.spinBox_harris_w.value()
        gaussian_size = self.spinBox_haris_gaussianSize.value()
        thershold = self.doubleSpinBox_harris_thershold.value()
        out = harris.get_corners(img, w, k, thershold, gaussian_size)
        self.getImageFromArray(out, self.label_harris_output)

    def autoGlobalThershold_btn(self):
        img = self.getGrayImage(self.histo_fileName)
        thersholdImg = hg.threshold_global_auto(img)
        self.getImageFromArray(thersholdImg, self.label_histograms_output)

    def globalThershold_btn(self):
        img = self.getGrayImage(self.histo_fileName)
        thersholdImg = hg.threshold_global(img, self.threshold_value)
        self.getImageFromArray(thersholdImg, self.label_histograms_output)

    def localThershold_btn(self):
        img = self.getGrayImage(self.histo_fileName)
        thersholdImg = hg.threshold_local(
            img, self.thershlod_filter_size, self.thershold_filter_type)
        self.getImageFromArray(thersholdImg, self.label_histograms_output)

    def changeThersholdValue(self):
        self.threshold_value = self.doubleSpinBox_Thershold_value.value()
        self.globalThershold_btn()

    def changeThersholdFilterSize(self):
        self.thershlod_filter_size = self.spinBox_thershlod_filter_size.value()
        self.localThershold_btn()

    def changeThersholdFilterType(self):
        self.thershold_filter_type = self.comboBox_thersholding_filter.currentText()

    def hybrid(self):
        hybrid_img = freq.hybrid(self.hyb1, self.hyb2,
                                 self.alpha, 13, self.effect)
        self.getImageFromArray(hybrid_img, self.label_histograms_output_2)

    def Lpass(self):
        img_array = freq.lowPassFilter(self.getGrayImage(self.freq))
        self.getImageFromArray(img_array, self.label_pass_output)

    def Hpass(self):
        img_array = freq.highPassFilter(self.getGrayImage(self.freq))
        self.getImageFromArray(img_array, self.label_pass_output)

    def combo_pass_filter(self):
        value = self.comboBox_pass_filter.currentText()
        if value == 'Low pass':
            self.Lpass()
        elif value == 'High pass':
            self.Hpass()
        elif value == 'Tab to select':
            self.label_pass_output.clear()

    def combo_selection(self):
        value = self.comboBox_filters.currentText()
        if value == 'Average filter':
            self.averageFilter()

        elif value == 'Gaussian filter':
            self.gaussianFilter()

        elif value == 'Median filter':
            self.medianFilter()

        elif value == 'Uniform noise':
            self.uniformNoise()

        elif value == 'Gaussian noise':
            self.gaussiaNoise()

        elif value == 'Salt & pepper noise':
            self.saltNpepper()

        elif value == 'Sobel ED':
            self.sobelED()

        elif value == 'Roberts ED':
            self.robertsED()

        elif value == 'Prewitt ED':
            self.prewittED()

        elif value == 'Canny ED':
            self.cannyED()

        elif value == 'Laplacian2':
            self.laplacian2()

        elif value == 'Laplacian1':
            self.laplacian1()

        elif value == 'Laplacian of Gaussian':
            self.laplacianGaussian()

        elif value == "Tab to Select":
            self.label_filters_output.clear()

    def saltNpepper(self):
        img = self.getGrayImage(self.fileName)
        img = backend.saltNpepper(img, self.low)
        self.getImageFromArray(img, self.label_filters_output)

    def medianFilter(self):
        img = self.getGrayImage(self.fileName)
        img = backend.median_filter(img, self.filterSize)
        self.getImageFromArray(img, self.label_filters_output)

    def gaussiaNoise(self):
        img = self.getGrayImage(self.fileName)
        img = backend.add_gaussian_noise(self.mu, self.sigma, img)
        self.getImageFromArray(img, self.label_filters_output)

    def averageFilter(self):
        img = self.getImage(self.fileName)
        img = backend.average_filter(img, self.filterSize)
        self.getImageFromArray(img, self.label_filters_output)

    def gaussianFilter(self):
        img = self.getImage(self.fileName)
        img = backend.img_gaussian_filter(img, self.filterSize, self.sigma)
        self.getImageFromArray(img, self.label_filters_output)

    def prewittED(self):
        img = self.getGrayImage(self.fileName)
        img = backend.prewitt(img)
        self.getImageFromArray(img, self.label_filters_output)

    def robertsED(self):
        img = self.getGrayImage(self.fileName)
        img = backend.roberts_edge_detection(img)
        self.getImageFromArray(img, self.label_filters_output)

    def uniformNoise(self):
        img = self.getGrayImage(self.fileName)
        img = backend.uniformNoise(img)
        self.getImageFromArray(img, self.label_filters_output)

    def cannyED(self):
        img = self.getGrayImage(self.fileName)
        img = Canny.canny(img)
        self.getImageFromArray(img, self.label_filters_output)

    def sobelED(self):
        img = self.getGrayImage(self.fileName)
        img = backend.sobel(img)
        self.getImageFromArray(img, self.label_filters_output)

    def laplacianGaussian(self):
        img = self.getGrayImage(self.fileName)
        img = backend.img_laplacian_of_gaussian(
            img, self.filterSize, self.sigma)
        self.getImageFromArray(img, self.label_filters_output)

    def laplacian2(self):
        img = self.getGrayImage(self.fileName)*255
        img = backend.laplacian_using_gaussian(img, self.filterSize)
        self.getImageFromArray(img, self.label_filters_output)

    def laplacian1(self):
        img = self.getGrayImage(self.fileName)
        img = backend.img_laplacian_filter(img)
        self.getImageFromArray(img, self.label_filters_output)

    def getImage(self, path):
        img = mpimg.imread(path)
        return img

    def getGrayImage(self, path):
        #img = backend.rgb2gray(mpimg.imread(path))
        return backend.rgb2gray(cv2.imread(path)).astype(np.uint8)

    def getImageFromArray(self, array, outlabel):
        qimg = qimage2ndarray.array2qimage(array, normalize=True)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            outlabel.width(), outlabel.height(), QtCore.Qt.KeepAspectRatio)
        outlabel.setPixmap(pixmap)


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()
