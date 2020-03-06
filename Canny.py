import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import CV404Filters as myFilters

def non_max_suppression(mag, theta):
    result = np.zeros_like(mag)
    theta[theta < 0] += np.pi
    m, n = mag.shape
    for i in range (1, m-1):
        for j in range(1,n-1):
            f = b = 1.0
            if(0<=theta[i,j]<=np.pi/8) or (0.875*np.pi <=theta[i,j] <=np.pi):
                f =mag[i+1,j]
                b = mag [i-1,j]
            elif(np.pi/8<=theta[i,j]<=np.pi*0.375):
                f = mag[i+1,j-1]
                b = mag[i+1,j-1]
            elif(np.pi*0.375<=theta[i,j]<=np.pi*0.652):
                f = mag[i,j+1]
                b = mag[i,j-1]
            elif(np.pi*0.652<=theta[i,j]<=0.875*np.pi):
                f = mag[i-1,j-1]
                b = mag[i+1,j+1]

            # print(f,b,mag[i,j])
            if(mag[i,j]>=f and mag[i,j]>= b):
                result[i,j]=mag[i,j]
    return result

def max_threshold(img, TL=0.02, TH=0.09, weak = 0.4, strong = 1):
    highThreshold = img.max() * TH
    lowThreshold = img.max() * TL
    result = np.zeros_like(img)
    result[img >= highThreshold] = strong
    result[img < lowThreshold] = 0
    result[(img <= highThreshold) & (img >= lowThreshold)]= weak
    return result

def hysteresis(img, weak= 0.4, strong=1):
    m, n = img.shape
    x = np.array([1,0,1,-1,1,-1, 0,-1])
    y = np.array([0,1,1,1,-1,-1,-1,0])
    for i in range(1, m-1):
        for j in range(1, n-1):
            if (img[i,j] == weak):
                img[i,j]=0
                for k in range(8):
                    if img[i+x[k], j+y[k]]==strong:
                        img[i, j] = strong
                        break
    return img

def canny(img, TL=0.02, TH=0.09, weak = 0.4, strong = 1):
    img = myFilters.img_gaussian_filter(img,5)
    mag, angle = myFilters.sobel1(img)
    result = non_max_suppression(mag, angle)
    result =max_threshold(result, TL, TH, weak, strong)
    return hysteresis(result, weak, strong)
