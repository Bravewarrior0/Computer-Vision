import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage, signal
from skimage import filters
import random 
import cv2 as cv

def rgb2gray(rgb_image):
    # convert RGB img to grayScale img
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

def make_odd(num = 3):
    num = num//2 *2 + 1
    return num

def convolve_img (img, kernal):
    if len(img.shape) == 3:
        out = np.zeros_like(img)
        for i in range(3):
            out[:,:,i]= signal.convolve2d(img[:,:,i],kernal,mode='same')
        return out
    elif len(img.shape) == 2:
        return signal.convolve2d(img,kernal,mode='same')

def padding_img(img,size =[0,0]):
    size = np.asarray(size)
    if(img.shape[0]>size[0]):
        size[0] = img.shape[0]
    if(img.shape[1]>size[1]):
        size[1] = img.shape[1]
    out = np.zeros(size)
    x_offset = np.abs(size[0]-img.shape[0])//2
    y_offset = np.abs(size[1]-img.shape[1])//2
    out[x_offset:img.shape[0]+x_offset,y_offset:img.shape[1]+y_offset] = img
    return out

def img_map_gray(img):
    if(img.min()<0):
        img-=img.min()
    img = img / img.max()
    return img
    
def img_map(img):
    if len(img.shape) == 3:
        for i in range(3):
            img[:,:,i]= img_map_gray(img[:,:,i])*255
    elif len(img.shape) == 2:
        img = img_map_gray(img)
    return img
# -------------------------------------------------
# ----------------- filters -----------------------
# -------------------------------------------------
def gaussian_filter(shape = 3, sigma ='auto'):
    # generate gaussian kernal
    shape = make_odd(shape)
    m = n = shape//2
    filter = np.zeros((shape,shape))
    if sigma is 'auto':
        sigma = sigma=np.sqrt(2*m+1)
    for x in range (-m, m+1):
        for y in range (-m, m+1):
            filter[x+m, y+m] =np.exp(-(x**2 + y**2)/(2*sigma**2))/(2*np.pi*(sigma**2))
    sum = filter.sum()
    filter = filter * (1/sum)
    return filter

def img_gaussian_filter(img, shape= 3, sigma='auto'):
    kernal = gaussian_filter(shape, sigma)
    return convolve_img(img,kernal)

def roberts_edge_detection(img):
    # output = np.sqrt(roberts_H_edge_detection(img)**2 + roberts_V_edge_detection(img)**2)
    # it gives the same output but it should be faster
    output = np.abs(roberts_H_edge_detection(img))+np.abs(roberts_V_edge_detection(img))
    return output

def roberts_H_edge_detection(img):
    ROBERTS_H_MASK= np.array([[1,0],[0,-1]])
    return convolve_img(img,ROBERTS_H_MASK)

def roberts_V_edge_detection(img):
    ROBERTS_V_MASK= np.array([[0,1],[-1,0]])
    return convolve_img(img,ROBERTS_V_MASK)

def laplacian_of_gaussian (shape = 3, sigma='auto'):
    shape = make_odd(shape)
    m= n = shape//2
    filter = np.zeros((shape, shape))
    if sigma is 'auto':
        sigma = np.sqrt(2*m+1)
    for x in range (-m, m+1):
        for y in range (-m, m+1):
            val = -(x**2 + y**2)/(2*sigma**2)
            filter[x+m, y+m] =np.exp(val)*(1+val)/(-1*np.pi*(sigma**4))
    return filter
def img_laplacian_of_gaussian(img, shape = 3, sigma='auto'):
    kernal = laplacian_of_gaussian(shape,sigma)
    return convolve_img(img,kernal)

def img_laplacian_filter(img):
    kernal = [[1,1,1],[1,-8,1],[1,1,1]]
    out = img_map(convolve_img(img, kernal))
    return  out

def laplacian_using_gaussian(img,shape = 3, sigma='auto'):
    filter = img_gaussian_filter(img,shape, sigma)
    return img - filter

def prewitt(img):
    vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    horizontal = vertical.transpose()
    hGrad = convolve_img(img, horizontal)
    vGrad = convolve_img(img, vertical)
    magnitude = np.sqrt(pow(hGrad, 2.0) + pow(vGrad, 2.0))
    # direction = np.arctan2(vGrad, hGrad)
    # magnitude /= np.max(magnitude)
    # hGrad /= np.max(hGrad)
    # vGrad /= np.max(vGrad)
    return img_map(magnitude)

def sobel1(img):
    vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hGrad = convolve_img(img, horizontal)
    vGrad = convolve_img(img, vertical)
    magnitude = np.sqrt(pow(hGrad, 2.0) + pow(vGrad, 2.0))
    direction = np.arctan2(vGrad, hGrad)
    magnitude /= np.max(magnitude)
    # hGrad /= np.max(hGrad)
    # vGrad /= np.max(vGrad)
    return magnitude , direction

def sobel(img):
    magnitude , direction = sobel1(img)
    return img_map(magnitude)

def canny(img):
    edges = cv2.Canny(img,100,200)
    return canny

def median_filter(img, filter_size):
    index = filter_size // 2
    filtered = np.zeros(img.shape, np.uint8)

    row, col = img.shape

    for i in range(row):
        for j in range(col):
            temp = []
            for z in range(filter_size):
                if i + z - index < 0 or i + z - index > img.shape[0] - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - index < 0 or j + index > img.shape[1] - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(img[i + z - index][j + k - index])

            temp.sort()
            filtered[i][j] = temp[len(temp) // 2]

    return filtered

def average_filter(img, filter_size = 3):
    filter_size = make_odd(filter_size)
    kernal = np.ones((filter_size,filter_size))/(filter_size**2)
    return convolve_img(img,kernal)
# -------------------------------------------------
# ------------------- Noise -----------------------
# -------------------------------------------------
def generate_gaussian_noise( mu, sigma, img_size ):
    #generrate random Gaussian noise array 
    return np.random.normal( mu, sigma, img_size)

def add_gaussian_noise(mu, sigma, img):
    #add randam Gaussian noise to the img
    gaussian_noise= generate_gaussian_noise(mu,sigma, img.shape)
    img_with_gaussian_noise = img + gaussian_noise
    return img_with_gaussian_noise

def saltNpepper(img, low):
    high = 1-low
    output = np.zeros(img.shape, np.uint8)
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            rndm = random.random()
            if rndm < low:
                output[i][j] = 0
            elif rndm > high:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


def uniformNoise (img):
    uniformNoise = np.zeros(img.shape, np.uint8)
    row, col = img.shape

    for x in range (row):
        for y in range (col):
            uniformNoise[x][y] = (random.uniform(0,255) + img[x][y]) / 2 
    return uniformNoise

