import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage, signal
from skimage import filters
import CV404Filters as filters

def hybrid(img1, img2, alpha =0.5,shape = 13, filterType='lp2'):
    '''
    img1 : Low pass filter
    img2: High pass filter
    0<alpha < 1
    low pass * alpha , high pass*(1-alpha)
    filterType for High pass default lp2
    'lp2' = laplacian_using_gaussian
    'lp1' = laplacian
    'prewitt' = prewitt
    'soble' = soble
    'roberts' = roberts
    '''
    if(0 >=alpha >=1):
        alpha =0.5
    low = filters.img_gaussian_filter(img1,shape)
    high = None
    if(filterType is 'lp1'):
        high = filters.img_laplacian_filter(img2)
    elif(filterType is 'soble'):
        high = filters.sobel(img2)
    elif(filterType is 'prewitt'):
        high = filters.prewitt(img2)
    elif(filterType is 'roberts'):
        high = filters.roberts_edge_detection(img2)
    else:
        gaussain = filters.img_gaussian_filter(img2,shape)
        high = img2 - gaussain
    high = filters.img_map(high)
    low = filters.img_map(low)
    high = filters.padding_img((1-alpha)*high, low.shape)
    low = filters.padding_img(alpha*low, high.shape)
    out = (filters.img_map(low+high))
    return out
