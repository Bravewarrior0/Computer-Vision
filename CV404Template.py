# this will include the implementation for template matching functions
import numpy as np
from scipy.signal import correlate2d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

# Correlation
def template_match_corr( img , temp ):
    out = np.empty(img.shape)
    out = correlate2d(x,temp,'same')
    return out
# Zero-mean correlation
def template_match_corr_zmean( img , temp ):
    return template_match_corr(img , temp - temp.mean())

# Sum of squared differences (SSD)
def template_match_ssd( img , temp ):
    term1 = np.sum( np.square( temp ))
    term2 = -2*correlate2d(img, temp,'same')
    term3 = correlate2d( np.square( img ), np.ones(temp.shape),'same' )
    ssd = np.maximum( term1 + term2 + term3 , 0 )
    return 1 - np.sqrt(ssd)

# Normalized cross correlations.
def template_match_xcorr( img , temp):
    f_c = img - correlate2d( img , np.ones(temp.shape)/np.prod(temp.shape), 'same') 
    t_c = temp - temp.mean()
    numerator = correlate2d( f_c , t_c , 'same' )
    d1 = correlate2d( np.square(f_c) , np.ones(temp.shape), 'same')
    d2 = np.sum( np.square( t_c ))
    denumerator = np.sqrt( np.maximum( d1 * d2 , 0 )) # to avoid sqrt of negative
    response = np.zeros( img.shape )
    valid = denumerator > np.finfo(np.float32).eps # mask to avoid division by zero
    response[valid] = numerator[valid]/denumerator[valid]
    return response

# def rgb2gray(rgb_image):
#     # convert RGB img to grayScale img
#     return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

def maximum_filter_ignore_nan(array, size):
    nans = np.isnan(array)
    replaced = np.where(nans, -np.inf, array)
    return filters.maximum_filter(replaced, size)

def local_maxima( x , size, threshold = 0.55 ):
    x_max = maximum_filter_ignore_nan(x,size)
#    x_min = minimum_filter_ignore_nan(x,size)
#    maxima = (x_max-x_min) >= (np.amax( x_max-x_min) * threshold)
    maxima = (x_max) >= (np.amax( x_max) * threshold)
    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(x, labeled, range(1, num_objects+1)))
    return xy

def get_rect_on_maximum(y,template):
    ij = np.unravel_index(np.argmax(y), y.shape)
    x , y = ij[::-1]
    # highlight matched region
    htemp, wtemp = template.shape
    rect = plt.Rectangle((x-wtemp/2, y-htemp/2), wtemp, htemp, edgecolor='r', facecolor='none')
    return rect,x,y
def make_rects(plt_object,xy,template):
    htemp, wtemp = template.shape
    for ridx in range(xy.shape[0]):
        y,x = xy[ridx]
        r =  plt.Rectangle((x-wtemp/2, y-htemp/2), wtemp, htemp, edgecolor='g', facecolor='none')
        plt_object.add_patch(r)
   
def make_circles(plt_object,xy,template):
    for ridx in range(xy.shape[0]):
        y,x = xy[ridx]
        plt_object.plot(x, y, 'o', markeredgecolor='g', markerfacecolor='none', markersize=10)

# x = rgb2gray(mpimg.imread('chess.jpg'))
# temp = rgb2gray(mpimg.imread('chess_temp.jpg'))
# mcorr = template_match_corr_zmean( x , temp )
# pcorr = local_maxima(mcorr,min(temp.shape)//8)

# fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()

# ax.imshow(x,cmap='gray')
# ax.autoscale(False)
# ax.axis('off')

# # imgplot = plt.imshow(out,cmap=plt.get_cmap('gray'))
# # plt.show()
# r,x,y = get_rect_on_maximum(mcorr,temp)
# make_rects( ax , pcorr, temp )
# ax.add_patch(r)
# # print(type(maxima))
# # out = threshold_local(out, min(temp.shape))

# ax1.imshow(mcorr, cmap = 'gray')
# ax1.autoscale(False)
# ax1.axis('off')
# make_circles(ax1, pcorr,temp)
# ax1.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)  


# plt.show()