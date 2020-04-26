# this will include the implementation for template matching functions
import numpy as np
from scipy.signal import correlate2d
from CV404Filters import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import time

# Correlation
def template_match_corr( img , temp ):
    out = np.empty(img.shape)
    out = correlate2d(img,temp,'same')
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

def maximum_filter_ignore_nan(array, size):
    nans = np.isnan(array)
    replaced = np.where(nans, -np.inf, array)
    return filters.maximum_filter(replaced, size)

def local_maxima( x , size, threshold = 0.55 ):
    x_max = maximum_filter_ignore_nan(x,size)
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



def template_match(img, temp, method = 'corr', threshold = 0.55, n = 8):
    '''
    img - > image that U want to match template in

    temp -> template image

    method -> one of four: 
            ('corr')-> Correlation-> default

            ('zmean')-> Zero-mean correlation

            ('ssd')-> Sum of squared differences

            ('xcorr')-> Normalized cross correlations

    theshold -> thershold of local maxima to find the best matches 0->1

    n -> int increase it for more matching result

    return

    ------------------------------
    matching_space, detected_patterns, elapsed_time -> the matching computation time for the method

    '''
    img_g = rgb2gray(img)
    temp_g = rgb2gray(temp)
    mcorr = None
    start = time.perf_counter()
    if method == 'zmean':
        mcorr = template_match_corr_zmean( img_g , temp_g )
    elif method == 'ssd':
        mcorr = template_match_ssd( img_g , temp_g )
    elif method == 'xcorr':
        mcorr = template_match_xcorr( img_g , temp_g )
    else:
        mcorr = template_match_corr(img_g , temp_g )

    elapsed_time = time.perf_counter()-start

    pcorr = local_maxima(mcorr,min(temp_g.shape)//n)

    w,h = np.array(img_g.shape)/100
    fig_match = plt.figure(figsize=(w,h))
    ax_match = fig_match.add_axes([0,0,1,1])
    fig_img = plt.figure(figsize=(w,h))
    
    ax_img = fig_img.add_axes([0,0,1,1])
    ax_img.imshow(img,cmap='gray')
    ax_img.autoscale(False)
    ax_img.axis('off')

    r,x,y = get_rect_on_maximum(mcorr,temp_g)
    make_rects(ax_img , pcorr, temp_g )
    ax_img.add_patch(r)

    ax_match.imshow(mcorr, cmap = 'gray')
    ax_match.autoscale(False)
    ax_match.axis('off')
    make_circles(ax_match, pcorr,temp)
    ax_match.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10) 

    fig_match.canvas.draw()
    fig_img.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    matching_space = np.array(fig_match.canvas.renderer.buffer_rgba())
    detected_patterns = np.array(fig_img.canvas.renderer.buffer_rgba())

    plt.close('all') 

    return matching_space, detected_patterns, elapsed_time
