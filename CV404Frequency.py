import numpy as np
from scipy import ndimage, signal
import CV404Filters as filters
import cv2 as cv

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


def highPassFilter (img):

    img_float32 = np.float32(img)

    dft = cv2.dft(np.float32(img_float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2) 

    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 0


    fshift = dft_shift * mask

    # fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

    f_ishift = np.fft.ifftshift(fshift)
    highPassFilter_img = cv2.idft(f_ishift)
    highPassFilter_img = cv2.magnitude(highPassFilter_img[:, :, 0], highPassFilter_img[:, :, 1])


    cv.normalize(highPassFilter_img, highPassFilter_img, 0, 1, cv.NORM_MINMAX) 

    return highPassFilter_img


def lowPassFilter(img):

    img_float32 = np.float32(img)

    dft = cv2.dft(np.float32(img_float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

    fshift = dft_shift*mask

    f_ishift = np.fft.ifftshift(fshift)
    lowPassFilter_img = cv2.idft(f_ishift)
    lowPassFilter_img = cv2.magnitude(lowPassFilter_img[:,:,0],lowPassFilter_img[:,:,1])

    cv.normalize(lowPassFilter_img, lowPassFilter_img, 0, 1, cv.NORM_MINMAX) 

    return lowPassFilter_img
