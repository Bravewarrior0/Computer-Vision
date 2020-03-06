import numpy as np
import CV404Filters as myFilters
import cv2

def histogram(img):
    img = img.flatten()
    histo = np.zeros(256, np.uint8)
    for i in range(len(img)):
            histo[img[i]] += 1
    return histo

def equalization(img):
    equalized_img = img.flatten()
    img_histo = histogram(img)
    total_num = img.size
    lvl =255
    pdf = np.zeros(256, np.float)
    cdf = np.zeros(256, np.float)
    rounded=np.zeros(256, np.uint8)
    equalized= np.zeros(256, np.uint8)
    for i in range(len(pdf)):
        pdf[i] = img_histo[i]/total_num

    cdf = np.cumsum(pdf)
    for j in range (len(cdf)):
        rounded[j] = round(lvl*cdf[j]) 
    
    for k in range(len(rounded)):
        equalized_img[k] = rounded[k]

    return equalized_img.reshape(len(img),len(img[0]))

# --------------thersholding----------------------------#
def threshold_global(gray_image , threshold =0.5 ):
    threshold = np.max( gray_image ) * threshold
    return 1 * ( gray_image > threshold )

def threshold_global_auto(gray_image):
    threshold = np.sum(gray_image)/(gray_image.size)
    return 1 * ( gray_image > threshold )

def threshold_local(img, filter_size = 11, filter_type ='gaussian'):
    '''
    filter_type can be 'gaussian', 'average', 'median'
    '''
    filter_type = filter_type.lower()
    thershold_img = None
    if(filter_type == 'average'):
        thershold_img  = myFilters.average_filter(img,filter_size)
    # elif(filter_type is 'median'):
    #     thershold_img = myFilters.median_filter(img, filter_size)
    else:
        thershold_img = myFilters.img_gaussian_filter(img, filter_size)
    return 1*(img > thershold_img)